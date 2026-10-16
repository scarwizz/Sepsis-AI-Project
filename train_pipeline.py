"""
Unified Full-Cohort Training Pipeline CLI for Sepsis Onset Prediction.
Supports:
- Full 40,336-patient cohort with 100% septic case retention
- Models: 'tcn', 'tcn_attention', 'hybrid_tcn_lstm', 'hybrid_signature'
- Multi-horizon early prediction: H in [4, 6, 10, 12, 18] hours
- Strict causal ffill + train population fallback + standardization + hemodynamic channels
- 70/15/15 Patient-level disjoint splitting
- CosineAnnealingLR scheduling and gradient clipping (max_norm=1.0)
- Validation-driven early stopping and bounded clinical threshold tuning
"""

import os
import sys
import time
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.utils.seed import set_seed
from src.utils.losses import FocalLoss, FocalTverskyLoss
from src.utils.metrics import compute_classification_metrics, find_optimal_threshold
from src.data.preprocess import CausalPreprocessPipeline
from src.data.features import compute_causal_rolling_features, compute_causal_derivatives
from src.data.dataset import (
    reconstruct_and_label_horizons,
    patient_level_split,
    SepsisSlidingWindowDataset,
    create_dataloaders
)
from src.models.tcn import MultiTCN, TCNWithAttention
from src.models.hybrids import TCNLSTMHybrid, HybridSepsisModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Full-Cohort Sepsis Onset Prediction Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config yaml")
    parser.add_argument("--model", type=str, default="hybrid_signature", choices=["tcn", "tcn_attention", "hybrid_tcn_lstm", "hybrid_signature"], help="Model architecture")
    parser.add_argument("--horizon", type=int, default=6, help="Prediction horizon in hours before onset (e.g. 4, 6, 10, 12, 18)")
    parser.add_argument("--seq_len", type=int, default=14, help="Time-series sequence window length")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--max_patients", type=int, default=0, help="Max patients (0 = full 40,336 cohort). Retains 100% of septic patients.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience epochs on validation AUPRC")
    parser.add_argument("--neg_stride", type=int, default=3, help="Training-only stride for non-septic control stays")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset CSV")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Deterministic Seeding
    seed = cfg["seed"]["global_seed"]
    set_seed(seed=seed, deterministic_cudnn=cfg["seed"]["deterministic_cudnn"])
    logger.info(f"Initialized deterministic seed: {seed}")

    # 2. Locate Data
    data_path = args.data_path or cfg["data"]["raw_csv_path"]
    if not os.path.exists(data_path):
        fallback = cfg["data"].get("fallback_csv_path", "data/cleaned_sepsis_dataset.csv")
        if os.path.exists(fallback):
            data_path = fallback
        else:
            logger.error(f"FATAL: Dataset not found at {data_path} or {fallback}")
            sys.exit(1)

    logger.info(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    total_pats = raw_df["Patient_ID"].nunique()
    total_septic = raw_df.groupby("Patient_ID")["SepsisLabel"].max().sum()
    logger.info(f"Loaded full dataset: {len(raw_df):,} rows | {total_pats:,} patients ({total_septic:,} septic).")

    # 3. Patient-Level Disjoint Cohort Partitioning (70/15/15)
    ratios = (
        cfg["data"]["split_ratios"]["train"],
        cfg["data"]["split_ratios"]["val"],
        cfg["data"]["split_ratios"]["test"]
    )
    max_pats = args.max_patients if args.max_patients > 0 else None
    logger.info(f"Partitioning cohort by Patient_ID with ratios {ratios} (max_patients={max_pats})...")
    train_df, val_df, test_df = patient_level_split(
        raw_df,
        patient_id_col=cfg["data"]["patient_id_col"],
        label_col=cfg["data"]["label_col"],
        split_ratios=ratios,
        random_seed=seed,
        max_patients=max_pats
    )
    logger.info(f"Cohort Split -> Train Patients: {train_df['Patient_ID'].nunique():,}, "
                f"Val Patients: {val_df['Patient_ID'].nunique():,}, "
                f"Test Patients: {test_df['Patient_ID'].nunique():,}")

    base_vitals = ["HR", "O2Sat", "Temp", "Resp", "SBP", "DBP"]
    # Reorder base_vitals if present in dataset
    base_vitals = [v for v in ["HR", "O2Sat", "Temp", "Resp", "SBP", "DBP"] if v in train_df.columns]
    logger.info(f"Base 6 wearable vitals: {base_vitals}")

    # 4. Strictly Causal Preprocessing & Hemodynamic Derivation
    logger.info("Fitting CausalPreprocessPipeline on Training cohort...")
    preprocessor = CausalPreprocessPipeline(
        feature_cols=base_vitals,
        patient_id_col=cfg["data"]["patient_id_col"],
        time_col=cfg["data"]["time_col"],
        label_col=cfg["data"]["label_col"]
    )
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)
    logger.info(f"Derived Hemodynamic Channels added. Core physiological features: {preprocessor.feature_cols}")

    # 5. Causal Feature Engineering (Rolling statistics & 1h derivatives)
    logger.info("Computing causal rolling stats and derivatives...")
    rolling_cols = preprocessor.feature_cols
    train_df = compute_causal_rolling_features(train_df, vitals=rolling_cols, windows=cfg["features"]["rolling_windows"])
    val_df = compute_causal_rolling_features(val_df, vitals=rolling_cols, windows=cfg["features"]["rolling_windows"])
    test_df = compute_causal_rolling_features(test_df, vitals=rolling_cols, windows=cfg["features"]["rolling_windows"])

    if cfg["features"]["include_derivatives"]:
        train_df = compute_causal_derivatives(train_df, vitals=rolling_cols)
        val_df = compute_causal_derivatives(val_df, vitals=rolling_cols)
        test_df = compute_causal_derivatives(test_df, vitals=rolling_cols)

    # 6. Corrected Ground Truth Horizon Labeling
    horizon = args.horizon
    target_col = f"Target_{horizon}h"
    logger.info(f"Reconstructing true t_sepsis and labeling {horizon}h horizon: {target_col}")
    train_df = reconstruct_and_label_horizons(train_df, prediction_horizon=horizon)
    val_df = reconstruct_and_label_horizons(val_df, prediction_horizon=horizon)
    test_df = reconstruct_and_label_horizons(test_df, prediction_horizon=horizon)

    non_features = [cfg["data"]["patient_id_col"], cfg["data"]["time_col"], cfg["data"]["label_col"]]
    feature_cols = [c for c in train_df.columns if c not in non_features and not c.startswith("Target_")]
    logger.info(f"Total features per timestep: {len(feature_cols)}")

    # 7. Construct Sliding Window Datasets
    seq_len = args.seq_len
    logger.info(f"Generating sliding window datasets (sequence_length={seq_len}h)...")
    train_dataset = SepsisSlidingWindowDataset(
        train_df, feature_cols=feature_cols, target_col=target_col,
        sequence_length=seq_len, is_train=True, neg_stride=args.neg_stride
    )
    val_dataset = SepsisSlidingWindowDataset(
        val_df, feature_cols=feature_cols, target_col=target_col,
        sequence_length=seq_len, is_train=False
    )
    test_dataset = SepsisSlidingWindowDataset(
        test_df, feature_cols=feature_cols, target_col=target_col,
        sequence_length=seq_len, is_train=False
    )
    logger.info(f"Window Counts -> Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, f"best_{args.model}_h{horizon}_seq{seq_len}.pth")

    # Deep Learning Models (PyTorch)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "auto" else "cpu")
    logger.info(f"Target execution device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    use_cuda = (device.type == "cuda")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        use_weighted_sampler=True,
        max_train_samples=180000,
        seed=seed,
        pin_memory=False
    )

    num_inputs = len(feature_cols)
    if args.model == "tcn":
        model = MultiTCN(
            vital_signs=num_inputs,
            num_channels=cfg["models"]["tcn"]["num_channels"],
            output_size=1,
            kernel_size=cfg["models"]["tcn"]["kernel_size"],
            dropout=cfg["models"]["tcn"]["dropout"]
        )
    elif args.model == "tcn_attention":
        model = TCNWithAttention(
            vital_signs=num_inputs,
            num_channels=cfg["models"]["tcn_attention"]["num_channels"],
            output_size=1,
            kernel_size=cfg["models"]["tcn_attention"]["kernel_size"],
            attention_dim=cfg["models"]["tcn_attention"]["attention_dim"],
            dropout=cfg["models"]["tcn_attention"]["dropout"]
        )
    elif args.model == "hybrid_tcn_lstm":
        model = TCNLSTMHybrid(
            num_inputs=num_inputs,
            tcn_num_channels=cfg["models"]["hybrid_tcn_lstm"]["tcn_channels"],
            lstm_hidden_size=cfg["models"]["hybrid_tcn_lstm"]["lstm_hidden_size"],
            lstm_num_layers=cfg["models"]["hybrid_tcn_lstm"]["lstm_num_layers"],
            output_size=1,
            kernel_size=cfg["models"]["hybrid_tcn_lstm"]["kernel_size"],
            dropout=cfg["models"]["hybrid_tcn_lstm"]["dropout"]
        )
    elif args.model == "hybrid_signature":
        model = HybridSepsisModel(
            num_inputs=num_inputs,
            sig_channels=len(base_vitals),
            tcn_channels=cfg["models"]["hybrid_signature"]["tcn_channels"],
            kernel_size=cfg["models"]["hybrid_signature"]["kernel_size"],
            mlp_hidden_size=cfg["models"]["hybrid_signature"]["mlp_hidden_size"],
            output_size=1,
            dropout=cfg["models"]["hybrid_signature"]["dropout"]
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized {args.model} | Trainable parameters: {total_params:,}")

    # Loss selection
    if cfg["training"]["loss_type"] == "focal":
        criterion = FocalLoss(
            alpha=cfg["training"]["focal"]["alpha"],
            gamma=cfg["training"]["focal"]["gamma"]
        )
    elif cfg["training"]["loss_type"] == "focal_tversky":
        criterion = FocalTverskyLoss(
            alpha=cfg["training"]["tversky"]["alpha"],
            beta=cfg["training"]["tversky"]["beta"],
            gamma=cfg["training"]["tversky"]["gamma"]
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["training"]["bce"]["pos_weight"]], device=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    best_val_aucpr = -1.0
    best_opt_thresh = 0.5
    best_val_metrics = {}
    patience_counter = 0
    epochs = args.epochs

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} (lr={current_lr:.6f}, {num_batches} batches)...")
        
        for batch_idx, batch in enumerate(train_loader):
            X, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(X)
                loss = criterion(outputs, y)
            
            if torch.isnan(loss):
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation loop
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                X, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True).unsqueeze(1)
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(X)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_labels.extend(y.cpu().numpy().flatten())

        val_metrics = compute_classification_metrics(np.array(val_labels), np.array(val_probs), threshold=0.5)
        # Bounded clinical threshold optimization (constrained F1 with specificity >= 0.80)
        opt_thresh = find_optimal_threshold(
            np.array(val_labels), np.array(val_probs),
            method="constrained_f1", min_specificity=0.80, grid_min=0.10, grid_max=0.90, grid_step=0.02
        )
        val_metrics_opt = compute_classification_metrics(np.array(val_labels), np.array(val_probs), threshold=opt_thresh)
        total_epoch_time = time.time() - t_start

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{total_epoch_time:.1f}s] | Loss: {avg_train_loss:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | Val AUPRC: {val_metrics['aucpr']:.4f} | "
            f"Val F1 (0.50): {val_metrics['f1_score']:.4f} | Val F1 (opt @ {opt_thresh:.2f}): {val_metrics_opt['f1_score']:.4f}"
        )

        if val_metrics["aucpr"] > best_val_aucpr:
            best_val_aucpr = val_metrics["aucpr"]
            best_opt_thresh = opt_thresh
            best_val_metrics = val_metrics_opt
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "horizon": horizon,
                "sequence_length": seq_len,
                "feature_cols": feature_cols,
                "base_vitals": base_vitals,
                "num_inputs": num_inputs,
                "optimal_threshold": float(best_opt_thresh),
                "val_aucpr": float(best_val_aucpr),
                "val_auroc": float(val_metrics["auroc"]),
                "val_f1_opt": float(val_metrics_opt["f1_score"]),
                "max_patients": args.max_patients,
                "config": cfg
            }, model_save_path)
            logger.info(f"--> Saved champion checkpoint: {model_save_path} (AUPRC: {best_val_aucpr:.4f}, Thresh: {best_opt_thresh:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info(f"Training Complete! Champion model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
