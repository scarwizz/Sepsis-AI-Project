"""
Unified Full-Cohort Evaluation Pipeline for Held-Out Test Set.
Evaluates:
- Generalization on unseen Test partition using optimal threshold tuned on Validation set
- Standard Discriminatory Metrics: AUROC, AUPRC
- Clinical Metrics: Sensitivity, Specificity, F1, PPV, NPV
- Official PhysioNet 2019 Normalized Utility Score (U_norm)
- Patient-level Alarm Lead-Time analysis
- Vectorized 95% Bootstrap Confidence Intervals (200 patient-level resamples in ~10s)
"""

import os
import sys
import argparse
import logging
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed
from src.utils.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    compute_physionet_utility,
    compute_lead_time_analysis,
    compute_patient_bootstrap_ci
)
from src.data.preprocess import CausalPreprocessPipeline
from src.data.features import compute_causal_rolling_features, compute_causal_derivatives
from src.data.dataset import (
    reconstruct_and_label_horizons,
    patient_level_split,
    SepsisSlidingWindowDataset
)
from src.models.tcn import MultiTCN, TCNWithAttention
from src.models.hybrids import TCNLSTMHybrid, HybridSepsisModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluatePipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sepsis Prediction Models on Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to raw CSV")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs", help="Output directory for reports")
    parser.add_argument("--max_patients", type=int, default=0, help="Max patients (0 = full cohort)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--bootstrap", action="store_true", default=True, help="Compute 95% bootstrap confidence intervals")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    device_str = "cuda" if torch.cuda.is_available() and args.device in ["auto", "cuda"] else "cpu"
    device = torch.device(device_str)
    logger.info(f"Loading checkpoint on {device}: {args.checkpoint}")
    ckpt_data = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg = ckpt_data["config"]
    horizon = ckpt_data["horizon"]
    seq_len = ckpt_data["sequence_length"]
    feature_cols = ckpt_data["feature_cols"]
    optimal_threshold = ckpt_data.get("optimal_threshold", 0.5)
    model_type = ckpt_data.get("model_type", "model")
    base_vitals = ckpt_data.get("base_vitals", ["HR", "O2Sat", "Temp", "Resp", "SBP", "DBP"])

    seed = cfg["seed"]["global_seed"] if "seed" in cfg else 42
    set_seed(seed)

    data_path = args.data_path or cfg["data"]["raw_csv_path"]
    if not os.path.exists(data_path):
        data_path = cfg["data"].get("fallback_csv_path", "data/cleaned_sepsis_dataset.csv")

    logger.info(f"Loading evaluation dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)

    ratios = (
        cfg["data"]["split_ratios"]["train"],
        cfg["data"]["split_ratios"]["val"],
        cfg["data"]["split_ratios"]["test"]
    )
    max_pats = args.max_patients if args.max_patients > 0 else ckpt_data.get("max_patients", 0)
    max_pats = max_pats if max_pats > 0 else None

    train_df, val_df, test_df = patient_level_split(
        raw_df,
        patient_id_col=cfg["data"]["patient_id_col"],
        label_col=cfg["data"]["label_col"],
        split_ratios=ratios,
        random_seed=seed,
        max_patients=max_pats
    )

    preprocessor = CausalPreprocessPipeline(
        feature_cols=base_vitals,
        patient_id_col=cfg["data"]["patient_id_col"],
        time_col=cfg["data"]["time_col"],
        label_col=cfg["data"]["label_col"]
    )
    train_df = preprocessor.fit_transform(train_df)
    test_df = preprocessor.transform(test_df)

    rolling_cols = preprocessor.feature_cols
    test_df = compute_causal_rolling_features(test_df, vitals=rolling_cols, windows=cfg["features"]["rolling_windows"])
    if cfg["features"]["include_derivatives"]:
        test_df = compute_causal_derivatives(test_df, vitals=rolling_cols)

    target_col = f"Target_{horizon}h"
    test_df = reconstruct_and_label_horizons(test_df, prediction_horizon=horizon)

    test_dataset = SepsisSlidingWindowDataset(
        test_df, feature_cols=feature_cols, target_col=target_col,
        sequence_length=seq_len, is_train=False
    )
    logger.info(f"Test sliding windows to evaluate: {len(test_dataset):,} across {test_df['Patient_ID'].nunique():,} patients")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    num_inputs = len(feature_cols)

    if model_type == "tcn":
        model = MultiTCN(vital_signs=num_inputs, num_channels=cfg["models"]["tcn"]["num_channels"], kernel_size=cfg["models"]["tcn"]["kernel_size"])
    elif model_type == "tcn_attention":
        model = TCNWithAttention(vital_signs=num_inputs, num_channels=cfg["models"]["tcn_attention"]["num_channels"], kernel_size=cfg["models"]["tcn_attention"]["kernel_size"])
    elif model_type == "hybrid_tcn_lstm":
        model = TCNLSTMHybrid(num_inputs=num_inputs, tcn_num_channels=cfg["models"]["hybrid_tcn_lstm"]["tcn_channels"], kernel_size=cfg["models"]["hybrid_tcn_lstm"]["kernel_size"])
    elif model_type == "hybrid_signature":
        model = HybridSepsisModel(
            num_inputs=num_inputs,
            sig_channels=len(base_vitals),
            tcn_channels=cfg["models"]["hybrid_signature"]["tcn_channels"],
            kernel_size=cfg["models"]["hybrid_signature"]["kernel_size"],
            mlp_hidden_size=cfg["models"]["hybrid_signature"]["mlp_hidden_size"],
            dropout=cfg["models"]["hybrid_signature"]["dropout"],
            output_size=1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.eval()

    use_cuda = (device.type == "cuda")
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch[0].to(device)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(X)
            p = torch.sigmoid(outputs).cpu().numpy().flatten()
            test_probs.extend(p.tolist())

    meta = test_dataset.get_metadata()
    test_pids, test_hours, test_labels = meta[0], meta[1], meta[2]
    raw_sepsis = meta[3] if len(meta) > 3 else test_labels

    y_true = np.array(test_labels)
    y_probs = np.array(test_probs)

    # 1. Window-level classification metrics
    std_metrics = compute_classification_metrics(y_true, y_probs, threshold=0.5)
    opt_metrics = compute_classification_metrics(y_true, y_probs, threshold=optimal_threshold)

    # 2. DataFrame for patient-level evaluation
    eval_df = pd.DataFrame({
        "Patient_ID": test_pids,
        "Hour": test_hours,
        "Target": y_true,
        "Probability": y_probs,
        "Prediction_05": (y_probs >= 0.5).astype(int),
        "Prediction_Opt": (y_probs >= optimal_threshold).astype(int),
        "SepsisLabel": raw_sepsis.astype(int)
    })

    # 3. Official PhysioNet Utility Score
    utility_results_std = compute_physionet_utility(eval_df, pred_label_col="Prediction_05")
    utility_results_opt = compute_physionet_utility(eval_df, pred_label_col="Prediction_Opt")

    # 4. Lead-Time Analysis
    lead_time_results_std = compute_lead_time_analysis(eval_df, pred_label_col="Prediction_05")
    lead_time_results_opt = compute_lead_time_analysis(eval_df, pred_label_col="Prediction_Opt")

    # 5. Bootstrap 95% Confidence Intervals
    bootstrap_results = {}
    if args.bootstrap:
        logger.info("Computing 200 patient-level bootstrap resamples for 95% CIs...")
        bootstrap_results = compute_patient_bootstrap_ci(
            eval_df, pred_label_col="Prediction_Opt", n_bootstraps=200, seed=seed
        )
        logger.info(f"AUROC 95% CI: [{bootstrap_results['auroc_ci_95'][0]:.4f}, {bootstrap_results['auroc_ci_95'][1]:.4f}]")
        logger.info(f"AUPRC 95% CI: [{bootstrap_results['aucpr_ci_95'][0]:.4f}, {bootstrap_results['aucpr_ci_95'][1]:.4f}]")
        logger.info(f"Utility 95% CI: [{bootstrap_results['utility_ci_95'][0]:.4f}, {bootstrap_results['utility_ci_95'][1]:.4f}]")

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "horizon_hours": horizon,
        "sequence_length": seq_len,
        "test_windows": len(y_true),
        "test_patients": int(eval_df["Patient_ID"].nunique()),
        "bootstrap_ci_95": bootstrap_results,
        "metrics_threshold_0_5": {**std_metrics, **utility_results_std, **lead_time_results_std},
        "metrics_threshold_optimal": {
            "tuned_threshold": float(optimal_threshold),
            **opt_metrics,
            **utility_results_opt,
            **lead_time_results_opt
        }
    }

    print("\n" + "=" * 80)
    print(f"[CLINICAL EVALUATION] {model_type.upper()} | HORIZON: {horizon}h | TEST SET (FULL COHORT)")
    print("=" * 80)
    print(f"AUROC (95% CI)      : {std_metrics['auroc']:.4f} [{bootstrap_results.get('auroc_ci_95', (0,0))[0]:.4f} - {bootstrap_results.get('auroc_ci_95', (0,0))[1]:.4f}]")
    print(f"AUPRC (95% CI)      : {std_metrics['aucpr']:.4f} [{bootstrap_results.get('aucpr_ci_95', (0,0))[0]:.4f} - {bootstrap_results.get('aucpr_ci_95', (0,0))[1]:.4f}]")
    print(f"Optimal Threshold   : {optimal_threshold:.2f}")
    print(f"Optimal F1          : {opt_metrics['f1_score']:.4f} | Recall: {opt_metrics['recall_sensitivity']:.4f} | Spec: {opt_metrics['specificity']:.4f}")
    print(f"PhysioNet Utility   : {utility_results_opt['physionet_utility_norm']:+.4f} [{bootstrap_results.get('utility_ci_95', (0,0))[0]:+.4f} - {bootstrap_results.get('utility_ci_95', (0,0))[1]:+.4f}]")
    print(f"Mean Lead Time      : {lead_time_results_opt['mean_lead_time_hours']:.2f} Hours")
    print(f"False Alarms/100h   : {lead_time_results_opt['false_alarm_rate_per_100h']:.2f}")
    print("=" * 80)

    csv_out = os.path.join(args.output_dir, f"test_predictions_{model_type}_h{horizon}.csv")
    eval_df.to_csv(csv_out, index=False)
    
    json_out = os.path.join(args.output_dir, f"test_report_{model_type}_h{horizon}.json")
    with open(json_out, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Predictions saved to: {csv_out}")
    logger.info(f"Evaluation report saved to: {json_out}")


if __name__ == "__main__":
    main()
