"""
Master Neural Architecture & Horizon Ablation Benchmark.
Executes:
1. Architecture Ablation: 4 Models ('tcn', 'tcn_attention', 'hybrid_tcn_lstm', 'hybrid_signature')
   across Horizon H=6h and H=12h (8 runs).
2. Evaluates on held-out Test set using Validation-tuned optimal threshold.
3. Selects top-performing champion neural architecture.
4. Horizon Sensitivity Sweep: Runs champion on H=10h and H=18h (2 additional runs).
5. Compiles results into results/neural_ablation_results.csv and prints master tables.
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np

MODELS = ["tcn", "tcn_attention", "hybrid_tcn_lstm", "hybrid_signature"]
HORIZONS = [6, 12]
SEQ_LEN = 14
EPOCHS = 5
BATCH_SIZE = 256
MAX_PATIENTS = 2000

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
EVAL_DIR = "evaluation_outputs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def run_cmd(cmd_list):
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"[ERROR] Command failed with returncode {res.returncode}")
        print(res.stderr)
        sys.exit(1)
    print(res.stdout[-400:] if len(res.stdout) > 400 else res.stdout)


def parse_eval_report(model_name: str, horizon: int):
    json_path = os.path.join(EVAL_DIR, f"test_report_{model_name}_h{horizon}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing evaluation report: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    opt_m = data["metrics_threshold_optimal"]
    std_m = data["metrics_threshold_0_5"]

    return {
        "model": model_name,
        "horizon_hours": horizon,
        "sequence_length": SEQ_LEN,
        "test_windows": data.get("test_windows", 0),
        "test_patients": data.get("test_patients", 0),
        "test_auroc": round(opt_m["auroc"], 4),
        "test_aucpr": round(opt_m["aucpr"], 4),
        "val_tuned_threshold": round(opt_m["tuned_threshold"], 4),
        "test_f1_opt": round(opt_m["f1_score"], 4),
        "test_sensitivity_opt": round(opt_m["recall_sensitivity"], 4),
        "test_specificity_opt": round(opt_m["specificity"], 4),
        "test_precision_opt": round(opt_m["precision"], 4),
        "test_f1_05": round(std_m["f1_score"], 4),
        "physionet_utility_opt": round(opt_m.get("physionet_utility_norm", 0.0), 4),
        "physionet_utility_05": round(std_m.get("physionet_utility_norm", 0.0), 4),
        "mean_lead_time_hours": round(opt_m.get("mean_lead_time_hours", 0.0), 2),
        "early_detection_rate_ge_4h": round(opt_m.get("early_detection_rate_ge_4h", 0.0) * 100.0, 1),
        "false_alarm_rate_per_100h": round(opt_m.get("false_alarm_rate_per_100h", 0.0), 2)
    }


def main():
    print("=" * 80)
    print("STARTING CLINICAL NEURAL ARCHITECTURE & HORIZON BENCHMARK SWEEP")
    print("=" * 80)

    records = []

    # STEP 1: 4 Architectures x 2 Horizons (H=6, H=12)
    for h in HORIZONS:
        for m in MODELS:
            print(f"\n>>> Running Ablation: Model={m} | Horizon={h}h | SeqLen={SEQ_LEN}h")
            # 1. Train
            train_cmd = [
                sys.executable, "-u", "train_pipeline.py",
                "--model", m,
                "--horizon", str(h),
                "--seq_len", str(SEQ_LEN),
                "--epochs", str(EPOCHS),
                "--batch_size", str(BATCH_SIZE),
                "--max_patients", str(MAX_PATIENTS)
            ]
            run_cmd(train_cmd)

            # 2. Evaluate
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{m}_h{h}_seq{SEQ_LEN}.pth")
            eval_cmd = [
                sys.executable, "-u", "evaluate_pipeline.py",
                "--checkpoint", ckpt_path,
                "--output_dir", EVAL_DIR
            ]
            run_cmd(eval_cmd)

            # 3. Parse report
            rec = parse_eval_report(m, h)
            records.append(rec)

    # Convert to DataFrame
    df_ablation = pd.DataFrame(records)

    # Identify Champion at H=6 based on compound score (AUPRC + Utility)
    df_h6 = df_ablation[df_ablation["horizon_hours"] == 6].copy()
    # Rank by test_aucpr and utility
    df_h6["rank_score"] = df_h6["test_aucpr"] * 2.0 + df_h6["physionet_utility_opt"]
    champion_model = df_h6.sort_values(by="rank_score", ascending=False).iloc[0]["model"]
    print(f"\n{'*'*80}")
    print(f"CHAMPION NEURAL ARCHITECTURE IDENTIFIED: '{champion_model.upper()}'")
    print(f"{'*'*80}\n")

    # STEP 2: Horizon Sensitivity Sweep for Champion on remaining horizons (10h, 18h)
    extra_horizons = [10, 18]
    for h in extra_horizons:
        print(f"\n>>> Running Horizon Sweep: Champion={champion_model} | Horizon={h}h")
        train_cmd = [
            sys.executable, "-u", "train_pipeline.py",
            "--model", champion_model,
            "--horizon", str(h),
            "--seq_len", str(SEQ_LEN),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--max_patients", str(MAX_PATIENTS)
        ]
        run_cmd(train_cmd)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{champion_model}_h{h}_seq{SEQ_LEN}.pth")
        eval_cmd = [
            sys.executable, "-u", "evaluate_pipeline.py",
            "--checkpoint", ckpt_path,
            "--output_dir", EVAL_DIR
        ]
        run_cmd(eval_cmd)

        rec = parse_eval_report(champion_model, h)
        records.append(rec)

    # Master Results DataFrame
    master_df = pd.DataFrame(records)
    master_csv_path = os.path.join(RESULTS_DIR, "neural_ablation_results.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"\n[DONE] Saved Master Results to: {master_csv_path}")

    # Display Master Tables
    print("\n" + "=" * 90)
    print("MASTER NEURAL ARCHITECTURE ABLATION TABLE (H=6h & H=12h)")
    print("=" * 90)
    cols_display = [
        "model", "horizon_hours", "test_auroc", "test_aucpr",
        "val_tuned_threshold", "test_f1_opt", "test_sensitivity_opt",
        "test_specificity_opt", "mean_lead_time_hours", "physionet_utility_opt"
    ]
    print(master_df[master_df["horizon_hours"].isin([6, 12])][cols_display].to_string(index=False))

    print("\n" + "=" * 90)
    print(f"HORIZON SENSITIVITY DEGRADATION TABLE ({champion_model.upper()}: 6h vs 10h vs 12h vs 18h)")
    print("=" * 90)
    df_sens = master_df[master_df["model"] == champion_model].sort_values(by="horizon_hours")
    print(df_sens[[
        "horizon_hours", "test_auroc", "test_aucpr", "test_f1_opt",
        "test_sensitivity_opt", "test_specificity_opt", "mean_lead_time_hours", "physionet_utility_opt"
    ]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
