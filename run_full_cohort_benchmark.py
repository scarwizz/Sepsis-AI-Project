"""
Master Script: Full-Cohort Clinical Benchmark & Horizon Sensitivity Sweep.
Executes:
1. Architecture Ablation (H=6h, H=12h, seq_len=14h):
   - 4 Models: 'tcn', 'tcn_attention', 'hybrid_tcn_lstm', 'hybrid_signature'
   - Trains on full 40,336-patient cohort (with 100% septic window retention).
   - Evaluates on 156,200 held-out Test windows across 6,051 test patients.
   - Computes 95% Bootstrap CIs (200 patient-level resamples).
2. Multi-Horizon Sensitivity Sweep (H in [4, 6, 10, 12, 18] hours):
   - Champion 'hybrid_signature' vs Baseline 'tcn'.
3. Compiles Master Results:
   - results/full_cohort_ablation_results.csv
   - results/full_cohort_horizon_sweep.csv
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np

MODELS = ["tcn", "tcn_attention", "hybrid_tcn_lstm", "hybrid_signature"]
ABLATION_HORIZONS = [6, 12]
SWEEP_HORIZONS = [4, 6, 10, 12, 18]
SEQ_LEN = 14
EPOCHS = 3
BATCH_SIZE = 256

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
    # Print trailing summary output
    print(res.stdout[-500:] if len(res.stdout) > 500 else res.stdout)


def parse_eval_report(model_name: str, horizon: int):
    json_path = os.path.join(EVAL_DIR, f"test_report_{model_name}_h{horizon}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing evaluation report: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    opt_m = data["metrics_threshold_optimal"]
    std_m = data["metrics_threshold_0_5"]
    boot_ci = data.get("bootstrap_ci_95", {})

    auroc = round(opt_m["auroc"], 4)
    aucpr = round(opt_m["aucpr"], 4)
    utility = round(opt_m.get("physionet_utility_norm", 0.0), 4)

    auroc_ci = boot_ci.get("auroc_ci_95", [auroc, auroc])
    aucpr_ci = boot_ci.get("aucpr_ci_95", [aucpr, aucpr])
    util_ci = boot_ci.get("utility_ci_95", [utility, utility])

    return {
        "model": model_name,
        "horizon_hours": horizon,
        "sequence_length": SEQ_LEN,
        "test_windows": data.get("test_windows", 0),
        "test_patients": data.get("test_patients", 0),
        "val_tuned_threshold": round(opt_m["tuned_threshold"], 4),
        "test_auroc": auroc,
        "test_auroc_ci_low": round(auroc_ci[0], 4),
        "test_auroc_ci_high": round(auroc_ci[1], 4),
        "test_aucpr": aucpr,
        "test_aucpr_ci_low": round(aucpr_ci[0], 4),
        "test_aucpr_ci_high": round(aucpr_ci[1], 4),
        "test_f1_opt": round(opt_m["f1_score"], 4),
        "test_sensitivity_opt": round(opt_m["recall_sensitivity"], 4),
        "test_specificity_opt": round(opt_m["specificity"], 4),
        "test_precision_opt": round(opt_m["precision"], 4),
        "test_f1_05": round(std_m["f1_score"], 4),
        "physionet_utility_opt": utility,
        "physionet_utility_ci_low": round(util_ci[0], 4),
        "physionet_utility_ci_high": round(util_ci[1], 4),
        "physionet_utility_05": round(std_m.get("physionet_utility_norm", 0.0), 4),
        "mean_lead_time_hours": round(opt_m.get("mean_lead_time_hours", 0.0), 2),
        "early_detection_rate_ge_4h": round(opt_m.get("early_detection_rate_ge_4h", 0.0) * 100.0, 1),
        "false_alarm_rate_per_100h": round(opt_m.get("false_alarm_rate_per_100h", 0.0), 2)
    }


def train_and_eval(model: str, horizon: int):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{model}_h{horizon}_seq{SEQ_LEN}.pth")
    report_path = os.path.join(EVAL_DIR, f"test_report_{model}_h{horizon}.json")
    
    # 1. Train if checkpoint missing
    if not os.path.exists(ckpt_path):
        train_cmd = [
            sys.executable, "-u", "train_pipeline.py",
            "--model", model,
            "--horizon", str(horizon),
            "--seq_len", str(SEQ_LEN),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--max_patients", "0"  # Full cohort
        ]
        run_cmd(train_cmd)
    else:
        print(f"[CACHE] Found existing checkpoint: {ckpt_path}")

    # 2. Evaluate if report missing
    if not os.path.exists(report_path):
        eval_cmd = [
            sys.executable, "-u", "evaluate_pipeline.py",
            "--checkpoint", ckpt_path,
            "--output_dir", EVAL_DIR,
            "--max_patients", "0"
        ]
        run_cmd(eval_cmd)
    else:
        print(f"[CACHE] Found existing evaluation report: {report_path}")

    return parse_eval_report(model, horizon)


def main():
    print("=" * 80)
    print("STARTING FULL-COHORT NEURAL BENCHMARK & MULTI-HORIZON SENSITIVITY SWEEP")
    print("=" * 80)

    ablation_records = []

    # -------------------------------------------------------------
    # 1. Architecture Ablation (H=6h & H=12h for all 4 models)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("PART 1: 4-ARCHITECTURE ABLATION (H=6h and H=12h)")
    print("#" * 80)
    for h in ABLATION_HORIZONS:
        for m in MODELS:
            print(f"\n>>> Running Ablation: Model={m} | Horizon={h}h")
            rec = train_and_eval(m, h)
            ablation_records.append(rec)

    df_ablation = pd.DataFrame(ablation_records)
    ablation_csv_path = os.path.join(RESULTS_DIR, "full_cohort_ablation_results.csv")
    df_ablation.to_csv(ablation_csv_path, index=False)
    print(f"\n[SAVED] Full-Cohort Ablation Table saved to: {ablation_csv_path}")

    # -------------------------------------------------------------
    # 2. Multi-Horizon Sensitivity Sweep (hybrid_signature vs. tcn)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("PART 2: MULTI-HORIZON SENSITIVITY SWEEP (hybrid_signature vs. tcn)")
    print("#" * 80)
    sweep_records = []
    sweep_models = ["hybrid_signature", "tcn"]

    for h in SWEEP_HORIZONS:
        for m in sweep_models:
            print(f"\n>>> Running Horizon Sweep: Model={m} | Horizon={h}h")
            rec = train_and_eval(m, h)
            sweep_records.append(rec)

    df_sweep = pd.DataFrame(sweep_records)
    sweep_csv_path = os.path.join(RESULTS_DIR, "full_cohort_horizon_sweep.csv")
    df_sweep.to_csv(sweep_csv_path, index=False)
    print(f"\n[SAVED] Full-Cohort Horizon Sweep saved to: {sweep_csv_path}")

    # Print summary tables
    print("\n" + "=" * 100)
    print("TABLE 1: FULL-COHORT 4-ARCHITECTURE ABLATION (H=6h & H=12h, 156,200 TEST WINDOWS)")
    print("=" * 100)
    cols_display = [
        "model", "horizon_hours", "val_tuned_threshold", "test_auroc", "test_aucpr",
        "test_f1_opt", "test_sensitivity_opt", "test_specificity_opt", "physionet_utility_opt",
        "mean_lead_time_hours", "false_alarm_rate_per_100h"
    ]
    print(df_ablation[cols_display].to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 2: MULTI-HORIZON SENSITIVITY SWEEP (HYBRID_SIGNATURE vs. TCN: 4h, 6h, 10h, 12h, 18h)")
    print("=" * 100)
    print(df_sweep.sort_values(by=["model", "horizon_hours"])[[
        "model", "horizon_hours", "val_tuned_threshold", "test_auroc", "test_aucpr",
        "test_f1_opt", "test_sensitivity_opt", "test_specificity_opt", "physionet_utility_opt",
        "mean_lead_time_hours", "false_alarm_rate_per_100h"
    ]].to_string(index=False))
    print("=" * 100)

    # 3. Automatically Generate All 4 Publication Figures
    print("\n" + "#" * 80)
    print("PART 3: GENERATING PUBLICATION FIGURES (figures/)")
    print("#" * 80)
    run_cmd([sys.executable, "-u", "generate_paper_figures.py"])


if __name__ == "__main__":
    main()
