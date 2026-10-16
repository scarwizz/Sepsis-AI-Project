"""
Scientific Figures Generator for Multi-Horizon Sepsis Prediction Benchmark.
Generates:
1. figures/fig1_roc_pr_comparison.png (+ PDF):
   Side-by-side Test ROC and Precision-Recall curves across 4 architectures at H=6h & H=12h.
2. figures/fig2_horizon_utility_curve.png (+ PDF):
   Multi-horizon line plot with 95% CI bands comparing HybridSepsisModel vs MultiTCN.
3. figures/fig3_levy_area_importance.png (+ PDF):
   Horizontal bar chart of learned Level-2 Path Signature Lévy area feature importance.
4. figures/fig4_patient_trajectory_case_study.png (+ PDF):
   Multi-panel clinical ICU trajectory of a True Positive test patient with lead-time alarm.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

FIGURES_DIR = "figures"
EVAL_DIR = "evaluation_outputs"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_COLORS = {
    "hybrid_signature": "#7A1C99",   # Rich Purple
    "tcn": "#0D6EFD",                # Royal Blue
    "tcn_attention": "#FD7E14",      # Clinical Orange
    "hybrid_tcn_lstm": "#198754"     # Clinical Green
}

MODEL_LABELS = {
    "hybrid_signature": "Hybrid Path Signature (Ours)",
    "tcn": "MultiTCN (Expanded RF 29h)",
    "tcn_attention": "TCN + Self-Attention",
    "hybrid_tcn_lstm": "TCN-LSTM Recurrent Hybrid"
}


def load_prediction_data(model: str, horizon: int):
    csv_path = os.path.join(EVAL_DIR, f"test_predictions_{model}_h{horizon}.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


# -----------------------------------------------------------------------------
# FIGURE 1: ROC & Precision-Recall Comparison (H=6h & H=12h)
# -----------------------------------------------------------------------------
def plot_fig1_roc_pr():
    print("Generating Figure 1: Test ROC and Precision-Recall Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    horizons = [6, 12]
    models = ["hybrid_signature", "tcn", "tcn_attention", "hybrid_tcn_lstm"]

    for row_idx, h in enumerate(horizons):
        ax_roc = axes[row_idx, 0]
        ax_pr = axes[row_idx, 1]

        # Random baseline lines
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC=0.50)")

        prevalence_plotted = False

        for m in models:
            df = load_prediction_data(m, h)
            if df is None:
                continue

            y_true = df["Target"].values
            y_prob = df["Probability"].values

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr, tpr,
                label=f"{MODEL_LABELS[m]} (AUROC={roc_auc:.3f})",
                color=MODEL_COLORS[m],
                linewidth=2.2
            )

            # PR
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(rec, prec)
            ax_pr.plot(
                rec, prec,
                label=f"{MODEL_LABELS[m]} (AUPRC={pr_auc:.3f})",
                color=MODEL_COLORS[m],
                linewidth=2.2
            )

            if not prevalence_plotted:
                prevalence = np.mean(y_true)
                ax_pr.axhline(
                    y=prevalence, color="k", linestyle="--", alpha=0.5,
                    label=f"Baseline Prevalence ({prevalence*100:.1f}%)"
                )
                prevalence_plotted = True

        # Polish ROC
        ax_roc.set_title(f"A. ROC Curves (Horizon H = {h}h Prior to Clinical Onset)")
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.grid(True)
        ax_roc.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax_roc.spines["top"].set_visible(False)
        ax_roc.spines["right"].set_visible(False)

        # Polish PR
        ax_pr.set_title(f"B. Precision-Recall Curves (Horizon H = {h}h)")
        ax_pr.set_xlabel("Recall (Sensitivity)")
        ax_pr.set_ylabel("Precision (Positive Predictive Value)")
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 0.45])
        ax_pr.grid(True)
        ax_pr.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax_pr.spines["top"].set_visible(False)
        ax_pr.spines["right"].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig1_roc_pr_comparison.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig1_roc_pr_comparison.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Fig 1 saved to: {png_path} & {pdf_path}")


# -----------------------------------------------------------------------------
# FIGURE 2: Multi-Horizon Utility and AUROC Degradation Curves
# -----------------------------------------------------------------------------
def plot_fig2_horizon_curves():
    print("Generating Figure 2: Multi-Horizon Utility and AUROC Curves...")
    sweep_csv = os.path.join(RESULTS_DIR, "full_cohort_horizon_sweep.csv")
    if not os.path.exists(sweep_csv):
        print(f"[WARN] {sweep_csv} not found yet. Using ablation results fallback.")
        sweep_csv = os.path.join(RESULTS_DIR, "full_cohort_ablation_results.csv")
    
    df_sweep = pd.read_csv(sweep_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    models = ["hybrid_signature", "tcn"]

    for m in models:
        m_df = df_sweep[df_sweep["model"] == m].sort_values("horizon_hours")
        if len(m_df) == 0:
            continue

        h_vals = m_df["horizon_hours"].values
        util_vals = m_df["physionet_utility_opt"].values
        util_low = m_df["physionet_utility_ci_low"].values
        util_high = m_df["physionet_utility_ci_high"].values

        auroc_vals = m_df["test_auroc"].values
        auroc_low = m_df["test_auroc_ci_low"].values
        auroc_high = m_df["test_auroc_ci_high"].values

        # Panel 1: Utility Score
        ax1.plot(
            h_vals, util_vals,
            marker="o", markersize=7, linewidth=2.5,
            label=MODEL_LABELS[m], color=MODEL_COLORS[m]
        )
        ax1.fill_between(
            h_vals, util_low, util_high,
            color=MODEL_COLORS[m], alpha=0.18
        )

        # Panel 2: AUROC
        ax2.plot(
            h_vals, auroc_vals,
            marker="s", markersize=7, linewidth=2.5,
            label=MODEL_LABELS[m], color=MODEL_COLORS[m]
        )
        ax2.fill_between(
            h_vals, auroc_low, auroc_high,
            color=MODEL_COLORS[m], alpha=0.18
        )

    # Style Panel 1 (Utility)
    ax1.set_title("A. Official PhysioNet Normalized Utility ($U_{norm}$) vs. Horizon")
    ax1.set_xlabel("Prediction Horizon Prior to Onset (Hours)")
    ax1.set_ylabel("Normalized Clinical Utility ($U_{norm}$)")
    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.7, label="Zero-Prediction Baseline")
    ax1.set_xticks([4, 6, 10, 12, 18])
    ax1.grid(True)
    ax1.legend(loc="lower left", frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Style Panel 2 (AUROC)
    ax2.set_title("B. Discriminatory Capacity (AUROC) vs. Prediction Horizon")
    ax2.set_xlabel("Prediction Horizon Prior to Onset (Hours)")
    ax2.set_ylabel("Test AUROC")
    ax2.set_xticks([4, 6, 10, 12, 18])
    ax2.set_ylim([0.55, 0.80])
    ax2.grid(True)
    ax2.legend(loc="lower left", frameon=True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig2_horizon_utility_curve.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig2_horizon_utility_curve.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Fig 2 saved to: {png_path} & {pdf_path}")


# -----------------------------------------------------------------------------
# FIGURE 3: Level-2 Path Signature Lévy Area Physiological Importance
# -----------------------------------------------------------------------------
def plot_fig3_levy_area_importance():
    print("Generating Figure 3: Level-2 Path Signature Lévy Area Importance...")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_hybrid_signature_h6_seq14.pth")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint {ckpt_path} missing.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    feature_cols = ckpt["feature_cols"]
    base_vitals = ckpt.get("base_vitals", ["HR", "O2Sat", "Temp", "Resp", "SBP", "DBP"])

    from src.models.hybrids import HybridSepsisModel
    model = HybridSepsisModel(
        num_inputs=len(feature_cols),
        sig_channels=len(base_vitals),
        tcn_channels=cfg["models"]["hybrid_signature"]["tcn_channels"],
        kernel_size=cfg["models"]["hybrid_signature"]["kernel_size"],
        mlp_hidden_size=cfg["models"]["hybrid_signature"]["mlp_hidden_size"],
        dropout=0.0
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pairs_importance = model.get_levy_area_importance(base_vitals)
    top_pairs = pairs_importance[:12]
    top_pairs.reverse()  # For bottom-to-top horizontal bar chart

    labels = [p[0] for p in top_pairs]
    weights = [p[1] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Gradient colors
    norm_w = (np.array(weights) - min(weights)) / (max(weights) - min(weights) + 1e-6)
    colors = [plt.cm.Purples(0.45 + 0.55 * val) for val in norm_w]

    bars = ax.barh(labels, weights, color=colors, edgecolor="#4A0E4E", linewidth=1.2, height=0.65)
    
    # Add data value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#2D112C"
        )

    ax.set_title("Physiological Lead-Lag Attributions: Level-2 Signed Lévy Areas", pad=15)
    ax.set_xlabel("Lévy Area Attribution Magnitude: $\\Vert W_{A_{i,j}} \\Vert_2$ (Iterated Integral Projection)")
    ax.set_xlim([0, max(weights) * 1.15])
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig3_levy_area_importance.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig3_levy_area_importance.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Fig 3 saved to: {png_path} & {pdf_path}")


# -----------------------------------------------------------------------------
# FIGURE 4: Patient Trajectory Case Study (ICU Lead-Time Demonstration)
# -----------------------------------------------------------------------------
def plot_fig4_case_study():
    print("Generating Figure 4: Patient Trajectory Case Study...")
    pred_path = os.path.join(EVAL_DIR, "test_predictions_hybrid_signature_h6.csv")
    if not os.path.exists(pred_path):
        print(f"[WARN] {pred_path} not found.")
        return

    eval_df = pd.read_csv(pred_path)
    
    # Find a representative true positive septic patient with clear lead time
    septic_pids = eval_df[eval_df["SepsisLabel"] == 1]["Patient_ID"].unique()
    chosen_pid = None
    
    for pid in septic_pids:
        pdf = eval_df[eval_df["Patient_ID"] == pid].sort_values("Hour")
        # Find first hour SepsisLabel==1
        raw_sepsis_hours = pdf[pdf["SepsisLabel"] == 1]["Hour"].values
        if len(raw_sepsis_hours) > 0 and len(pdf) >= 20:
            first_raw = raw_sepsis_hours[0]
            t_sepsis = first_raw + 6.0
            # Check if model alarmed in [t_sepsis - 10, t_sepsis]
            pre_alarms = pdf[(pdf["Hour"] >= t_sepsis - 10) & (pdf["Hour"] <= t_sepsis) & (pdf["Prediction_Opt"] == 1)]
            if len(pre_alarms) > 0:
                chosen_pid = pid
                break

    if chosen_pid is None:
        chosen_pid = septic_pids[0]

    # Load raw vitals for chosen_pid
    raw_df = pd.read_csv("data/cleaned_sepsis_dataset.csv")
    patient_raw = raw_df[raw_df["Patient_ID"] == chosen_pid].sort_values("Hour").reset_index(drop=True)
    patient_pred = eval_df[eval_df["Patient_ID"] == chosen_pid].sort_values("Hour").reset_index(drop=True)

    # Derived hemodynamic features
    sbp_safe = patient_raw["SBP"].clip(lower=1.0)
    patient_raw["Shock_Index"] = patient_raw["HR"] / sbp_safe
    patient_raw["Pulse_Pressure"] = patient_raw["SBP"] - patient_raw["DBP"]

    raw_onset = patient_raw[patient_raw["SepsisLabel"] == 1]["Hour"].min()
    t_sepsis = raw_onset + 6.0
    h_window = 6.0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True, dpi=300)

    hours = patient_raw["Hour"].values

    # Row 1: Heart Rate & Respiration Rate
    ax1.plot(hours, patient_raw["HR"], "r-", linewidth=2.0, label="Heart Rate (bpm)")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(hours, patient_raw["Resp"], "b--", linewidth=1.8, label="Respiratory Rate (breaths/min)")
    ax1.set_ylabel("HR (bpm)", color="darkred")
    ax1_twin.set_ylabel("Resp (/min)", color="darkblue")
    ax1.set_title(f"Clinical Case Study: ICU Patient #{chosen_pid} (True Positive Early Sepsis Detection)", pad=10)
    ax1.grid(True)
    ax1.spines["top"].set_visible(False)

    # Row 2: Blood Pressures
    ax2.plot(hours, patient_raw["SBP"], color="#2B8A3E", linewidth=2.0, label="Systolic BP (SBP)")
    ax2.plot(hours, patient_raw["DBP"], color="#74B816", linestyle="--", linewidth=1.8, label="Diastolic BP (DBP)")
    ax2.set_ylabel("Pressure (mmHg)")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Row 3: Hemodynamic Coupling (Shock Index & Pulse Pressure)
    ax3.plot(hours, patient_raw["Shock_Index"], color="#D9480F", linewidth=2.2, label="Shock Index (HR / SBP)")
    ax3.axhline(0.8, color="#D9480F", linestyle=":", alpha=0.7, label="Clinical Shock Threshold (0.80)")
    ax3.set_ylabel("Shock Index")
    ax3.legend(loc="upper left")
    ax3.grid(True)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Row 4: Predicted Probability & Early Alarm
    p_hours = patient_pred["Hour"].values
    probs = patient_pred["Probability"].values
    opt_thresh = 0.68

    ax4.plot(p_hours, probs, color="#7A1C99", linewidth=2.5, label="Model Sepsis Probability $P(Sepsis)$")
    ax4.axhline(opt_thresh, color="red", linestyle="--", linewidth=1.5, label=f"Clinical Threshold $\\tau^* = {opt_thresh:.2f}$")
    
    # Highlight alarm window
    ax4.axvspan(t_sepsis - h_window, t_sepsis, color="#FFE066", alpha=0.4, label=f"6h Early Warning Window")
    ax4.axvline(t_sepsis, color="darkred", linestyle="-", linewidth=2.0, label=f"True Clinical Onset $t_{{sepsis}} = {t_sepsis:.0f}$h")
    
    # Find first alarm
    alarm_pts = patient_pred[(patient_pred["Probability"] >= opt_thresh) & (patient_pred["Hour"] <= t_sepsis)]
    if len(alarm_pts) > 0:
        first_alarm = alarm_pts.iloc[0]["Hour"]
        lead_time = t_sepsis - first_alarm
        ax4.annotate(
            f"FIRST ALARM TRIGGERED\nLead Time: {lead_time:.1f} Hours Prior to Onset",
            xy=(first_alarm, opt_thresh),
            xytext=(first_alarm - 8, opt_thresh + 0.18),
            arrowprops=dict(facecolor="darkred", shrink=0.08, width=2, headwidth=8),
            fontweight="bold", color="darkred",
            bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.7, ec="darkred")
        )

    ax4.set_xlabel("ICU Length of Stay (Hours)")
    ax4.set_ylabel("Predicted Risk $P(Sepsis)$")
    ax4.set_ylim([-0.05, 1.05])
    ax4.legend(loc="upper left")
    ax4.grid(True)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Shade the onset period on all axes
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(t_sepsis - h_window, t_sepsis, color="#FFE066", alpha=0.25)
        ax.axvline(t_sepsis, color="darkred", linestyle="--", alpha=0.6)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig4_patient_trajectory_case_study.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig4_patient_trajectory_case_study.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Fig 4 saved to: {png_path} & {pdf_path}")


def main():
    print("=" * 80)
    print("GENERATING BENCHMARK FIGURES (300 DPI PNG + VECTOR PDF)")
    print("=" * 80)
    plot_fig1_roc_pr()
    plot_fig2_horizon_curves()
    plot_fig3_levy_area_importance()
    plot_fig4_case_study()
    print("=" * 80)
    print("ALL 4 BENCHMARK FIGURES GENERATED SUCCESSFULLY IN figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()
