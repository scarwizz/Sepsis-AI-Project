# Multi-Horizon Early Sepsis Detection via Causal Wearable Time-Series & Rough Path Lévy Areas

[![PhysioNet 2019](https://img.shields.io/badge/PhysioNet-CinC%202019-blue.svg)](https://physionet.org/content/challenge-2019/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a modular, reproducible benchmark for **Early Sepsis Prediction from Non-Invasive Wearable Vital Signs** using causal deep learning and geometric rough path theory on the full **PhysioNet Computing in Cardiology Challenge 2019** cohort (40,336 ICU stays across MIMIC-III and Emory eICU).

---

## 1. Clinical Motivation & Problem Statement

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. In intensive care units (ICUs), clinical evidence shows that **in-hospital mortality increases by 7.6% for every hour of delayed antimicrobial therapy** following onset. 

However, existing machine learning pipelines suffer from three pervasive flaws in real-world bedside deployment:
1. **Invasive & Latent Laboratory Dependency:** Traditional models rely heavily on invasive arterial blood draws and laboratory tests (e.g., Lactate, Bilirubin, Platelets, Creatinine) which exhibit collection latencies of 4 to 24 hours. This work restricts inputs **strictly to 6 non-invasive wearable vital signs** (`HR`, `O2Sat`, `Temp`, `Resp`, `SBP`, `DBP`), enabling continuous non-invasive bedside and telemetry monitoring.
2. **Double-Buffering & The "-6 Hour" Target Distortion:** In PhysioNet 2019, raw challenge labels already start 6 hours before clinical onset ($\text{SepsisLabel}_t = 1 \iff t \ge t_{sepsis} - 6$). Prior literature erroneously shifted this raw label by $-H$, inadvertently shifting the prediction target up to $t_{sepsis} - (H + 6)$ hours before onset (e.g. attempting to predict 18h prior while assuming 12h).
3. **Severe Look-Ahead Data Leakage:** Pervasive use of bidirectional backward-fill (`bfill`), full-stay mean imputation, and dataset-wide feature scaling leaks future physiological deterioration into earlier timestamps, inflating reported offline AUROC while collapsing in true clinical deployment.

---

## 2. Mathematical Formulation

### 2.1 Causal Horizon Reconstruction
To eliminate the 6-hour label distortion, we mathematically reconstruct the true clinical onset time $t_{sepsis}$:

$$\begin{aligned}
t_{sepsis} &= \min \left\{ t \;\middle|\; \text{SepsisLabel}_t = 1 \right\} + 6 \\
\text{Early Warning Target } y_t^{(H)} &= \begin{cases} 
1 & \text{if } t \in [\max(0, t_{sepsis} - H), \; t_{sepsis}] \\
0 & \text{otherwise}
\end{cases}
\end{aligned}$$

where $H \in \{4\text{h}, 6\text{h}, 10\text{h}, 12\text{h}, 18\text{h}\}$ defines the exact clinical early warning window prior to onset.

### 2.2 Domain-Specific Wearable Hemodynamic Coupling
We derive 4 point-in-time hemodynamic coupling channels prior to feature standardization:

$$\begin{aligned}
\text{Shock Index (SI)} &= \frac{\text{HR}}{\text{SBP} + 10^{-5}} \\
\text{Pulse Pressure (PP)} &= \text{SBP} - \text{DBP} \\
\text{Estimated MAP (eMAP)} &= \text{DBP} + \frac{\text{SBP} - \text{DBP}}{3} \\
\text{Respiration-Oxygenation (ROX)} &= \frac{\text{O2Sat}}{\text{Resp} + 10^{-5}}
\end{aligned}$$

Combined with causal rolling statistics (means and standard deviations over 3h and 6h windows) and causal 1h derivatives ($\frac{dx}{dt} = x_t - x_{t-1}$), the input representation provides $C = 60$ causal features per timestep.

### 2.3 Rough Path Invariants & Level-2 Signed Lévy Areas
While 1D dilated convolutions model univariate temporal rates of change, they are blind to the **non-linear geometric circulation** between paired physiological variables. For a multi-dimensional trajectory $X: [0, T] \to \mathbb{R}^D$ augmented with time:

$$\begin{aligned}
\text{Level 1 Increment: } \Delta X^i &= X^i_T - X^i_0 \\
\text{Level 2 Iterated Integral: } \mathbb{X}^{i, j} &= \int_{0}^T (X^i_t - X^i_0) \, dX^j_t \\
\text{Signed Lévy Area: } A^{i, j} &= \frac{1}{2} \left( \mathbb{X}^{i, j} - \mathbb{X}^{j, i} \right) = \frac{1}{2} \int_{0}^T \left( X^i_t \, dX^j_t - X^j_t \, dX^i_t \right)
\end{aligned}$$

Geometrically, $A^{i, j}$ represents the **signed cross-sectional area** enclosed by the trajectory projected onto the $(X^i, X^j)$ coordinate plane. In septic shock pathogenesis:
- When heart rate accelerates while blood pressure decompensates, the trajectory circulates in a clockwise loop ($A^{\text{HR}, \text{SBP}} > 0$).
- When respiratory rate surges ahead of oxygen desaturation, the loop area quantifies respiratory compensation latency.

The multi-stream **`HybridSepsisModel`** fuses dilated causal TCN representations with exact GPU-computed Level-2 Lévy areas in $O(B \cdot D \cdot T)$.

---

## 3. Repository Architecture

```text
E:\AI\
├── configs/
│   └── default_config.yaml    # Centralized hyperparameters, seeds, and dilation schedules
├── data/
│   └── cleaned_sepsis_dataset.csv       # Full PhysioNet 2019 dataset (40,336 patients, 66.3 MB)
├── figures/                   # High-resolution benchmark figures (300 DPI PNG + Vector PDF)
│   ├── fig1_roc_pr_comparison.png       # Test ROC and Precision-Recall curves
│   ├── fig2_horizon_utility_curve.png   # Multi-horizon utility and AUROC error bands
│   ├── fig3_levy_area_importance.png    # Learned physiological Lévy area feature importance
│   └── fig4_patient_trajectory_case_study.png # Bedside ICU early alarm trajectory
├── results/
│   ├── full_cohort_ablation_results.csv # Master Table 1 (4 models, H=6h & H=12h)
│   └── full_cohort_horizon_sweep.csv    # Master Table 2 (hybrid_signature vs. tcn: 4h to 18h)
├── src/
│   ├── data/
│   │   ├── preprocess.py      # Stateful causal ffill + train median fallback + standard scaling
│   │   ├── features.py        # Hemodynamic coupling (SI, PP, eMAP, ROX) + rolling stats + derivatives
│   │   └── dataset.py         # Fast memory-indexed sliding window dataset (117k samples/sec)
│   ├── models/
│   │   ├── tcn.py             # MultiTCN & TCNWithAttention (k=3, dilations [1,2,4,8], RF=29h)
│   │   └── hybrids.py         # TCNLSTMHybrid & Dual-Stream HybridSepsisModel (Path Signatures)
│   └── utils/
│       ├── losses.py          # Mathematically calibrated alpha-balanced Focal Loss
│       ├── metrics.py         # Official PhysioNet Utility (U_norm), Lead-Time, 95% Bootstrap CIs
│       └── seed.py            # Complete deterministic random seeding
├── train_pipeline.py          # Unified CLI entry point for full-cohort training
├── evaluate_pipeline.py       # Unified held-out Test evaluation with 200 patient bootstrap CIs
├── run_full_cohort_benchmark.py # Master runner executing the full benchmark experiment matrix
├── generate_paper_figures.py  # Standalone figure generation script
├── requirements.txt           # Pinned dependencies
└── README.md
```

---

## 4. Benchmark Results (Full 40,336-Patient Cohort)

All models are trained under strict 70/15/15 patient-level disjoint stratification (28,235 train / 6,050 validation / 6,051 test patients). Thresholds $\tau^*$ are tuned on the Validation partition with clinical specificity constraint ($\ge 80\%$) and evaluated out-of-sample on **156,200 held-out test windows**. 95% Confidence Intervals are derived from **200 patient-level bootstrap resamples**.

### Table 1: Architecture Comparison ($H = 6\text{h}$ and $H = 12\text{h}$, Sequence Length $= 14\text{h}$)

| Model Architecture | Horizon | Test AUROC [95% CI] | Test AUPRC [95% CI] | Optimal Threshold $\tau^*$ | Test $F_1$ | Sensitivity | Specificity | Mean Lead Time | False Alarms / 100h | PhysioNet Utility ($U_{norm}$) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tcn`** | 6h | **0.7365** [0.7124 - 0.7585] | **0.0662** [0.0545 - 0.0784] | 0.68 | **0.1254** | 21.07% | 95.67% | **38.82h** | **3.59** | +0.1083 [+0.0862 - +0.1317] |
| **`tcn_attention`** | 6h | 0.7147 [0.6903 - 0.7405] | 0.0548 [0.0457 - 0.0655] | 0.68 | 0.1100 | **28.98%** | 91.98% | 34.82h | 6.74 | **+0.1701** [+0.1386 - +0.1996] |
| **`hybrid_tcn_lstm`** | 6h | 0.7180 [0.6930 - 0.7425] | 0.0550 [0.0456 - 0.0667] | 0.70 | 0.1176 | 18.61% | **96.02%** | 36.26h | 3.28 | +0.0989 [+0.0734 - +0.1218] |
| **`hybrid_signature` (Ours)** | 6h | 0.7244 [0.7020 - 0.7502] | 0.0573 [0.0488 - 0.0686] | 0.68 | 0.1214 | 22.33% | 95.05% | 33.83h | 4.08 | +0.1231 [+0.0942 - +0.1488] |
| **`tcn`** | 12h | 0.7229 [0.6976 - 0.7484] | 0.0813 [0.0689 - 0.0954] | 0.66 | 0.1445 | 29.92% | 91.12% | 34.92h | 7.49 | +0.1866 [+0.1523 - +0.2207] |
| **`tcn_attention`** | 12h | **0.7287** [0.7064 - 0.7531] | 0.0774 [0.0672 - 0.0899] | 0.66 | 0.1407 | **40.88%** | 86.23% | 36.82h | 11.85 | **+0.2501** [+0.2087 - +0.2896] |
| **`hybrid_tcn_lstm`** | 12h | 0.7257 [0.7000 - 0.7482] | 0.0787 [0.0664 - 0.0930] | 0.68 | 0.1446 | 25.99% | **92.70%** | **38.04h** | **6.15** | +0.1527 [+0.1210 - +0.1825] |
| **`hybrid_signature` (Ours)** | 12h | 0.7257 [0.6989 - 0.7509] | **0.0819** [0.0706 - 0.0948] | 0.66 | **0.1517** | 28.53% | 92.26% | 37.09h | 6.57 | +0.1723 [+0.1398 - +0.2029] |

---

### Table 2: Multi-Horizon Sensitivity Sweep (`hybrid_signature` vs. `tcn`)

| Horizon ($H$) | Model | Test AUROC [95% CI] | Test AUPRC [95% CI] | Threshold $\tau^*$ | Test $F_1$ | Specificity | Lead Time | False Alarms / 100h | PhysioNet Utility ($U_{norm}$) [95% CI] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4h** | `tcn` | 0.7337 [0.7130 - 0.7580] | 0.0522 [0.0442 - 0.0623] | 0.70 | 0.1159 | **96.75%** | 34.30h | **2.65** | +0.0905 [+0.0701 - +0.1136] |
| **4h** | **`hybrid_signature`** | **0.7350** [0.7125 - 0.7580] | **0.0536** [0.0446 - 0.0648] | 0.68 | **0.1180** | 95.74% | **37.48h** | 3.42 | **+0.1165** [+0.0924 - +0.1420] |
| **6h** | `tcn` | **0.7365** [0.7124 - 0.7585] | **0.0662** [0.0545 - 0.0784] | 0.68 | **0.1254** | **95.67%** | **38.82h** | **3.59** | +0.1083 [+0.0862 - +0.1317] |
| **6h** | **`hybrid_signature`** | 0.7244 [0.7020 - 0.7502] | 0.0573 [0.0488 - 0.0686] | 0.68 | 0.1214 | 95.05% | 33.83h | 4.08 | **+0.1231** [+0.0942 - +0.1488] |
| **10h** | `tcn` | 0.7137 [0.6887 - 0.7382] | **0.0685** [0.0587 - 0.0802] | 0.66 | 0.1336 | **93.87%** | 35.78h | **5.18** | +0.1378 [+0.1083 - +0.1677] |
| **10h** | **`hybrid_signature`** | **0.7172** [0.6934 - 0.7398] | 0.0679 [0.0594 - 0.0794] | 0.66 | **0.1356** | 93.35% | **37.20h** | 5.63 | **+0.1477** [+0.1191 - +0.1767] |
| **12h** | `tcn` | 0.7229 [0.6976 - 0.7484] | 0.0813 [0.0689 - 0.0954] | 0.66 | 0.1445 | 91.12% | 34.92h | 7.49 | **+0.1866** [+0.1523 - +0.2207] |
| **12h** | **`hybrid_signature`** | **0.7257** [0.6989 - 0.7509] | **0.0819** [0.0706 - 0.0948] | 0.66 | **0.1517** | **92.26%** | **37.09h** | **6.57** | +0.1723 [+0.1398 - +0.2029] |
| **18h** | `tcn` | **0.7173** [0.6950 - 0.7407] | **0.1001** [0.0866 - 0.1160] | 0.66 | 0.1659 | **91.42%** | 36.87h | **7.29** | **+0.1895** [+0.1591 - +0.2218] |
| **18h** | **`hybrid_signature`** | 0.7155 [0.6926 - 0.7375] | 0.0995 [0.0864 - 0.1160] | 0.66 | **0.1672** | 91.24% | **37.45h** | 7.51 | +0.1875 [+0.1573 - +0.2187] |

---

## 5. Benchmark Figures

The complete suite of benchmark figures is rendered at 300 DPI in `figures/` (available in both high-resolution PNG and vector PDF):

| Figure | Description | File Link |
| :--- | :--- | :--- |
| **Figure 1** | Side-by-side Test ROC and Precision-Recall Curves ($H=6\text{h}$ and $H=12\text{h}$) | [`figures/fig1_roc_pr_comparison.png`](figures/fig1_roc_pr_comparison.png) |
| **Figure 2** | Multi-Horizon Normalized Utility ($U_{norm}$) & AUROC Curves with 95% Bootstrap Error Bands | [`figures/fig2_horizon_utility_curve.png`](figures/fig2_horizon_utility_curve.png) |
| **Figure 3** | Level-2 Path Signature Signed Lévy Area Attribution ($A^{\text{HR}, \text{SBP}}$, $A^{\text{Resp}, \text{O2Sat}}$, etc.) | [`figures/fig3_levy_area_importance.png`](figures/fig3_levy_area_importance.png) |
| **Figure 4** | ICU Bedside Trajectory Case Study: Real-time Vital Coupling and Early Detection Lead Time | [`figures/fig4_patient_trajectory_case_study.png`](figures/fig4_patient_trajectory_case_study.png) |

---

## 6. Reproduction & Execution Guide

### 6.1 Installation
```bash
git clone https://github.com/scarwizz/Sepsis-AI-Project.git
cd Sepsis-AI-Project
pip install -r requirements.txt
```

### 6.2 Training on Full Cohort
Train the champion `hybrid_signature` model on the full 40,336-patient dataset at Horizon $H=6\text{h}$:
```bash
python train_pipeline.py --model hybrid_signature --horizon 6 --seq_len 14 --epochs 5 --batch_size 256 --max_patients 0
```

Train vanilla `tcn` at Horizon $H=12\text{h}$:
```bash
python train_pipeline.py --model tcn --horizon 12 --seq_len 14 --epochs 5 --batch_size 256 --max_patients 0
```

### 6.3 Evaluation on Held-Out Test Set
Evaluate a checkpoint on the held-out Test set with 200 patient-level bootstrap resamples:
```bash
python evaluate_pipeline.py --checkpoint checkpoints/best_hybrid_signature_h6_seq14.pth --output_dir evaluation_outputs
```

### 6.4 Full Benchmark Sweep & Figure Generation
To run the automated multi-architecture ablation, multi-horizon sweep, and figure rendering in one command:
```bash
python run_full_cohort_benchmark.py
```

To regenerate the benchmark figures independently:
```bash
python generate_paper_figures.py
```

---

## 7. License
This repository is open-source under the [MIT License](LICENSE).
