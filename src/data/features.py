"""
Feature Engineering for Clinical Time-Series Sepsis Prediction.
Features:
- Domain-specific wearable hemodynamic channels:
    * Shock Index (SI): HR / (SBP + 1e-5)
    * Pulse Pressure (PP): SBP - DBP
    * Estimated MAP (eMAP): DBP + (SBP - DBP) / 3.0
    * Respiration-Oxygenation Ratio (ROX): O2Sat / (Resp + 1e-5)
- Causal rolling statistics (mean, std) over 3h and 6h windows
- Causal 1-hour first-difference derivatives (rate of physiological change)
- Rough path signatures & Level-2 Lévy areas
Strictly causal: Zero backward-fill (bfill). Anti-anticipative throughout.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_hemodynamic_channels(
    df: pd.DataFrame,
    hr_col: str = "HR",
    sbp_col: str = "SBP",
    dbp_col: str = "DBP",
    resp_col: str = "Resp",
    o2sat_col: str = "O2Sat"
) -> pd.DataFrame:
    """
    Computes 4 domain-specific wearable hemodynamic coupling channels:
    - Shock Index (SI): HR / (SBP + 1e-5)
    - Pulse Pressure (PP): SBP - DBP
    - Estimated MAP (eMAP): DBP + (SBP - DBP) / 3.0
    - Respiration-Oxygenation Ratio (ROX): O2Sat / (Resp + 1e-5)
    
    Zero future information leakage: instantaneous point-in-time calculation.
    """
    df = df.copy()
    sbp_safe = df[sbp_col].clip(lower=1.0)
    resp_safe = df[resp_col].clip(lower=1.0)

    df["Shock_Index"] = df[hr_col] / sbp_safe
    df["Pulse_Pressure"] = df[sbp_col] - df[dbp_col]
    df["Estimated_MAP"] = df[dbp_col] + (df[sbp_col] - df[dbp_col]) / 3.0
    df["ROX_Index"] = df[o2sat_col] / resp_safe
    return df


def compute_causal_rolling_features(
    df: pd.DataFrame,
    vitals: List[str],
    windows: List[int] = [3, 6],
    patient_id_col: str = "Patient_ID",
    time_col: str = "Hour"
) -> pd.DataFrame:
    """
    Computes causal rolling mean and standard deviation per patient.
    
    Formula:
        mean_{k}(t) = (1/k) * sum_{i=0}^{k-1} x(t - i)
        std_{k}(t) = sqrt( (1/(k-1)) * sum_{i=0}^{k-1} (x(t - i) - mean_{k}(t))^2 )
    """
    df = df.sort_values(by=[patient_id_col, time_col]).copy()
    grouped = df.groupby(patient_id_col, sort=False)[vitals]

    feature_dfs = [df]

    for w in windows:
        r_mean = (
            grouped.rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .add_suffix(f"_mean_{w}hr")
        )
        r_std = (
            grouped.rolling(window=w, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .add_suffix(f"_std_{w}hr")
        )
        feature_dfs.extend([r_mean, r_std])

    df_out = pd.concat(feature_dfs, axis=1)
    return df_out


def compute_causal_derivatives(
    df: pd.DataFrame,
    vitals: List[str],
    patient_id_col: str = "Patient_ID",
    time_col: str = "Hour"
) -> pd.DataFrame:
    """
    Computes causal 1-hour rate of change (first difference):
        dx/dt (t) = x(t) - x(t - 1)
    At t = 0 (ICU admission), rate of change is initialized to 0.0 (no backward leakage).
    """
    df = df.sort_values(by=[patient_id_col, time_col]).copy()
    diffs = (
        df.groupby(patient_id_col, sort=False)[vitals]
        .diff()
        .fillna(0.0)
        .add_suffix("_derivative_1hr")
    )
    return pd.concat([df, diffs], axis=1)


def compute_path_signatures(
    df: pd.DataFrame,
    signature_cols: List[str],
    seq_len: int = 14,
    depth: int = 2,
    patient_id_col: str = "Patient_ID",
    time_col: str = "Hour"
) -> pd.DataFrame:
    """
    Fallback NumPy Level-2 path signatures (increments + anti-symmetric Levy areas).
    Pads initial timesteps (< seq_len) with 0.0 without backward leakage.
    """
    df = df.sort_values(by=[patient_id_col, time_col]).copy()
    d = len(signature_cols)
    sig_col_names = [f"sig_inc_{c}" for c in signature_cols] + [
        f"sig_area_{signature_cols[i]}_{signature_cols[j]}"
        for i in range(d) for j in range(i + 1, d)
    ]
    s_dim = len(sig_col_names)
    
    sig_rows = []
    for pid, group in df.groupby(patient_id_col, sort=False):
        vals = group[signature_cols].values
        n_steps = len(vals)
        patient_sigs = np.zeros((n_steps, s_dim), dtype=np.float32)
        
        if n_steps >= seq_len:
            for idx in range(n_steps - seq_len + 1):
                win = vals[idx : idx + seq_len]
                inc = win[-1] - win[0]
                dx = np.diff(win, axis=0)
                cum_x = win[:-1]
                areas = []
                for i in range(d):
                    for j in range(i + 1, d):
                        a_ij = np.sum(cum_x[:, i] * dx[:, j])
                        a_ji = np.sum(cum_x[:, j] * dx[:, i])
                        areas.append(a_ij - a_ji)
                patient_sigs[idx + seq_len - 1] = np.concatenate([inc, areas])
                
        sig_rows.append(patient_sigs)
        
    all_sigs = np.vstack(sig_rows)
    sig_df = pd.DataFrame(all_sigs, columns=sig_col_names, index=df.index)
    return pd.concat([df, sig_df], axis=1)
