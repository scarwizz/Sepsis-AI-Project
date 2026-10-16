"""
Strictly Causal Preprocessing Pipeline for Clinical Time-Series.
Zero Future Information Leakage:
- Forward-fill (ffill) strictly per patient trajectory
- Initial missing values imputed using population medians fitted strictly on the Training set
- Derivation of wearable hemodynamic coupling channels (Shock Index, Pulse Pressure, Estimated MAP, ROX)
- StandardScaler fitted strictly on the Training split
- Total elimination of backward-fill (bfill) across all variables
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from src.data.features import compute_hemodynamic_channels


class CausalPreprocessPipeline:
    """
    Stateful clinical preprocessor ensuring strict anti-anticipative causality.
    """
    def __init__(
        self,
        feature_cols: List[str],
        patient_id_col: str = "Patient_ID",
        time_col: str = "Hour",
        label_col: str = "SepsisLabel"
    ):
        self.base_vitals = list(feature_cols)
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.label_col = label_col
        
        self.feature_cols = list(self.base_vitals)
        self.hemodynamic_cols = ["Shock_Index", "Pulse_Pressure", "Estimated_MAP", "ROX_Index"]

        # Learned statistics from training partition
        self.train_medians_: Optional[pd.Series] = None
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_scale_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "CausalPreprocessPipeline":
        """
        Calculates population fallback medians and standardization moments strictly on train_df.
        """
        df = train_df.sort_values(by=[self.patient_id_col, self.time_col]).reset_index(drop=True).copy()
        
        # 1. Forward fill within patient stays
        df[self.base_vitals] = df.groupby(self.patient_id_col, sort=False)[self.base_vitals].ffill()
        
        # 2. Fit train population medians for features that start with NaN
        self.train_medians_ = df[self.base_vitals].median()
        df[self.base_vitals] = df[self.base_vitals].fillna(self.train_medians_).fillna(0.0)
        
        # 3. Compute derived hemodynamic channels prior to standardization
        df = compute_hemodynamic_channels(df)
        for col in self.hemodynamic_cols:
            if col not in self.feature_cols:
                self.feature_cols.append(col)

        # 4. Fit StandardScaler moments across all base + hemodynamic channels
        self.scaler_mean_ = df[self.feature_cols].mean().values
        self.scaler_scale_ = df[self.feature_cols].std().replace(0.0, 1.0).values
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies causal ffill + train-median fallback + hemodynamic derivation + train standardization.
        """
        if not self.is_fitted:
            raise RuntimeError("CausalPreprocessPipeline must be fit on training set before transform.")
            
        df_out = df.sort_values(by=[self.patient_id_col, self.time_col]).reset_index(drop=True).copy()
        
        # 1. Causal per-patient forward-fill
        df_out[self.base_vitals] = df_out.groupby(self.patient_id_col, sort=False)[self.base_vitals].ffill()
        
        # 2. Impute initial missing timestamps with training population medians
        df_out[self.base_vitals] = df_out[self.base_vitals].fillna(self.train_medians_).fillna(0.0)
        
        # 3. Compute derived hemodynamic channels
        df_out = compute_hemodynamic_channels(df_out)
        
        # 4. Apply training standardization moments: (x - mu) / sigma
        scaled_vals = (df_out[self.feature_cols].values - self.scaler_mean_) / self.scaler_scale_
        df_out[self.feature_cols] = scaled_vals
        
        return df_out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)
