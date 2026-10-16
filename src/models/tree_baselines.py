"""
Tree-Based Competitive Baselines for Sepsis Prediction.
Transforms sliding temporal windows into rich tabular summary features:
- Current (last) values
- Window summary statistics: Mean, Std, Min, Max
- Trajectory dynamics: Delta (Last - First), Linear Trend/Slope
Supports LightGBM, XGBoost, and scikit-learn HistGradientBoostingClassifier.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from src.utils.metrics import compute_classification_metrics, find_optimal_threshold

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def extract_window_tabular_features(
    dataset,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Featurizes a SepsisSlidingWindowDataset into a 2D tabular feature matrix for tree models.
    
    For each physiological variable x in the window [0 ... T-1]:
    - x_last: current physiological state
    - x_mean: average level over window
    - x_std: volatility/variability
    - x_min, x_max: dynamic range
    - x_delta: total window trajectory shift (x[T-1] - x[0])
    """
    if hasattr(dataset, "X"):
        # Vectorized batch processing: dataset.X is (N, C, T)
        X_np = dataset.X.numpy()  # (N, C, T)
        curr = X_np[:, :, -1]     # (N, C)
        mean = np.mean(X_np, axis=2)
        std = np.std(X_np, axis=2)
        min_v = np.min(X_np, axis=2)
        max_v = np.max(X_np, axis=2)
        delta = curr - X_np[:, :, 0]
        slope = delta / float(max(1, X_np.shape[2] - 1))
        
        X_mat = np.concatenate([curr, mean, std, min_v, max_v, delta, slope], axis=1).astype(np.float32)
        y_vals = dataset.y.numpy().astype(int)
        pids = np.array(dataset.pids)
        hours = dataset.hours.numpy()
        return X_mat, y_vals, pids, hours

    X_rows = []
    y_vals = []
    pids = []
    hours = []

    for item in dataset.samples:
        X_ts = item[0].numpy()  # (C, T)
        y = int(item[1 if len(item) == 4 else 2].item())
        pid = item[2 if len(item) == 4 else 3]
        hr = item[3 if len(item) == 4 else 4]

        curr = X_ts[:, -1]
        mean = np.mean(X_ts, axis=1)
        std = np.std(X_ts, axis=1)
        min_v = np.min(X_ts, axis=1)
        max_v = np.max(X_ts, axis=1)
        delta = X_ts[:, -1] - X_ts[:, 0]
        slope = delta / float(max(1, X_ts.shape[1] - 1))

        row_feats = np.concatenate([curr, mean, std, min_v, max_v, delta, slope])
        X_rows.append(row_feats)
        y_vals.append(y)
        pids.append(pid)
        hours.append(hr)

    return np.array(X_rows, dtype=np.float32), np.array(y_vals, dtype=int), np.array(pids), np.array(hours)


class TreeSepsisBaseline:
    """
    Unified Tree-Based Sepsis Predictor.
    Prioritizes LightGBM -> XGBoost -> HistGradientBoostingClassifier.
    """
    def __init__(
        self,
        model_type: str = "auto",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        scale_pos_weight: float = 10.0,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None
        self.optimal_threshold: float = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TreeSepsisBaseline":
        """
        Fits tree model with appropriate positive class weighting.
        """
        # Determine backend
        backend = self.model_type
        if backend == "auto":
            if HAS_LIGHTGBM:
                backend = "lightgbm"
            elif HAS_XGBOOST:
                backend = "xgboost"
            else:
                backend = "hist_gb"

        if backend == "lightgbm" and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif backend == "xgboost" and HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1
            )
        else:
            # High-performance native scikit-learn GBDT
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.random_state
            )

        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, method: str = "max_f1") -> float:
        val_probs = self.predict_proba(X_val)
        self.optimal_threshold = find_optimal_threshold(y_val, val_probs, method=method)
        return self.optimal_threshold

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        eval_thresh = threshold if threshold is not None else self.optimal_threshold
        test_probs = self.predict_proba(X_test)
        metrics = compute_classification_metrics(y_test, test_probs, threshold=eval_thresh)
        metrics["model_type"] = type(self.model).__name__
        return metrics
