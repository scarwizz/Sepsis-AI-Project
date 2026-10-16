"""
High-Performance Dataset and Cohort Partitioning for Sepsis Prediction.
Features:
- Vectorized PhysioNet -6h Ground Truth Reconstruction:
    In PhysioNet 2019, raw SepsisLabel=1 begins at t_sepsis - 6.
    Reconstructs true clinical onset t_sepsis = raw_onset + 6 before shifting.
- Multi-horizon early prediction window construction (e.g. 2h, 4h, 6h, 8h, 10h, 12h, 18h).
- Strict 70/15/15 Patient-Level Disjoint Stratification (Train / Val / Test).
- Memory-Indexed PyTorch Sliding Window Dataset storing a contiguous 2D feature tensor
  and an integer index array to eliminate RAM duplication and achieve 100k+ samples/sec.
- Training-only negative window subsampling with 100% retention of all septic patient windows.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional, Union
from src.utils.seed import seed_worker


def reconstruct_and_label_horizons(
    df: pd.DataFrame,
    prediction_horizon: int = 6,
    patient_id_col: str = "Patient_ID",
    time_col: str = "Hour",
    raw_label_col: str = "SepsisLabel",
    mask_post_onset: bool = False
) -> pd.DataFrame:
    """
    Vectorized correction of the PhysioNet '-6h' buffer and early prediction target generation.
    """
    df_out = df.sort_values(by=[patient_id_col, time_col]).reset_index(drop=True).copy()
    target_col = f"Target_{prediction_horizon}h"
    df_out[target_col] = 0

    septic_mask = (df_out[raw_label_col] == 1)
    if septic_mask.any():
        first_septic_hour = df_out[septic_mask].groupby(patient_id_col)[time_col].min()
        t_sepsis_series = first_septic_hour + 6.0
        
        t_sepsis = df_out[patient_id_col].map(t_sepsis_series)
        has_t_sepsis = t_sepsis.notnull()

        h_float = float(prediction_horizon)
        early_start = (t_sepsis - h_float).clip(lower=0.0)
        early_mask = has_t_sepsis & (df_out[time_col] >= early_start) & (df_out[time_col] <= t_sepsis)
        post_mask = has_t_sepsis & (df_out[time_col] > t_sepsis)

        df_out.loc[early_mask, target_col] = 1

        if mask_post_onset:
            df_out.loc[post_mask, target_col] = -1
        else:
            df_out.loc[post_mask, target_col] = 1

    return df_out


def patient_level_split(
    df: pd.DataFrame,
    patient_id_col: str = "Patient_ID",
    label_col: str = "SepsisLabel",
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
    max_patients: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits cohort into Train, Validation, and Test partitions strictly by Patient_ID.
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    train_frac, val_frac, test_frac = split_ratios

    patient_labels = df.groupby(patient_id_col)[label_col].max()
    unique_pids = patient_labels.index.values
    unique_labels = patient_labels.values

    if max_patients is not None and 0 < max_patients < len(unique_pids):
        septic_mask = (unique_labels == 1)
        septic_pids = unique_pids[septic_mask]
        non_septic_pids = unique_pids[~septic_mask]

        rng = np.random.RandomState(random_seed)
        num_septic = len(septic_pids)

        if max_patients > num_septic:
            num_non_septic = max_patients - num_septic
            chosen_non_septic = rng.choice(non_septic_pids, size=num_non_septic, replace=False)
            selected_pids = np.concatenate([septic_pids, chosen_non_septic])
        else:
            selected_pids = rng.choice(unique_pids, size=max_patients, replace=False)

        selected_mask = np.isin(unique_pids, selected_pids)
        unique_pids = unique_pids[selected_mask]
        unique_labels = unique_labels[selected_mask]
        df = df[df[patient_id_col].isin(set(unique_pids))].copy()

    temp_frac = val_frac + test_frac
    class_counts = np.bincount(unique_labels)
    can_stratify = len(class_counts) > 1 and np.min(class_counts) >= 3
    
    train_pids, temp_pids, train_y, temp_y = train_test_split(
        unique_pids,
        unique_labels,
        test_size=temp_frac,
        stratify=unique_labels if can_stratify else None,
        random_state=random_seed
    )

    relative_test_frac = test_frac / temp_frac
    temp_class_counts = np.bincount(temp_y) if len(temp_y) > 0 else []
    can_stratify_temp = len(temp_class_counts) > 1 and np.min(temp_class_counts) >= 2
    
    val_pids, test_pids, _, _ = train_test_split(
        temp_pids,
        temp_y,
        test_size=relative_test_frac,
        stratify=temp_y if can_stratify_temp else None,
        random_state=random_seed
    )

    train_pids_set = set(train_pids)
    val_pids_set = set(val_pids)
    test_pids_set = set(test_pids)

    assert len(train_pids_set.intersection(val_pids_set)) == 0
    assert len(train_pids_set.intersection(test_pids_set)) == 0
    assert len(val_pids_set.intersection(test_pids_set)) == 0

    train_df = df[df[patient_id_col].isin(train_pids_set)].copy()
    val_df = df[df[patient_id_col].isin(val_pids_set)].copy()
    test_df = df[df[patient_id_col].isin(test_pids_set)].copy()

    return train_df, val_df, test_df


class SepsisSlidingWindowDataset(Dataset):
    """
    High-Performance Memory-Indexed PyTorch Dataset for Full-Cohort Clinical Time-Series.
    Stores a single contiguous 2D float32 feature tensor and an integer index matrix
    [start_row, end_row, label, hour, pid_idx, raw_label] to eliminate duplicate window slicing.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        sequence_length: int = 14,
        is_train: bool = False,
        neg_stride: int = 3,
        patient_id_col: str = "Patient_ID",
        time_col: str = "Hour",
        raw_label_col: str = "SepsisLabel"
    ):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols

        df_sorted = df.sort_values(by=[patient_id_col, time_col]).reset_index(drop=True)
        
        feats_np = df_sorted[feature_cols].to_numpy(dtype=np.float32)
        self.features = torch.from_numpy(feats_np)

        pids_raw = df_sorted[patient_id_col].values
        targets_raw = df_sorted[target_col].to_numpy(dtype=np.float32)
        hours_raw = df_sorted[time_col].to_numpy(dtype=np.float32)
        
        has_raw_label = (raw_label_col in df_sorted.columns)
        raw_labels = df_sorted[raw_label_col].to_numpy(dtype=np.float32) if has_raw_label else targets_raw

        unique_pids, pid_inverse = np.unique(pids_raw, return_inverse=True)
        self.unique_pids = unique_pids

        diff = np.where(pid_inverse[:-1] != pid_inverse[1:])[0]
        starts = np.r_[0, diff + 1]
        ends = np.r_[diff + 1, len(pids_raw)]

        indices = []
        for s, e in zip(starts, ends):
            plen = e - s
            if plen < sequence_length:
                continue
            
            patient_is_septic = (np.any(raw_labels[s:e] == 1) or np.any(targets_raw[s:e] == 1))
            stride = 1 if (not is_train or patient_is_septic) else neg_stride

            for i in range(0, plen - sequence_length + 1, stride):
                row_end = s + i + sequence_length
                row_start = s + i
                target_val = targets_raw[row_end - 1]
                if target_val < 0:
                    continue
                hour_val = hours_raw[row_end - 1]
                raw_lbl = raw_labels[row_end - 1]
                indices.append((row_start, row_end, target_val, hour_val, pid_inverse[s], raw_lbl))

        if len(indices) > 0:
            self.indices = np.array(indices, dtype=np.float64)
            self.y = torch.from_numpy(self.indices[:, 2].astype(np.float32))
            self.hours = torch.from_numpy(self.indices[:, 3].astype(np.float32))
            self.raw_labels = torch.from_numpy(self.indices[:, 5].astype(np.float32))
        else:
            self.indices = np.empty((0, 6), dtype=np.float64)
            self.y = torch.empty((0,), dtype=torch.float32)
            self.hours = torch.empty((0,), dtype=torch.float32)
            self.raw_labels = torch.empty((0,), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        s = int(self.indices[idx, 0])
        e = int(self.indices[idx, 1])
        x = self.features[s:e].t()
        y = self.y[idx]
        return x, y

    def get_metadata(self):
        """Returns patient IDs, prediction hours, ground truth target labels, and raw challenge labels."""
        if len(self.indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        pid_indices = self.indices[:, 4].astype(int)
        pids = self.unique_pids[pid_indices]
        hours = self.hours.numpy()
        labels = self.y.numpy()
        raw_labels = self.raw_labels.numpy()
        return pids, hours, labels, raw_labels


def create_dataloaders(
    train_dataset: SepsisSlidingWindowDataset,
    val_dataset: SepsisSlidingWindowDataset,
    test_dataset: SepsisSlidingWindowDataset,
    batch_size: int = 256,
    use_weighted_sampler: bool = True,
    max_train_samples: int = 180000,
    seed: int = 42,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs deterministic DataLoaders with balanced sampler on the training partition.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    if use_weighted_sampler and len(train_dataset) > 0:
        train_labels = train_dataset.y.long().numpy()
        class_counts = np.bincount(train_labels)
        
        if len(class_counts) > 1 and class_counts[1] > 0:
            weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
            sample_weights = weights[train_labels]
            num_samples = min(len(sample_weights), max_train_samples)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=num_samples,
                replacement=True,
                generator=g
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=pin_memory
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=pin_memory
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
