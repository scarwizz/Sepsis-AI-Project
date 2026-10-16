"""
Hybrid Deep Learning Architectures for Sepsis Prediction.
Includes:
- TCNLSTMHybrid: Local temporal feature extraction via dilated convolutions followed by long-term recurrent memory.
- HybridSepsisModel: Multi-modal fusion of raw physiological trajectory (TCN stream) with geometric path signatures (MLP stream).
- PathSignatureLayer: Exact GPU-accelerated Level 1 increments and Level 2 iterated integrals & signed Lévy areas.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from src.models.tcn import TCNExtractor


class TCNLSTMHybrid(nn.Module):
    """
    Hybrid TCN-LSTM Model.
    Architecture:
        1. Dilated Causal TCN extracts multi-scale local pattern representations.
        2. Recurrent LSTM models long-range physiological dependencies across the sequence.
        3. Fully-connected head predicts sepsis onset logit.
    """
    def __init__(
        self,
        num_inputs: int,
        tcn_num_channels: List[int] = [32, 64, 128],
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        output_size: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TCNLSTMHybrid, self).__init__()
        self.tcn = TCNExtractor(
            num_inputs=num_inputs,
            num_channels=tcn_num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        last_tcn_channel = tcn_num_channels[-1]
        self.lstm = nn.LSTM(
            input_size=last_tcn_channel,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_features, T)
        tcn_feats = self.tcn(x)                 # (B, C_last, T)
        lstm_input = tcn_feats.permute(0, 2, 1)  # (B, T, C_last)

        lstm_out, _ = self.lstm(lstm_input)     # (B, T, H)
        last_step = lstm_out[:, -1, :]          # (B, H)
        logits = self.classifier(last_step)     # (B, output_size)
        return logits


class PathSignatureLayer(nn.Module):
    """
    Computes exact truncated rough path signature (level 1 increments + level 2 Lévy areas).
    Augments input path with normalized time channel t in [0, 1] -> dimension D = C + 1.
    Output:
      - Level 1 signature: total increment delta_X_i = X_T - X_0 (dimension D)
      - Level 2 signature: iterated integrals / cross-vital Lévy areas (dimension D * D)
    Total signature dimension: D + D*D.
    Fully causal, GPU-vectorized, differentiable, O(B * D * T).
    """
    def __init__(self, num_channels: int = 6, include_time: bool = True):
        super(PathSignatureLayer, self).__init__()
        self.num_channels = num_channels
        self.include_time = include_time
        self.total_dim = num_channels + 1 if include_time else num_channels
        self.sig_dim = self.total_dim + self.total_dim * self.total_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        if self.include_time:
            time_ch = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype).view(1, 1, T).expand(B, 1, T)
            path = torch.cat([x, time_ch], dim=1)  # (B, D, T)
        else:
            path = x
        D = path.shape[1]

        # Increments dX_t = X_{t} - X_{t-1}
        dX = path[:, :, 1:] - path[:, :, :-1]  # (B, D, T-1)

        # Level 1: sum of increments = X_T - X_0
        sig1 = torch.sum(dX, dim=2)  # (B, D)

        # Level 2 Stratonovich / Trapezoidal iterated integral:
        X_rel = path[:, :, :-1] - path[:, :, :1]  # (B, D, T-1)
        X_mid = X_rel + 0.5 * dX                  # (B, D, T-1)

        # Batched outer product over time
        sig2 = torch.bmm(X_mid, dX.transpose(1, 2))  # (B, D, D)
        sig2_flat = sig2.reshape(B, D * D)           # (B, D*D)

        sig = torch.cat([sig1, sig2_flat], dim=1)    # (B, D + D*D)
        return sig


class HybridSepsisModel(nn.Module):
    """
    Multi-Stream Hybrid Model combining Dilated Causal TCN and Path Signatures.
    Extracts geometric path signature invariants on the fly via PathSignatureLayer.
    Both streams are projected to 1D feature vectors before concatenation.
    """
    def __init__(
        self,
        num_inputs: int,
        sig_channels: int = 6,
        tcn_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        mlp_hidden_size: int = 64,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(HybridSepsisModel, self).__init__()
        self.sig_channels = sig_channels
        
        # Stream 1: TCN Feature Extractor on all input features
        self.tcn_branch = TCNExtractor(
            num_inputs=num_inputs,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        tcn_out_dim = tcn_channels[-1]

        # Stream 2: Path Signature Layer on core vitals + MLP projection
        self.sig_layer = PathSignatureLayer(num_channels=sig_channels, include_time=True)
        sig_features = self.sig_layer.sig_dim

        self.signature_branch = nn.Sequential(
            nn.Linear(sig_features, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 32),
            nn.ReLU()
        )

        # Fusion Head: Concatenate (tcn_out_dim + 32)
        fusion_dim = tcn_out_dim + 32
        self.classifier_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor, x_sig: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C_total, T)
        tcn_out = self.tcn_branch(x)
        tcn_vector = tcn_out[:, :, -1]  # Extract final time step: (B, C_tcn)

        # Path Signature computation on core vitals
        if x_sig is None:
            x_core = x[:, :self.sig_channels, :]
            x_sig = self.sig_layer(x_core)
        sig_vector = self.signature_branch(x_sig)  # (B, 32)

        combined = torch.cat([tcn_vector, sig_vector], dim=1)
        logits = self.classifier_head(combined)  # (B, output_size)
        return logits

    def get_levy_area_importance(self, channel_names: List[str]) -> List[Tuple[str, float]]:
        """
        Calculates attribution magnitude of Level-2 signed Lévy areas:
            A_{i,j} = 0.5 * (X^{i,j} - X^{j,i})
        from the first linear layer weights of signature_branch.
        """
        linear = self.signature_branch[0]
        W = linear.weight.detach().cpu().numpy()  # (mlp_hidden_size, sig_dim)
        D = self.sig_channels + 1  # includes time channel
        names = list(channel_names[:self.sig_channels]) + ["Time"]
        
        # Offset: Level 1 increments occupy indices 0..D-1
        # Level 2 matrix occupies indices D .. D + D*D - 1
        levy_importances = []
        for i in range(D):
            for j in range(i + 1, D):
                idx_ij = D + i * D + j
                idx_ji = D + j * D + i
                
                # Weight vector for anti-symmetric Levy area
                w_area = 0.5 * (W[:, idx_ij] - W[:, idx_ji])
                importance = float(np.linalg.norm(w_area))
                pair_name = f"{names[i]} <-> {names[j]}"
                levy_importances.append((pair_name, importance))
                
        # Sort descending by attribution norm
        levy_importances.sort(key=lambda x: x[1], reverse=True)
        return levy_importances
