"""
Fixed Temporal Convolutional Network (TCN) Architecture.
Solves all prior architectural bugs:
- Safe causal padding with Chomp1d (no empty slice when chomp_size=0)
- Kernel size k=3 with d=[1,2,4] yields receptive field RF = 29 hours (exceeds all sequence lengths)
- Modular separation: TCNExtractor (feature maps) vs MultiTCN (classification model with head)
- TCNWithAttention: Attention pooling over temporal steps for sepsis precursor detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Optional


class Chomp1d(nn.Module):
    """
    Strips trailing padding from PyTorch Conv1d to enforce strict anti-anticipative causal convolution.
    Safe against empty slice bug when chomp_size == 0.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual Dilated Causal Convolutional Block.
    Contains two weight-normalized dilated convolutions with chomp, ReLU, and dropout.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # 1x1 conv residual projection when channel dimensions change
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNExtractor(nn.Module):
    """
    Core TCN Backbone feature extractor: maps (B, C_in, T) -> (B, C_out, T).
    Receptive Field formula:
        RF = 1 + sum_{l=0}^{L-1} 2 * (kernel_size - 1) * dilation_l
    For k=3 and num_channels=[32, 64, 128] (3 levels with d=[1,2,4]):
        RF = 1 + 2 * (3 - 1) * (1 + 2 + 4) = 1 + 4 * 7 = 29 hours.
    """
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TCNExtractor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]
        self.receptive_field = 1 + sum(2 * (kernel_size - 1) * (2 ** i) for i in range(num_levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiTCN(nn.Module):
    """
    Complete Standalone TCN Classifier.
    Restores the linear classification head (fixes regression bug from legacy codebase).
    Maps: (B, C_in, T) -> (B, output_size).
    """
    def __init__(
        self,
        vital_signs: int,
        num_channels: List[int] = [32, 64, 128],
        output_size: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.2,
        time_samples: Optional[int] = None
    ):
        super(MultiTCN, self).__init__()
        self.tcn = TCNExtractor(
            num_inputs=vital_signs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, T)
        features = self.tcn(x)               # (B, C_last, T)
        last_timestep = features[:, :, -1]    # (B, C_last)
        logits = self.linear(last_timestep)   # (B, output_size)
        return logits


class TCNWithAttention(nn.Module):
    """
    Attention-Augmented TCN.
    Uses learnable additive attention over the temporal sequence of TCN feature maps:
        alpha_t = Softmax( v^T tanh(W * h_t + b) )
        c = sum_t alpha_t * h_t
    Maps: (B, C_in, T) -> (B, output_size).
    """
    def __init__(
        self,
        vital_signs: int,
        num_channels: List[int] = [32, 64, 128],
        output_size: int = 1,
        kernel_size: int = 3,
        attention_dim: int = 64,
        dropout: float = 0.2,
        time_samples: Optional[int] = None
    ):
        super(TCNWithAttention, self).__init__()
        self.tcn = TCNExtractor(
            num_inputs=vital_signs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        last_channel = num_channels[-1]

        self.attention = nn.Sequential(
            nn.Linear(last_channel, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.classifier = nn.Linear(last_channel, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, T)
        tcn_out = self.tcn(x)              # (B, C, T)
        tcn_out = tcn_out.transpose(1, 2)  # (B, T, C)

        attn_scores = self.attention(tcn_out)      # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, T, 1)

        context = torch.sum(attn_weights * tcn_out, dim=1) # (B, C)
        logits = self.classifier(context)                  # (B, output_size)
        return logits
