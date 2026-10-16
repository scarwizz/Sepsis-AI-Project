"""
Clinically Appropriate Loss Functions for Imbalanced Sepsis Time-Series Prediction.
Includes:
- Mathematically correct alpha-balanced Focal Loss (Lin et al. 2017)
- Focal Tversky Loss (Salehi et al. 2017)
- Pairwise AUCPR ranking surrogate loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Mathematically rigorous Alpha-Balanced Focal Loss for binary classification.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        where:
            p_t = p if y=1 else (1 - p)
            alpha_t = alpha if y=1 else (1 - alpha)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw unbounded logits from model of shape (N, 1) or (N,).
            targets: Binary ground truth labels {0, 1} of matching shape.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # Calculate p_t
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        
        # Calculate alpha_t: alpha for positive class, (1 - alpha) for negative class
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        
        # Modulating factor
        focal_weight = alpha_t * torch.pow((1.0 - pt).clamp(min=0.0, max=1.0), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for optimizing precision-recall tradeoff in severe class imbalance.
    Reference: Salehi et al., 2017.
    
    Tversky Index:
        TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    Loss:
        FTL = (1 - TI)^gamma
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 2.0, smooth: float = 1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Weight on false positives (penalizes low precision)
        self.beta = beta    # Weight on false negatives (penalizes low recall)
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1).float()

        tp = (probs * targets).sum()
        fp = ((1.0 - targets) * probs).sum()
        fn = (targets * (1.0 - probs)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky).clamp(min=0.0, max=1.0), self.gamma)

        return focal_tversky


class PairwiseAUCPRLoss(nn.Module):
    """
    Pairwise hinge ranking surrogate loss that directly penalizes negative samples scored above positive samples.
    """
    def __init__(self, margin: float = 1.0):
        super(PairwiseAUCPRLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        pos_probs = probs[targets == 1]
        neg_probs = probs[targets == 0]

        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        pos_expanded = pos_probs.unsqueeze(1)  # (N_pos, 1)
        neg_expanded = neg_probs.unsqueeze(0)  # (1, N_neg)

        # Hinge loss: want pos_score >= neg_score + margin
        loss = F.relu(self.margin - (pos_expanded - neg_expanded)).mean()
        return loss
