# losses.py

import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """
    Simple combination of MSE and L1.
    You can extend this later with perceptual / shape terms if you want.
    """

    def __init__(self, mse_weight=1.0, l1_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        loss = 0.0
        if self.mse_weight > 0:
            loss = loss + self.mse_weight * F.mse_loss(pred, target)
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * F.l1_loss(pred, target)
        return loss
