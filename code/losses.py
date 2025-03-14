import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

def gaussian_weight_map(shape, sigma=0.3):
    """Generates a Gaussian weight map centered in the middle of the image."""
    B, C, H, W = shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device='cuda'), torch.linspace(-1, 1, W, device='cuda'))
    d = torch.sqrt(x**2 + y**2)
    weights = torch.exp(- (d**2) / (2 * sigma**2))
    return weights.expand(B, C, H, W)

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 for structure preservation."""
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:5]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred.repeat(1, 3, 1, 1))
        target_features = self.feature_extractor(target.repeat(1, 3, 1, 1))
        return F.mse_loss(pred_features, target_features)

class HybridLoss(nn.Module):
    """Hybrid loss: Configurable weights for MSE, L1, and Perceptual Loss"""
    def __init__(self, mse_weight=1.0, l1_weight=0.01, perceptual_weight=0.01):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        mse_loss = self.mse_weight * F.mse_loss(pred, target)
        l1_loss = self.l1_weight * F.l1_loss(pred, target)
        perceptual_loss = self.perceptual_weight * self.perceptual_loss(pred, target)
        return mse_loss + l1_loss + perceptual_loss

