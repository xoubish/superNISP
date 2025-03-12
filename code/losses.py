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
    """Simplified hybrid loss: Weighted MSE + Optional Perceptual Loss."""
    def __init__(self, center_boost=3.0, base_weight=1.0, perceptual_weight=0.01):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.center_boost = center_boost
        self.base_weight = base_weight
        self.perceptual_weight = perceptual_weight  # Lower weight for perceptual loss

    def forward(self, pred, target):
        # Weighted MSE Loss (Prioritizes Galaxy)
        weight_map = gaussian_weight_map(target.shape, sigma=0.4)
        weight_map = self.base_weight + (self.center_boost - self.base_weight) * weight_map
        mse_loss = (weight_map * (pred - target) ** 2).mean()

        # Optional Perceptual Loss (Add only if needed)
        perceptual_loss = self.perceptual_weight * self.perceptual_loss(pred, target) if self.perceptual_weight > 0 else 0
        
        return mse_loss + perceptual_loss
