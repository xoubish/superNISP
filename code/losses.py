import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import vgg16, VGG16_Weights

def gaussian_weight_map(shape, sigma=0.3, device=None):
    """Generates a Gaussian weight map centered in the middle of the image."""
    B, C, H, W = shape
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    d = torch.sqrt(x**2 + y**2)
    weights = torch.exp(- (d**2) / (2 * sigma**2))
    return weights.expand(B, C, H, W)

def compute_ellipticity_from_moments(img):
    """
    Computes (e1, e2) ellipticity for a batch of images.
    img: Tensor of shape (B, 1, H, W)
    """
    B, _, H, W = img.shape
    device = img.device

    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    x = x[None, None, :, :].expand(B, 1, H, W)
    y = y[None, None, :, :].expand(B, 1, H, W)

    # Total flux
    flux = img.sum(dim=[2, 3], keepdim=True)

    # Centroid
    x_bar = (img * x).sum(dim=[2, 3], keepdim=True) / (flux + 1e-8)
    y_bar = (img * y).sum(dim=[2, 3], keepdim=True) / (flux + 1e-8)

    # Centered coords
    dx = x - x_bar
    dy = y - y_bar

    # Moments
    Mxx = (img * dx**2).sum(dim=[2, 3]) / (flux.squeeze() + 1e-8)
    Myy = (img * dy**2).sum(dim=[2, 3]) / (flux.squeeze() + 1e-8)
    Mxy = (img * dx * dy).sum(dim=[2, 3]) / (flux.squeeze() + 1e-8)

    # e1 and e2 (ellipticity)
    e1 = (Mxx - Myy) / (Mxx + Myy + 1e-8)
    e2 = 2 * Mxy / (Mxx + Myy + 1e-8)

    return torch.stack([e1, e2], dim=1)  # shape: (B, 2)

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
    def __init__(self, mse_weight=1.0, l1_weight=0.01, perceptual_weight=0.01, shape_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.shape_weight = shape_weight
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        mse_loss = self.mse_weight * F.mse_loss(pred, target)
        l1_loss = self.l1_weight * F.l1_loss(pred, target)
        perceptual_loss = self.perceptual_weight * self.perceptual_loss(pred, target)

        # Differentiable shape loss
        if self.shape_weight > 0:
            e_pred = compute_ellipticity_from_moments(pred)
            e_true = compute_ellipticity_from_moments(target)
            shape_loss = self.shape_weight * F.mse_loss(e_pred, e_true)
        else:
            shape_loss = 0.0

        return mse_loss + l1_loss + perceptual_loss + shape_loss
