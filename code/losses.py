import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import vgg16, VGG16_Weights

def compute_centroid(img):
    """
    Computes the centroid (center of mass) for a batch of images.
    img: Tensor of shape (B, C, H, W)
    Returns: (x_bar, y_bar) each of shape (B, C)
    """
    B, C, H, W = img.shape
    device = img.device
    
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    x = x[None, :, :].expand(B, C, H, W)
    y = y[None, :, :].expand(B, C, H, W)
    
    # Total flux
    flux = img.sum(dim=[2, 3], keepdim=True)  # Shape: (B, C, 1, 1)
    
    # Centroid
    x_bar = (img * x).sum(dim=[2, 3], keepdim=True) / (flux + 1e-8)  # Shape: (B, C, 1, 1)
    y_bar = (img * y).sum(dim=[2, 3], keepdim=True) / (flux + 1e-8)  # Shape: (B, C, 1, 1)
    
    return x_bar, y_bar

def gaussian_weight_map(shape, sigma=0.3, device=None, center_img=None):
    """
    Generates a Gaussian weight map.
    If center_img is provided, centers the weight on the galaxy centroid.
    Otherwise centers at image center.
    """
    B, C, H, W = shape
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create coordinate grids in pixel space
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    x = x[None, None, :, :].expand(B, C, H, W)  # Shape: (B, C, H, W)
    y = y[None, None, :, :].expand(B, C, H, W)  # Shape: (B, C, H, W)
    
    # Determine center: use galaxy centroid if provided, otherwise image center
    if center_img is not None:
        x_bar, y_bar = compute_centroid(center_img)
        # Expand to match spatial dimensions
        x_bar = x_bar.expand(B, C, H, W)
        y_bar = y_bar.expand(B, C, H, W)
    else:
        # Default to image center
        x_bar = torch.full((B, C, H, W), W / 2.0, device=device)
        y_bar = torch.full((B, C, H, W), H / 2.0, device=device)
    
    # Distance from center
    dx = x - x_bar
    dy = y - y_bar
    d = torch.sqrt(dx**2 + dy**2)
    
    # Normalize by image size to make sigma scale-invariant
    max_dist = torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32, device=device))
    d_normalized = d / max_dist
    
    # Gaussian weight
    weights = torch.exp(- (d_normalized**2) / (2 * sigma**2))
    
    # Add a minimum weight to background (so it's not completely ignored)
    min_weight = 0.1
    weights = weights * (1 - min_weight) + min_weight
    
    return weights

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

    # Total flux - sum over spatial dimensions
    flux = img.sum(dim=[2, 3])  # Shape: (B, 1)

    # Centroid
    x_bar = (img * x).sum(dim=[2, 3]) / (flux + 1e-8)  # Shape: (B, 1)
    y_bar = (img * y).sum(dim=[2, 3]) / (flux + 1e-8)  # Shape: (B, 1)

    # Centered coords
    dx = x - x_bar.unsqueeze(-1).unsqueeze(-1)  # Broadcast: (B, 1, H, W)
    dy = y - y_bar.unsqueeze(-1).unsqueeze(-1)  # Broadcast: (B, 1, H, W)

    # Moments - sum over spatial dimensions
    Mxx = (img * dx**2).sum(dim=[2, 3]) / (flux + 1e-8)  # Shape: (B, 1)
    Myy = (img * dy**2).sum(dim=[2, 3]) / (flux + 1e-8)  # Shape: (B, 1)
    Mxy = (img * dx * dy).sum(dim=[2, 3]) / (flux + 1e-8)  # Shape: (B, 1)

    # Squeeze the channel dimension to get (B,)
    Mxx = Mxx.squeeze(1)  # Shape: (B,)
    Myy = Myy.squeeze(1)  # Shape: (B,)
    Mxy = Mxy.squeeze(1)  # Shape: (B,)

    # e1 and e2 (ellipticity)
    e1 = (Mxx - Myy) / (Mxx + Myy + 1e-8)  # Shape: (B,)
    e2 = 2 * Mxy / (Mxx + Myy + 1e-8)  # Shape: (B,)

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
