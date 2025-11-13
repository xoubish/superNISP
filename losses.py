import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import vgg16, VGG16_Weights

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class SuperResDiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, activation_fn=nn.ReLU, timestep_embed_dim=128):
        super().__init__()
        self.activation_fn = activation_fn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.timestep_embed_dim = timestep_embed_dim
        self.time_embed = SinusoidalPositionalEmbedding(timestep_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim * 4),
            self.activation_fn(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4)
        )

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            self.activation_fn()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            self.activation_fn()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            self.activation_fn()
        )

        # Condition Projection
        self.condition_proj = nn.Conv2d(1, self.hidden_dim * 4, kernel_size=3, padding=1)

        # Cross-Attention (improved: concatenate and project)
        self.cross_attention = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 4 * 2, self.hidden_dim * 4, kernel_size=1),
            self.activation_fn(),
            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 4, kernel_size=3, padding=1)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim * 4, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1, output_padding=1),
            self.activation_fn()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=4, stride=2, padding=1, output_padding=1),
            self.activation_fn()
        )
        self.decoder3 = nn.Conv2d(self.hidden_dim, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, condition, t_emb=None):
        # Encode
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Process condition
        condition = self.condition_proj(condition)
        condition = F.interpolate(condition, size=(x3.shape[2], x3.shape[3]), mode="bilinear", align_corners=True)

        # Apply timestep embedding if provided
        if t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            # Reshape to spatial dimensions
            B, C = t_emb.shape
            H, W = x3.shape[2], x3.shape[3]
            t_emb = t_emb.view(B, C, 1, 1).expand(B, C, H, W)
            # Add timestep information to the bottleneck
            x3 = x3 + t_emb[:, :self.hidden_dim * 4, :, :]

        # Cross-attention (concatenation + projection)
        x3 = torch.cat([x3, condition], dim=1)
        x3 = self.cross_attention(x3)

        # Decode with skip connections
        x = self.decoder1(x3)
        x = self.align_dims(x, x2)
        x = x + x2

        x = self.decoder2(x)
        x = self.align_dims(x, x1)
        x = x + x1

        x = self.decoder3(x)
        return x

    def align_dims(self, x, target):
        if x.size() != target.size():
            diffY = target.size(2) - x.size(2)
            diffX = target.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x

class Upsampler(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels * (self.upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.pixel_shuffle(self.conv1(x)))
        x = self.conv2(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, timesteps=500):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps

    def forward(self, x, t, condition):
        # Get timestep embedding
        t_emb = self.unet.time_embed(t.float())
        # UNet predicts noise (for diffusion training)
        return self.unet(x, condition, t_emb)

class SuperResolutionDiffusion(nn.Module):
    def __init__(self, unet_model, upsampler, timesteps=500, output_size=None):
        super().__init__()
        self.upsampler = upsampler
        self.diffusion = DiffusionModel(unet_model, timesteps)
        self.timesteps = timesteps
        self.output_size = output_size  # Make configurable

    def forward(self, x, t, training=True):
        """
        Forward pass for diffusion-based super-resolution.
        
        Args:
            x: Low-resolution input
            t: Timestep tensor
            training: If True, add noise and predict noise (training mode).
                     If False, predict denoised image (inference mode).
        """
        upscaled = self.upsampler(x)
        
        if training:
            # Training: add noise and predict the noise
            noise = torch.randn_like(upscaled, device=upscaled.device)
            alpha_t = cosine_schedule(t, self.timesteps)
            # Ensure alpha_t has correct shape
            if alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1, 1)
            noisy_image = alpha_t * upscaled + (1 - alpha_t) * noise
            # Predict noise (standard diffusion training)
            predicted_noise = self.diffusion(noisy_image, t, upscaled)
            # Return predicted noise for loss computation
            return predicted_noise, noise
        else:
            # Inference: predict denoised image
            # For inference, we typically start with noise or use the upscaled image
            output = self.diffusion(upscaled, t, upscaled)
            # Resize to target size if specified
            if self.output_size is not None:
                output = F.interpolate(output, size=self.output_size, mode="bilinear", align_corners=True)
            return output

# Diffusion noise schedule
def cosine_schedule(t, total_timesteps=500):
    """Cosine noise schedule for diffusion."""
    return torch.cos((t.float() / total_timesteps) * (0.5 * torch.pi))

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
