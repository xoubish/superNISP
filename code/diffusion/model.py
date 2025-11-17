# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Cosine Noise Schedule ================================================

def cosine_schedule(t, total_timesteps=500):
    """
    Cosine schedule interpreted as alpha_t in [0,1].

    t: (B,) or scalar tensor
    """
    t = t.float()
    return torch.cos((t / float(total_timesteps)) * (0.5 * torch.pi))


# === Timestep Embedding ====================================================

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Standard sinusoidal timestep embedding used in diffusion models.

    timesteps: (B,)
    returns: (B, embedding_dim)
    """
    half_dim = embedding_dim // 2
    exponent = -torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device
    ) * (math.log(10000.0) / (half_dim - 1))

    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent.unsqueeze(0))
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


# === UNet with Condition + Time Embedding =================================

class SuperResDiffusionUNet(nn.Module):
    """
    UNet that takes:
      - x_t: noisy HR image
      - condition: upsampled LR image
      - t_embed: timestep embedding

    and outputs a denoised / super-resolved HR image.
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_dim=64,
        activation_fn=nn.ReLU,
        time_embed_dim=128,
    ):
        super().__init__()
        self.activation_fn = activation_fn

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),  # 125 -> 62
            activation_fn()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),  # 62 -> 31
            activation_fn()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),  # 31 -> 15
            activation_fn()
        )

        # Conditioning and time
        self.condition_proj = nn.Conv2d(1, hidden_dim * 4, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim * 4),
            activation_fn()
        )
        self.cross_attention = nn.Conv2d(hidden_dim * 4 * 2, hidden_dim * 4, kernel_size=1)

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            activation_fn()
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            activation_fn()
        )
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            activation_fn()
        )

        self.output_conv = nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, condition, t_embed):
        """
        x:         (B, 1, H, W) noisy image (HR grid)
        condition: (B, 1, H, W) upsampled LR image
        t_embed:   (B, D) timestep embedding
        """
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Project condition to bottleneck resolution
        condition_proj = self.condition_proj(condition)
        condition_proj = F.interpolate(
            condition_proj,
            size=(x3.shape[2], x3.shape[3]),
            mode="bilinear",
            align_corners=True,
        )

        # Time embedding → spatial
        t_proj = self.time_mlp(t_embed).view(
            x3.shape[0], -1, 1, 1
        ).expand(-1, -1, x3.shape[2], x3.shape[3])

        # Fuse x3, t, and condition
        x3 = torch.cat([x3 + t_proj, condition_proj], dim=1)
        x3 = self.cross_attention(x3)

        # Decoder with skip connections
        x = self.decoder1(x3)
        x = self.align_dims(x, x2)
        x = x + x2

        x = self.decoder2(x)
        x = self.align_dims(x, x1)
        x = x + x1

        x = self.decoder3(x)
        x = self.output_conv(x)

        # Final align to 125x125
        return self.align_dims(x, target=(125, 125))

    def align_dims(self, x, target):
        """
        Align 'x' spatial dims to 'target':
          - if target is tensor, use its H,W
          - if tuple, treat as (H, W)
        """
        if isinstance(target, torch.Tensor):
            target_h, target_w = target.size(2), target.size(3)
        else:
            target_h, target_w = target[0], target[1]

        diffY = target_h - x.size(2)
        diffX = target_w - x.size(3)
        return F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )


# === Upsampler Module (25 -> 125) =========================================

class Upsampler(nn.Module):
    """
    Learnable upsampler: 25x25 → 125x125 via PixelShuffle + final bilinear.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 25 -> 50
            nn.ReLU(),
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 50 -> 100
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.block(x)
        return F.interpolate(x, size=(125, 125), mode='bilinear', align_corners=False)


# === Diffusion Model Wrapper ==============================================

class DiffusionModel(nn.Module):
    """
    Wraps the UNet and handles timestep embeddings.
    """
    def __init__(self, unet_model, timesteps=500, time_embed_dim=128):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps
        self.time_embed_dim = time_embed_dim

    def forward(self, x, t, condition):
        """
        x:         (B,1,H,W) noisy image
        t:         (B,) timesteps
        condition: (B,1,H,W) upsampled LR image
        """
        t_embed = get_timestep_embedding(t, self.time_embed_dim)
        return self.unet(x, condition, t_embed)


# === Full Super-Resolution Diffusion-Inspired Model =======================

class SuperResolutionDiffusion(nn.Module):
    """
    Noise-conditioned super-resolution model:

      - Takes LR image x (25x25)
      - Upsamples to HR grid (125x125)
      - Adds noise according to cosine schedule at timestep t
      - UNet denoises/refines to output HR

    Not a full DDPM sampler; t is used as a noise-level conditioning variable.
    """
    def __init__(self, unet_model, upsampler, timesteps=500, time_embed_dim=128):
        super().__init__()
        self.upsampler = upsampler
        self.diffusion = DiffusionModel(
            unet_model,
            timesteps=timesteps,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, x, t, add_noise=True):
        """
        x:         (B,1,25,25) low-res input
        t:         (B,) noise level indices
        add_noise: if False, skip adding noise and just pass upscaled LR

        returns: (B,1,125,125) predicted HR
        """
        upscaled = self.upsampler(x)  # (B,1,125,125)

        if add_noise:
            noise = torch.randn_like(upscaled, device=upscaled.device)
            alpha_t = cosine_schedule(t, self.diffusion.timesteps).view(-1, 1, 1, 1)
            noisy = alpha_t * upscaled + (1.0 - alpha_t) * noise
        else:
            noisy = upscaled

        return self.diffusion(noisy, t, upscaled)
