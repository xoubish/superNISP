import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperResDiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, num_heads=2):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Condition Projection (Ensures `condition` is the same shape as `x3`)
        self.condition_proj = nn.Conv2d(1, hidden_dim * 4, kernel_size=3, padding=1)

        # ✅ Fix: Cross-Attention Without Flattening
        self.cross_attention = nn.Conv2d(hidden_dim * 4 * 2, hidden_dim * 4, kernel_size=1)

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.decoder3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x, condition):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        condition = self.condition_proj(condition)

        # ✅ Fix: Ensure `condition` is the same size as `x3`
        condition = F.interpolate(condition, size=(x3.shape[2], x3.shape[3]), mode="bilinear", align_corners=True)

        # ✅ Now Concatenation Works
        x3 = torch.cat([x3, condition], dim=1)  
        x3 = self.cross_attention(x3)  

        # Decoder
        x = self.decoder1(x3)
        if x.shape != x2.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x = x + x2  # Skip connection

        x = self.decoder2(x)
        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x = x + x1  # Skip connection

        x = self.decoder3(x)

        return x

# Upsampler that ensures output is exactly 66×66
class Upsampler(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # ✅ Anti-aliasing upsampling
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        return self.upsample(x)


# Diffusion Model with Cosine Noise Schedule
def cosine_schedule(t, total_timesteps=500):
    return torch.cos((t / total_timesteps) * (0.5 * torch.pi))


class DiffusionModel(nn.Module):
    def __init__(self, unet_model, timesteps=500):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps

    def forward(self, x, t, condition):
        x = x.to(next(self.parameters()).device)
        t = t.to(next(self.parameters()).device)
        condition = condition.to(next(self.parameters()).device)
        return self.unet(x, condition)


# Full Super-Resolution Diffusion Model
class SuperResolutionDiffusion(nn.Module):
    def __init__(self, unet_model, upsampler):
        super().__init__()
        self.upsampler = upsampler
        self.diffusion = DiffusionModel(unet_model)

    def forward(self, x, t):
        x = x.to(next(self.parameters()).device)
        t = t.to(next(self.parameters()).device)

        upscaled = self.upsampler(x)

        # Add progressive noise with cosine schedule
        noise = torch.randn_like(upscaled, device=upscaled.device)
        alpha_t = cosine_schedule(t, self.diffusion.timesteps).view(-1, 1, 1, 1)
        noisy_image = alpha_t * upscaled + (1 - alpha_t) * noise

        output = self.diffusion(noisy_image, t, upscaled)

        # Final size correction
        return nn.functional.interpolate(output, size=(66, 66), mode="bilinear", align_corners=True)
