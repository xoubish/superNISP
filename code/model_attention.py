import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        batch, channels, height, width = x.size()
        depth = channels // self.num_heads

        # Reshape for multi-head attention by splitting channels into multiple heads
        q = self.query_conv(x).view(batch, self.num_heads, depth, height * width)
        k = self.key_conv(x).view(batch, self.num_heads, depth, height * width)
        v = self.value_conv(x).view(batch, self.num_heads, depth, height * width)

        # Change permutation to correctly align dimensions for multiplication
        q = q.permute(0, 1, 3, 2)  # Permute to align for dot product: [batch, num_heads, width*height, depth]
        k = k.permute(0, 1, 2, 3)  # Correct dimension for matmul: [batch, num_heads, depth, width*height]

        attention_scores = torch.matmul(q, k)  # Should be [batch, num_heads, width*height, width*height]
        attention = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, v.permute(0, 1, 3, 2))  # Adjust v for matmul: [batch, num_heads, width*height, depth]

        # Permute and reshape back to the original dimensions
        out = out.permute(0, 1, 3, 2).contiguous().view(batch, -1, height, width)  # Reshape to combine heads back into channels

        return out

class SuperResDiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, num_heads=4, activation_fn=nn.ReLU):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 4, 2, 1), activation_fn())
        self.attention1 = CrossAttention(hidden_dim, num_heads)
        self.decoder1 = nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.attention1(x)
        x = self.decoder1(x)
        return x

class Upsampler(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, upscale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.pixel_shuffle(self.conv1(x)))
        x = self.conv2(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, timesteps=500):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps

    def forward(self, x, t):
        return self.unet(x)

class SuperResolutionDiffusion(nn.Module):
    def __init__(self, unet_model, upsampler):
        super().__init__()
        self.upsampler = upsampler
        self.diffusion = DiffusionModel(unet_model)

    def forward(self, x, t):
        upscaled = self.upsampler(x)
        noise = torch.randn_like(upscaled, device=upscaled.device)
        alpha_t = cosine_schedule(t, self.diffusion.timesteps).view(-1, 1, 1, 1)
        noisy_image = alpha_t * upscaled + (1 - alpha_t) * noise
        output = self.diffusion(noisy_image, t)
        return nn.functional.interpolate(output, size=(66, 66), mode="bilinear", align_corners=True)

def cosine_schedule(t, total_timesteps=500):
    return torch.cos((t / total_timesteps) * (0.5 * torch.pi))
