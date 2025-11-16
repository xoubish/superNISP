# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Positional embedding
# ------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: (B,)
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        arg = time[:, None].float() * freq[None, :]
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (B, dim)
        return emb


# ------------------------------------------------------------
# UNet backbone
# ------------------------------------------------------------

class SuperResDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_dim=64,
        activation_fn=nn.ReLU,
        timestep_embed_dim=128,
    ):
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
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
        )

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            self.activation_fn(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            self.activation_fn(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            self.activation_fn(),
        )

        # Multi-scale conditioning
        self.condition_proj1 = nn.Conv2d(1, self.hidden_dim, kernel_size=3, padding=1)
        self.condition_proj2 = nn.Conv2d(1, self.hidden_dim * 2, kernel_size=3, padding=1)
        self.condition_proj3 = nn.Conv2d(1, self.hidden_dim * 4, kernel_size=3, padding=1)
        self.condition_proj_dec1 = nn.Conv2d(1, self.hidden_dim * 2, kernel_size=3, padding=1)
        self.condition_proj_dec2 = nn.Conv2d(1, self.hidden_dim, kernel_size=3, padding=1)

        # Bottleneck fusion
        self.cross_attention = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 4 * 2, self.hidden_dim * 4, kernel_size=1),
            self.activation_fn(),
            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 4, kernel_size=3, padding=1),
            self.activation_fn(),
        )

        # Decoder fusion modules
        self.fusion_dec1 = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, kernel_size=1),
            self.activation_fn(),
        )
        self.fusion_dec2 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            self.activation_fn(),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dim * 4,
                self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.activation_fn(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dim * 2,
                self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.activation_fn(),
        )
        self.decoder3 = nn.Conv2d(self.hidden_dim, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, condition, t_emb):
        # Encode
        x1 = self.encoder1(x)
        cond1 = self.condition_proj1(condition)
        cond1 = F.interpolate(cond1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x1 = x1 + cond1

        x2 = self.encoder2(x1)
        cond2 = self.condition_proj2(condition)
        cond2 = F.interpolate(cond2, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x2 = x2 + cond2

        x3 = self.encoder3(x2)
        cond3 = self.condition_proj3(condition)
        cond3 = F.interpolate(cond3, size=x3.shape[2:], mode="bilinear", align_corners=True)

        # Timestep embedding at bottleneck
        if t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            B, C = t_emb.shape
            H, W = x3.shape[2], x3.shape[3]
            t_emb = t_emb.view(B, C, 1, 1).expand(B, C, H, W)
            x3 = x3 + t_emb[:, : self.hidden_dim * 4, :, :]

        # Bottleneck fusion
        x3 = torch.cat([x3, cond3], dim=1)
        x3 = self.cross_attention(x3)

        # Decode
        x = self.decoder1(x3)
        x = self.align_dims(x, x2)
        cond_dec1 = self.condition_proj_dec1(condition)
        cond_dec1 = F.interpolate(cond_dec1, size=x.shape[2:], mode="bilinear", align_corners=True)
        x = x + x2 + self.fusion_dec1(cond_dec1)

        x = self.decoder2(x)
        x = self.align_dims(x, x1)
        cond_dec2 = self.condition_proj_dec2(condition)
        cond_dec2 = F.interpolate(cond_dec2, size=x.shape[2:], mode="bilinear", align_corners=True)
        x = x + x1 + self.fusion_dec2(cond_dec2)

        x = self.decoder3(x)
        return x

    @staticmethod
    def align_dims(x, target):
        if x.size() != target.size():
            diffY = target.size(2) - x.size(2)
            diffX = target.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x


# ------------------------------------------------------------
# Upsampler: pure bilinear 5×
# ------------------------------------------------------------

class Upsampler(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, upscale_factor=5):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )


# ------------------------------------------------------------
# Diffusion wrapper around UNet
# ------------------------------------------------------------

class DiffusionModel(nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model

    def forward(self, x, t, condition):
        # x: noisy image, condition: upscaled LR
        t_emb = self.unet.time_embed(t)
        return self.unet(x, condition, t_emb)


# ------------------------------------------------------------
# Noise schedule
# ------------------------------------------------------------

def cosine_schedule(t, total_timesteps=500):
    """Cosine schedule interpreted directly as alphā_t."""
    return torch.cos((t.float() / total_timesteps) * (0.5 * torch.pi))


# ------------------------------------------------------------
# Super-resolution diffusion model
# ------------------------------------------------------------

class SuperResolutionDiffusion(nn.Module):
    def __init__(
        self,
        unet_model,
        upsampler,
        timesteps=500,
        output_size=None,
        inference_timesteps=None,
    ):
        super().__init__()
        self.upsampler = upsampler
        self.diffusion = DiffusionModel(unet_model)
        self.timesteps = timesteps
        self.output_size = output_size
        self.inference_timesteps = inference_timesteps if inference_timesteps is not None else timesteps

        # alphā_t from cosine schedule
        t_all = torch.arange(self.timesteps, dtype=torch.long)
        alpha_bar = cosine_schedule(t_all, self.timesteps)  # (T,)
        alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=0.99999)

        alpha_prev = torch.ones_like(alpha_bar)
        alpha_prev[1:] = alpha_bar[:-1]
        alphas = alpha_bar / alpha_prev
        betas = 1.0 - alphas

        self.register_buffer("alphas_cumprod", alpha_bar)
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alpha_bar),
        )

    # -------- forward diffusion q(x_t | x_0) --------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

        # -------- SAMPLING LOOP (DDIM-like) --------
    @torch.no_grad()
    def sample(self, lr, num_steps=None):
        """
        Generate high-res sample given low-res input.

        lr: (B, 1, 25, 25)
        returns: (B, 1, HR, HR)  (HR ~ 125x125)
        """
        if num_steps is None:
            num_steps = self.inference_timesteps

        device = lr.device
        B = lr.shape[0]

        # conditioning image (upsampled LR)
        cond = self.upsampler(lr)           # (B, 1, Hc, Wc)
        x = torch.randn_like(cond)          # start from pure noise

        # map num_steps onto training timesteps [0 .. self.timesteps-1]
        step_size = max(self.timesteps // num_steps, 1)

        for i in reversed(range(num_steps)):
            t_val = i * step_size
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # predict noise ε_t = ε_θ(x_t, t, cond)
            eps = self.diffusion(x, t, cond)

            # make sure eps has same spatial size as x
            if eps.shape[2:] != x.shape[2:]:
                eps = F.interpolate(
                    eps,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            # current alphā_t and its complement
            sqrt_ab_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

            # estimate x0 from x_t and ε̂
            x0_hat = (x - sqrt_one_minus_t * eps) / (sqrt_ab_t + 1e-8)

            if i > 0:
                # go to "previous" timestep t_prev (DDIM-style deterministic step)
                t_prev_val = (i - 1) * step_size
                t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

                sqrt_ab_prev = self.sqrt_alphas_cumprod[t_prev].view(B, 1, 1, 1)
                sqrt_one_minus_prev = self.sqrt_one_minus_alphas_cumprod[t_prev].view(B, 1, 1, 1)

                # deterministic DDIM-like update:
                # x_{t_prev} = sqrt(alphā_{t_prev}) * x0_hat + sqrt(1 - alphā_{t_prev}) * ε̂
                x = sqrt_ab_prev * x0_hat + sqrt_one_minus_prev * eps
            else:
                # final step: return x0_hat
                x = x0_hat

        # final resize if a specific output_size is requested
        if self.output_size is not None and x.shape[2:] != tuple(self.output_size):
            x = F.interpolate(
                x,
                size=self.output_size,
                mode="bilinear",
                align_corners=True,
            )

        return x
