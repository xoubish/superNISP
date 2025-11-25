# model_sr3.py
#
# SR3-style conditional diffusion model for super-resolution.
#  - UNet with FiLM modulation from timestep + LR condition
#  - Cosine (Improved DDPM) beta schedule
#  - Works with arbitrary H,W (pads to multiple of 8, then crops back)
#
# Used together with train_sr3.py.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# SR3 / Improved-DDPM cosine beta schedule
# ------------------------------------------------------------------

def make_sr3_cosine_betas(num_timesteps: int, s: float = 0.008,
                          max_beta: float = 0.999) -> torch.Tensor:
    """
    SR3-style cosine schedule from Nichol & Dhariwal (Improved DDPM).

    Returns:
        betas: (T,) tensor of diffusion betas in [1e-8, max_beta].
    """
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1,
                           dtype=torch.float64)
    t = steps / num_timesteps

    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]

    alpha_bar_t = alpha_bar[:-1]
    alpha_bar_next = alpha_bar[1:]
    betas = 1.0 - (alpha_bar_next / alpha_bar_t)

    return torch.clamp(betas.float(), min=1e-8, max=max_beta)


cosine_beta_schedule = make_sr3_cosine_betas


# ------------------------------------------------------------------
# Sinusoidal timestep embedding
# ------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,)
        Returns: (B, dim)
        """
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device).float()
            / (half_dim - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


# ------------------------------------------------------------------
# Residual Block with FiLM time+condition modulation
# ------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_ch, groups=8):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)
        self.cond_proj = nn.Conv2d(cond_ch, out_ch * 2, 1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb, cond_feat):
        """
        x:        (B, in_ch, H, W)
        t_emb:    (B, time_emb_dim)
        cond_feat:(B, cond_ch, H, W)
        """
        B, _, H, W = x.shape

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM modulation from time embedding
        t_scale_shift = self.time_proj(t_emb).view(B, -1, 1, 1)
        t_scale, t_shift = torch.chunk(t_scale_shift, 2, dim=1)

        # FiLM modulation from condition features
        c_scale_shift = self.cond_proj(cond_feat)
        c_scale, c_shift = torch.chunk(c_scale_shift, 2, dim=1)

        h = self.norm2(h)
        h = (1.0 + t_scale + c_scale) * h + (t_shift + c_shift)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ------------------------------------------------------------------
# SR3 UNet (pads to multiple of 8, then crops back)
# ------------------------------------------------------------------

class SR3UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 4),
        time_emb_dim: int = 256,
    ):
        super().__init__()

        # --- time embedding ---
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # --- input convs ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.cond_conv_in = nn.Conv2d(cond_channels, base_channels, 3, padding=1)

        chs = [base_channels]
        in_ch = base_channels
        cond_in_ch = base_channels

        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()
        self.cond_projs = nn.ModuleList()

        # -----------------------------
        # Down path
        # -----------------------------
        for mult in channel_mults:
            out_ch = base_channels * mult

            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(in_ch, out_ch, time_emb_dim, cond_ch=out_ch),
                        ResBlock(out_ch, out_ch, time_emb_dim, cond_ch=out_ch),
                        nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                    ]
                )
            )

            self.cond_projs.append(nn.Conv2d(cond_in_ch, out_ch, 1))
            self.cond_downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

            chs.append(out_ch)
            in_ch = out_ch
            cond_in_ch = out_ch

        # bottleneck
        self.bottleneck1 = ResBlock(in_ch, in_ch, time_emb_dim, cond_ch=in_ch)
        self.bottleneck2 = ResBlock(in_ch, in_ch, time_emb_dim, cond_ch=in_ch)

        # -----------------------------
        # Up path
        # -----------------------------
        self.ups = nn.ModuleList()
        self.cond_ups = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult

            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                        ResBlock(out_ch + chs.pop(), out_ch, time_emb_dim, cond_ch=out_ch),
                        ResBlock(out_ch, out_ch, time_emb_dim, cond_ch=out_ch),
                    ]
                )
            )
            self.cond_ups.append(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
            )
            in_ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t, cond):
        """
        x:    (B,1,H,W) noisy HR image at timestep t
        cond: (B,1,H,W) LR image upsampled to HR grid
        """
        orig_H, orig_W = x.shape[2], x.shape[3]

        # --- pad to multiple of 8 ---
        pad_h = (8 - orig_H % 8) % 8
        pad_w = (8 - orig_W % 8) % 8
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode="reflect")
            cond = F.pad(cond, (pad_left, pad_right, pad_top, pad_bot), mode="reflect")

        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)

        # time embedding
        t_emb = self.time_embedding(t)

        # condition embedding
        c = self.cond_conv_in(cond)

        hs = []
        h = self.input_conv(x)
        hs.append(h)

        # down path
        for (res1, res2, down), c_proj, c_down in zip(
            self.downs, self.cond_projs, self.cond_downs
        ):
            c = c_proj(c)
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)

            h = res1(h, t_emb, c)
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)

            h = res2(h, t_emb, c)
            hs.append(h)

            h = down(h)
            c = c_down(c)

        # bottleneck
        c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.bottleneck1(h, t_emb, c)
        c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.bottleneck2(h, t_emb, c)

        # up path
        for (up, res1, res2), c_up in zip(self.ups, self.cond_ups):
            h = up(h)

            skip = hs.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)

            c = c_up(c)
            if c.shape[2:] != h.shape[2:]:
                c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)

            h = res1(h, t_emb, c)
            h = res2(h, t_emb, c)

        out = self.out_conv(h)

        # crop back to original HR size
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, pad_top : pad_top + orig_H, pad_left : pad_left + orig_W]

        return out


# ------------------------------------------------------------------
# SR3 Diffusion wrapper
# ------------------------------------------------------------------

class SR3SuperResolution(nn.Module):
    """
    SR3 diffusion wrapper.

    Typical usage:
        model = SR3SuperResolution(unet, timesteps=1000, upscale_factor=5)
        pred_eps, true_eps, x0_pred = model(lr, hr, t)

    Sampling:
        sr = model.sample(lr, num_steps=100, deterministic=True, init_sigma=1.0)
    """

    def __init__(self, unet: SR3UNet, timesteps: int = 1000, upscale_factor: int = 5):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.upscale_factor = upscale_factor

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        alphas_cum_prev = torch.cat(
            [torch.tensor([1.0], dtype=alphas.dtype), alphas_cum[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cum)
        self.register_buffer("alphas_cumprod_prev", alphas_cum_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cum))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cum)
        )

        posterior_var = betas * (1.0 - alphas_cum_prev) / (1.0 - alphas_cum)
        posterior_var = torch.clamp(posterior_var, min=1e-20)

        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(posterior_var)
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cum_prev) / (1.0 - alphas_cum),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cum_prev) * torch.sqrt(alphas) / (1.0 - alphas_cum),
        )

    # -----------------------------------------------------
    # Forward diffusion q(x_t | x_0)
    # -----------------------------------------------------

    def q_sample(self, x0, t, noise=None):
        """
        x0: (B,1,H,W)
        t:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_om * noise, noise

    # -----------------------------------------------------
    # Training: model predicts epsilon
    # -----------------------------------------------------

    def forward(self, lr, hr, t):
        """
        lr: (B,1,21,21)  low-res input (already normalized)
        hr: (B,1,105,105) high-res target (already normalized)
        t:  (B,) timesteps
        """
        # upsample LR → HR grid for conditioning
        cond = F.interpolate(
            lr, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False
        )

        # ensure hr matches cond size (if for some reason it doesn't)
        if hr.shape[2:] != cond.shape[2:]:
            hr_up = F.interpolate(
                hr, size=cond.shape[2:], mode="bilinear", align_corners=False
            )
        else:
            hr_up = hr

        # forward diffusion
        x_t, noise = self.q_sample(hr_up, t)

        # predict noise
        pred_noise = self.unet(x_t, t, cond)

        # align shapes if needed
        if pred_noise.shape[2:] != noise.shape[2:]:
            pred_noise = F.interpolate(
                pred_noise, size=noise.shape[2:], mode="bilinear", align_corners=False
            )

        # reconstruct x0 for perceptual loss
        B = x_t.shape[0]
        sqrt_a = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        x0_pred = (x_t - sqrt_om * pred_noise) / (sqrt_a + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return pred_noise, noise, x0_pred

    # -----------------------------------------------------
    # Reverse step helpers
    # -----------------------------------------------------

    @torch.no_grad()
    def p_mean_variance(self, lr, x_t, t):
        """
        Compute posterior mean & log variance for q(x_{t-1} | x_t, x0)
        via epsilon prediction.
        """
        B = x_t.shape[0]

        cond = F.interpolate(
            lr, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False
        )
        if cond.shape[2:] != x_t.shape[2:]:
            cond = F.interpolate(
                cond, size=x_t.shape[2:], mode="bilinear", align_corners=False
            )

        eps_theta = self.unet(x_t, t, cond)
        if eps_theta.shape[2:] != x_t.shape[2:]:
            eps_theta = F.interpolate(
                eps_theta, size=x_t.shape[2:], mode="bilinear", align_corners=False
            )

        sqrt_a = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        x0_pred = (x_t - sqrt_om * eps_theta) / (sqrt_a + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        coef1 = self.posterior_mean_coef1[t].view(B, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(B, 1, 1, 1)
        mean = coef1 * x0_pred + coef2 * x_t

        log_var = self.posterior_log_variance_clipped[t].view(B, 1, 1, 1)
        return mean, log_var, x0_pred

    @torch.no_grad()
    def p_sample(self, lr, x_t, t, deterministic: bool = False):
        """
        One reverse diffusion step.

        If deterministic=True, no extra noise is added (DDIM-like step).
        """
        mean, log_var, x0_pred = self.p_mean_variance(lr, x_t, t)

        if deterministic:
            return mean, x0_pred

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev, x0_pred

    # -----------------------------------------------------
    # Full sampling loop
    # -----------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        lr: torch.Tensor,
        num_steps: int | None = None,
        deterministic: bool = True,
        init_sigma: float = 1.0,
    ):
        """
        lr: (B,1,21,21)
        num_steps: how many reverse steps to take. If None or >= T, uses full chain.
        deterministic: if True, don't add noise in reverse steps.
        init_sigma: initialize x_T as LR_up + sigma * N(0,1). If 0, start from LR_up.

        Returns:
            x_0 samples: (B,1,H_hr,W_hr)
        """
        B = lr.shape[0]
        device = lr.device

        if num_steps is None or num_steps >= self.timesteps:
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            timesteps = torch.linspace(
                self.timesteps - 1, 0, num_steps, device=device
            ).long()

        # HR conditioning grid
        cond_up = F.interpolate(
            lr, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False
        )

        # initialize x_T around LR_up
        if init_sigma > 0:
            x_t = cond_up + init_sigma * torch.randn_like(cond_up)
        else:
            x_t = cond_up.clone()

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t, _ = self.p_sample(lr, x_t, t, deterministic=deterministic)

        return x_t
