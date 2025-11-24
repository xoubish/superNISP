import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# SR3 / Improved-DDPM cosine beta schedule
# ------------------------------------------------------------------

def make_sr3_cosine_betas(num_timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """
    SR3-style cosine schedule from Nichol & Dhariwal (Improved DDPM).
    Returns:
      betas: (T,) tensor of diffusion betas in [0, max_beta].
    """
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
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
        """
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device).float()
            / (half_dim - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


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
        B, _, H, W = x.shape

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM modulation
        t_scale_shift = self.time_proj(t_emb).view(B, -1, 1, 1)
        t_scale, t_shift = torch.chunk(t_scale_shift, 2, dim=1)

        c_scale_shift = self.cond_proj(cond_feat)
        c_scale, c_shift = torch.chunk(c_scale_shift, 2, dim=1)

        h = self.norm2(h)
        h = (1 + t_scale + c_scale) * h + (t_shift + c_shift)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ------------------------------------------------------------------
# SR3 UNet (with generic padding to multiples of 8)
# ------------------------------------------------------------------

class SR3UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        cond_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
    ):
        super().__init__()

        # --- T embedding ---
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # --- Input convolutions ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.cond_conv_in = nn.Conv2d(cond_channels, base_channels, 3, padding=1)

        # Track channels for skip connections
        chs = [base_channels]
        in_ch = base_channels
        cond_in_ch = base_channels

        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()
        self.cond_projs = nn.ModuleList()

        # --- Down path ---
        for mult in channel_mults:
            out_ch = base_channels * mult

            self.downs.append(
                nn.ModuleList([
                    ResBlock(in_ch, out_ch, time_emb_dim, cond_ch=out_ch),
                    ResBlock(out_ch, out_ch, time_emb_dim, cond_ch=out_ch),
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),  # downsample
                ])
            )

            # Condition projection for this scale
            self.cond_projs.append(nn.Conv2d(cond_in_ch, out_ch, 1))
            self.cond_downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

            chs.append(out_ch)
            in_ch = out_ch
            cond_in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck1 = ResBlock(in_ch, in_ch, time_emb_dim, cond_ch=in_ch)
        self.bottleneck2 = ResBlock(in_ch, in_ch, time_emb_dim, cond_ch=in_ch)

        # --- Up path ---
        self.ups = nn.ModuleList()
        self.cond_ups = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult

            self.ups.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    ResBlock(out_ch + chs.pop(), out_ch, time_emb_dim, cond_ch=out_ch),
                    ResBlock(out_ch, out_ch, time_emb_dim, cond_ch=out_ch),
                ])
            )
            self.cond_ups.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1),
        )

    # ------------------------------------------------------------------
    # Forward pass with PAD-TO-MULTIPLE-OF-8, then crop back
    # ------------------------------------------------------------------
    def forward(self, x, t, cond):
        """
        x: (B,1,H,W) noisy HR image at timestep t
        cond: (B,1,H,W) LR image upsampled to HR grid
        """
        orig_H, orig_W = x.shape[2], x.shape[3]

        # --- 1) Compute padding so that H, W become multiples of 8 ---
        pad_h = (8 - orig_H % 8) % 8
        pad_w = (8 - orig_W % 8) % 8

        pad_top = pad_h // 2
        pad_bot  = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode="reflect")
            cond = F.pad(cond, (pad_left, pad_right, pad_top, pad_bot), mode="reflect")

        # ensure same size
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)

        # --- T embedding ---
        t_emb = self.time_embedding(t)

        # --- Condition embedding ---
        c = self.cond_conv_in(cond)

        hs = []
        h = self.input_conv(x)
        hs.append(h)

        # --- Down ---
        for (res1, res2, down), c_proj, c_down in zip(self.downs, self.cond_projs, self.cond_downs):
            c = c_proj(c)
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)

            h = res1(h, t_emb, c)
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)

            h = res2(h, t_emb, c)
            hs.append(h)

            h = down(h)
            c = c_down(c)

        # --- Bottleneck ---
        c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.bottleneck1(h, t_emb, c)

        c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.bottleneck2(h, t_emb, c)

        # --- Up ---
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

        # --- 2) Crop back to original HR size ---
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, pad_top:pad_top + orig_H, pad_left:pad_left + orig_W]

        return out


# ------------------------------------------------------------------
# SR3 Diffusion wrapper
# ------------------------------------------------------------------

class SR3SuperResolution(nn.Module):

    def __init__(self, unet: SR3UNet, timesteps=1000, upscale_factor=5):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.upscale_factor = upscale_factor

        # Schedule buffers
        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        alphas_cum_prev = torch.cat([torch.tensor([1.0], dtype=alphas.dtype), alphas_cum[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cum)
        self.register_buffer("alphas_cumprod_prev", alphas_cum_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cum))

        posterior_var = betas * (1 - alphas_cum_prev) / (1 - alphas_cum)
        posterior_var = torch.clamp(posterior_var, min=1e-20)

        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_var))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cum_prev) / (1 - alphas_cum))
        self.register_buffer("posterior_mean_coef2", (1 - alphas_cum_prev) * torch.sqrt(alphas) / (1 - alphas_cum))

    # -----------------------------------------------------
    # Forward diffusion q(x_t | x_0)
    # -----------------------------------------------------

    def q_sample(self, x0, t, noise=None):
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
        lr: (B,1,21,21)
        hr: (B,1,105,105)
        """
        # Upsample LR → HR grid
        cond = F.interpolate(lr, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False)

        # Ensure hr matches cond grid
        if hr.shape[2:] != cond.shape[2:]:
            hr_up = F.interpolate(hr, size=cond.shape[2:], mode="bilinear", align_corners=False)
        else:
            hr_up = hr

        # Forward diffusion
        x_t, noise = self.q_sample(hr_up, t)

        # Predict noise
        pred_noise = self.unet(x_t, t, cond)

        # Align shapes if needed
        if pred_noise.shape[2:] != noise.shape[2:]:
            pred_noise = F.interpolate(pred_noise, size=noise.shape[2:], mode="bilinear", align_corners=False)

        # Reconstruct x0
        B = x_t.shape[0]
        sqrt_a = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        x0_pred = (x_t - sqrt_om * pred_noise) / (sqrt_a + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        return pred_noise, noise, x0_pred

    # -----------------------------------------------------
    # Reverse step
    # -----------------------------------------------------

    @torch.no_grad()
    def p_mean_variance(self, lr, x_t, t):
        B = x_t.shape[0]

        cond = F.interpolate(lr, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False)
        if cond.shape[2:] != x_t.shape[2:]:
            cond = F.interpolate(cond, size=x_t.shape[2:], mode="bilinear", align_corners=False)

        eps_theta = self.unet(x_t, t, cond)
        if eps_theta.shape[2:] != x_t.shape[2:]:
            eps_theta = F.interpolate(eps_theta, size=x_t.shape[2:], mode="bilinear", align_corners=False)

        sqrt_a = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        x0_pred = (x_t - sqrt_om * eps_theta) / (sqrt_a + 1e-8)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        coef1 = self.posterior_mean_coef1[t].view(B, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(B, 1, 1, 1)
        mean = coef1 * x0_pred + coef2 * x_t

        log_var = self.posterior_log_variance_clipped[t].view(B, 1, 1, 1)
        return mean, log_var, x0_pred

    # -----------------------------------------------------
    # One sampling step
    # -----------------------------------------------------

    @torch.no_grad()
    def p_sample(self, lr, x_t, t):
        mean, log_var, x0_pred = self.p_mean_variance(lr, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise, x0_pred

    # -----------------------------------------------------
    # Full sampling loop
    # -----------------------------------------------------

    @torch.no_grad()
    def sample(self, lr, num_steps=None):
        B = lr.shape[0]
        device = lr.device

        if num_steps is None or num_steps >= self.timesteps:
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        # Calculate HR shape from LR dynamically
        H_hr = lr.shape[2] * self.upscale_factor
        W_hr = lr.shape[3] * self.upscale_factor

        x_t = torch.randn((B, 1, H_hr, W_hr), device=device)

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t, _ = self.p_sample(lr, x_t, t)

        return x_t
