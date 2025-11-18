# model_sr3.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# SR3 / Improved-DDPM cosine beta schedule
# ------------------------------------------------------------------

def make_sr3_cosine_betas(num_timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """
    SR3-style cosine schedule from Nichol & Dhariwal (Improved DDPM):
      alpha_bar(t) = cos^2( (t/T + s)/(1+s) * pi/2 )

    We derive betas such that:
      alpha_bar_{t+1} = alpha_bar_t * (1 - beta_t)

    Returns:
      betas: (T,) tensor of diffusion betas in [0, max_beta].
    """
    # t goes from 0 to T
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)  # T+1 points
    # Normalized time in [0,1]
    t = steps / num_timesteps

    # alphā(t) continuous function
    alpha_bar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    # Normalize to start exactly at 1
    alpha_bar = alpha_bar / alpha_bar[0]

    # alphā_t and alphā_{t+1}
    alpha_bar_t = alpha_bar[:-1]      # length T
    alpha_bar_next = alpha_bar[1:]    # length T

    betas = 1.0 - (alpha_bar_next / alpha_bar_t)
    betas = torch.clamp(betas, min=1e-8, max=max_beta)
    return betas.float()

# Aliasing the function name for the SR3SuperResolution class
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
        t: (B,) integer or float timesteps
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device).float()
            / (half_dim - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


# ------------------------------------------------------------------
# Simple residual block with time + cond injection (FiLM-like)
# ------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_ch, groups=8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # project time embedding to scale/shift
        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)

        # project cond feature (same spatial size) to scale/shift
        self.cond_proj = nn.Conv2d(cond_ch, out_ch * 2, 1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t_emb, cond_feat):
        """
        x: (B, in_ch, H, W)
        t_emb: (B, time_emb_dim)
        cond_feat: (B, cond_ch, H, W)
        """
        B, _, H, W = x.shape

        # --- first conv ---
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # --- add time + cond via FiLM ---
        # time embedding -> (B, 2*out_ch) -> (B, 2*out_ch, 1, 1)
        t_scale_shift = self.time_proj(t_emb).view(B, 2 * self.out_ch, 1, 1)
        t_scale, t_shift = torch.chunk(t_scale_shift, 2, dim=1)

        # cond feature -> (B, 2*out_ch, H, W)
        c_scale_shift = self.cond_proj(cond_feat)
        c_scale, c_shift = torch.chunk(c_scale_shift, 2, dim=1)

        h = self.norm2(h)
        # combine time + cond modulation
        scale = 1.0 + t_scale + c_scale
        shift = t_shift + c_shift
        h = scale * h + shift
        h = self.act(h)

        h = self.conv2(h)

        return h + self.skip(x)


# ------------------------------------------------------------------
# UNet backbone (fairly SR3-like, but compact)
# ------------------------------------------------------------------

class SR3UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,      # noisy HR channel(s)
        cond_channels=1,    # upsampled LR channel(s)
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # initial conv on x_t
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # a small conv tower on the condition; we will downsample it
        self.cond_conv_in = nn.Conv2d(cond_channels, base_channels, 3, padding=1)

        chs = [base_channels]
        in_ch = base_channels
        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()
        self.cond_projs = nn.ModuleList()  # Channel projection layers

        # --- Down path ---
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
            # Project condition channels to match out_ch
            # Note: This projection is necessary because in_ch != out_ch across blocks
            self.cond_projs.append(nn.Conv2d(in_ch, out_ch, 1))
            self.cond_downs.append(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            )
            chs.append(out_ch)
            in_ch = out_ch

        # bottleneck
        self.bottleneck1 = ResBlock(in_ch, in_ch, time_emb_dim, cond_ch=in_ch)
        self.bottleneck2 = ResBlock(in_ch, in_ch, time_emb_dim, cond_ch=in_ch)

        # --- Up path ---
        self.ups = nn.ModuleList()
        # *** REMOVED self.cond_ups as it was redundant/unnecessary ***
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
            in_ch = out_ch

        # final conv to noise prediction
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t, cond):
        """
        x: (B, 1, H, W)  noisy HR at timestep t
        t: (B,) timesteps
        cond: (B, 1, H, W) upsampled LR (fixed, noise-free)
        """
        # Pad to 128x128 if input is 125x125 (for cleaner downsampling)
        original_size = x.shape[2:]
        if x.shape[2] == 125 and x.shape[3] == 125:
            x = F.pad(x, (1, 2, 1, 2), mode='reflect')  # Pad to 128x128
            cond = F.pad(cond, (1, 2, 1, 2), mode='reflect')
        
        # unify spatial size (if slightly off)
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)

        t_emb = self.time_embedding(t)  # (B, time_emb_dim)

        # initial condition features
        c = self.cond_conv_in(cond)

        hs = []
        h = self.input_conv(x)
        # First skip connection is after input_conv
        hs.append(h)

        # down path
        for (res1, res2, down), c_proj, c_down in zip(self.downs, self.cond_projs, self.cond_downs):
            # c_proj matches c's channel depth to h's channel depth (in_ch -> out_ch)
            c = c_proj(c)
            # keep cond features aligned spatially with h
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
            h = res1(h, t_emb, c)
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
            h = res2(h, t_emb, c)
            hs.append(h)

            # downsample both h and c
            h = down(h)
            c = c_down(c) # c is now ready for the next level or bottleneck

        # bottleneck
        c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.bottleneck1(h, t_emb, c)
        c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.bottleneck2(h, t_emb, c)

        # up path
        for up, res1, res2 in self.ups:
            h = up(h)
            
            # pop skip connection and match size
            skip = hs.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)

            # *** UPDATED: Upsample condition feature `c` via interpolation ***
            # c retains its channel depth, just gets upscaled spatially
            c = F.interpolate(c, size=h.shape[2:], mode="bilinear", align_corners=False)

            h = res1(h, t_emb, c)
            h = res2(h, t_emb, c)

        output = self.out_conv(h)
        
        # Crop back to original size if we padded
        if original_size == (125, 125) and output.shape[2:] == (128, 128):
            output = output[:, :, 1:126, 1:126]  # Crop back to 125x125
        
        return output


# ------------------------------------------------------------------
# Diffusion wrapper (DDPM) for SR3-style conditioning
# ------------------------------------------------------------------

class SR3SuperResolution(nn.Module):
    """
    SR3-style conditional diffusion model for 5x super-resolution.

    Training:
      - sample t ~ Uniform{0..T-1}
      - x0 = HR target
      - x_t = sqrt(ā_t) * x0 + sqrt(1-ā_t) * ε
      - predict ε_θ(x_t, t, cond), cond = upsampled LR
      - loss = MSE(ε, ε_θ).

    Sampling:
      - start from x_T ~ N(0, I)
      - run DDPM reverse process, conditioned on upsampled LR.
    """

    def __init__(
        self,
        unet: SR3UNet,
        timesteps: int = 1000,
        upscale_factor: int = 5,
    ):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.upscale_factor = upscale_factor

        # --- SR3 / Improved-DDPM cosine schedule ---
        betas = cosine_beta_schedule(timesteps)  # length T
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "sqrt_recipm1_alphas",
            torch.sqrt(1.0 / alphas - 1.0),
        )

        # posterior q(x_{t-1} | x_t, x0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    # ---------------- forward diffusion q(x_t | x_0) -----------------

    def q_sample(self, x0, t, noise=None):
        """
        x0: (B,1,H,W)
        t:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_a_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a_bar_t * x0 + sqrt_one_minus * noise, noise

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def forward(self, lr, hr, t):
        """
        lr: (B,1,25,25) low-res input
        hr: (B,1,125,125) high-res target
        t:  (B,) timesteps
        returns:
          pred_noise: model ε_θ(x_t, t, cond)
          true_noise: ε used to generate x_t
        """
        # upsample LR once, treat as fixed condition
        cond = F.interpolate(
            lr,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=False,
        )

        # ensure HR matches cond size
        if hr.shape[2:] != cond.shape[2:]:
            hr = F.interpolate(
                hr,
                size=cond.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        x_t, noise = self.q_sample(hr, t)
        pred_noise = self.unet(x_t, t, cond)

        # small spatial mismatches → interpolate
        if pred_noise.shape[2:] != noise.shape[2:]:
            pred_noise = F.interpolate(
                pred_noise,
                size=noise.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return pred_noise, noise, x0_pred 

    # ------------------------------------------------------------------
    # DDPM reverse mean / variance (SR3 style)
    # ------------------------------------------------------------------

    def p_mean_variance(self, lr, x_t, t):
        """
        Compute posterior mean & log variance for q(x_{t-1} | x_t, x0)
        using SR3 / DDPM epsilon-prediction parametrization.
        """
        B = x_t.shape[0]

        # conditioning: upsample LR → HR grid
        cond = F.interpolate(
            lr,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=False,
        )
        if cond.shape[2:] != x_t.shape[2:]:
            cond = F.interpolate(cond, size=x_t.shape[2:], mode="bilinear", align_corners=False)

        # predict ε_θ(x_t, t, cond)
        eps_theta = self.unet(x_t, t, cond)
        if eps_theta.shape[2:] != x_t.shape[2:]:
            eps_theta = F.interpolate(
                eps_theta,
                size=x_t.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        sqrt_a_bar_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_a_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # x0 = (x_t - sqrt(1 - ā_t) * eps) / sqrt(ā_t)
        x0_pred = (x_t - sqrt_one_minus_a_bar_t * eps_theta) / (sqrt_a_bar_t + 1e-8)
        # SR3 clips x0 to data range (assuming inputs normalized to [-1,1])
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        coef1 = self.posterior_mean_coef1[t].view(B, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(B, 1, 1, 1)
        posterior_mean = coef1 * x0_pred + coef2 * x_t

        posterior_log_variance = self.posterior_log_variance_clipped[t].view(B, 1, 1, 1)

        return posterior_mean, posterior_log_variance, x0_pred

    # ------------------------------------------------------------------
    # One reverse step p(x_{t-1} | x_t, lr)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, lr, x_t, t):
        """
        lr: (B,1,25,25)
        x_t: (B,1,H,W) at timestep t
        t:  (B,) integer timesteps
        """
        B = x_t.shape[0]

        mean, log_var, x0_pred = self.p_mean_variance(lr, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(B, 1, 1, 1)  # no noise when t = 0

        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev, x0_pred

    # ------------------------------------------------------------------
    # Full SR3 sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, lr, num_steps=None):
        """
        lr: (B,1,25,25)
        returns: (B,1,HR,HR) sampled super-res images

        NOTE: For SR3-style faithful sampling, it's best to use the
        full chain (num_steps=None). We'll still allow subsampling,
        but that's more like a DDIM shortcut.
        """
        B = lr.shape[0]
        device = lr.device

        # full DDPM chain by default
        if num_steps is None or num_steps >= self.timesteps:
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            # simple sub-sampling of timesteps (DDIM-style index set)
            timesteps = torch.linspace(
                self.timesteps - 1, 0, num_steps, device=device
            ).long()

        # initial image: pure Gaussian noise on HR grid
        cond_shape = (
            B,
            1,
            lr.shape[2] * self.upscale_factor,
            lr.shape[3] * self.upscale_factor,
        )
        x_t = torch.randn(cond_shape, device=device)

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t, _ = self.p_sample(lr, x_t, t)

        # at t=0, x_t should already be x0
        return x_t