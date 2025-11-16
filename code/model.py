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
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


# ------------------------------------------------------------
# UNet backbone (same architecture as you had)
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
        t_emb = self.unet.time_embed(t.float())
        return self.unet(x, condition, t_emb)


# ------------------------------------------------------------
# Noise schedule (same as your original cosine_schedule)
# ------------------------------------------------------------

def cosine_schedule(t, total_timesteps=500):
    """Cosine noise schedule."""
    return torch.cos((t.float() / total_timesteps) * (0.5 * torch.pi))


# ------------------------------------------------------------
# Super-resolution diffusion model
# ------------------------------------------------------------

class SuperResolutionDiffusion(nn.Module):
    """
    Training:
      - model(lr, t, training=True, hr_target=hr) → predicted_noise, true_noise

    Inference:
      - model(lr, t, training=False) → super-res image
      - or model.sample(lr, num_steps)
    """

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
    
        # We treat cosine_schedule over all timesteps as alphā_t (cumulative product of alphas)
        t_all = torch.arange(self.timesteps, dtype=torch.long)
        alpha_bar = cosine_schedule(t_all, self.timesteps)  # (T,)

        # avoid zeros to keep things numerically safe
        alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=0.99999)

        # derive per-step alphas and betas from alphā_t
        # alpha_t = alphā_t / alphā_{t-1}
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


    def q_sample(self, x0, t, noise=None):
        """
        Diffusion forward process: sample x_t given clean x0 and timestep t,
        using the same cosine schedule as the rest of the model.

        x0:   (B, C, H, W) clean target (HR)
        t:    (B,) integer timesteps
        noise: optional; if None, Gaussian noise is sampled
        returns: (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        alpha_t = cosine_schedule(t, self.timesteps)  # (B,)
        if alpha_t.dim() == 1:
            alpha_t = alpha_t.view(-1, 1, 1, 1)

        x_t = alpha_t * x0 + (1.0 - alpha_t) * noise
        return x_t, noise
    # -------- TRAINING FORWARD --------
    def forward(self, x, t, training=True, hr_target=None):
        """
        x: low-res input (B,1,25,25)
        t: (B,) timesteps
        training=True: returns (predicted_noise, true_noise)
        training=False: runs sampling and returns SR image
        """
        upscaled = self.upsampler(x)  # (B,1,HR,HR)

        if training:
            if hr_target is None:
                raise ValueError("hr_target required for training")

            # resize HR to match upscaled
            if hr_target.shape[2:] != upscaled.shape[2:]:
                hr_target = F.interpolate(
                    hr_target,
                    size=upscaled.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            # add noise to HR target
            noise = torch.randn_like(hr_target, device=hr_target.device)
            alpha_t = cosine_schedule(t, self.timesteps)
            if alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1, 1)

            noisy_image = alpha_t * hr_target + (1.0 - alpha_t) * noise

            # predict noise conditioned on upscaled LR
            predicted_noise = self.diffusion(noisy_image, t, upscaled)

            # align shapes if UNet output is smaller
            if predicted_noise.shape[2:] != noise.shape[2:]:
                predicted_noise = F.interpolate(
                    predicted_noise,
                    size=noise.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            return predicted_noise, noise

        else:
            # use proper iterative sampler
            return self.sample(x, num_steps=self.inference_timesteps)

    # -------- SAMPLING LOOP --------
    @torch.no_grad()
    def sample(self, lr, num_steps=None):
        """
        Generate high-res sample given low-res input.

        lr: (B, 1, 25, 25)
        returns: (B, 1, HR, HR) (HR ~ 125x125)
        """
        if num_steps is None:
            num_steps = self.inference_timesteps

        device = lr.device
        upscaled = self.upsampler(lr)       # conditioning image (B,1,HR,HR)
        sample = torch.randn_like(upscaled) # start from noise

        # map inference steps to training timesteps
        step_size = max(self.timesteps // num_steps, 1)

        for step_idx in range(num_steps - 1, -1, -1):
            t_val = step_idx * step_size
            t_batch = torch.full(
                (lr.shape[0],),
                t_val,
                device=device,
                dtype=torch.long,
            )

            # predict noise
            predicted_noise = self.diffusion(sample, t_batch, upscaled)

            # make sure predicted_noise has same spatial size
            if predicted_noise.shape[2:] != sample.shape[2:]:
                predicted_noise = F.interpolate(
                    predicted_noise,
                    size=sample.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            # alpha_t and (optionally) alpha_{t-1}
            alpha_t = cosine_schedule(t_batch, self.timesteps)
            if alpha_t.dim() == 1:
                alpha_t = alpha_t.view(-1, 1, 1, 1)

            if step_idx > 0:
                t_prev_val = (step_idx - 1) * step_size
                t_prev_batch = torch.full(
                    (lr.shape[0],),
                    t_prev_val,
                    device=device,
                    dtype=torch.long,
                )
                alpha_t_prev = cosine_schedule(t_prev_batch, self.timesteps)
                if alpha_t_prev.dim() == 1:
                    alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # predict x0 under your alpha/noise parameterization
            pred_x0 = (sample - (1.0 - alpha_t) * predicted_noise) / (alpha_t + 1e-8)

            # DDPM-style update
            if step_idx > 0:
                noise_term = torch.randn_like(sample) if step_idx > 1 else torch.zeros_like(sample)
                sample = alpha_t_prev * pred_x0 + (1.0 - alpha_t_prev) * noise_term
            else:
                sample = pred_x0

        # final resize if requested
        if self.output_size is not None:
            sample = F.interpolate(sample, size=self.output_size, mode="bilinear", align_corners=True)

        return sample
