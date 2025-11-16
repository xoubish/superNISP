# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from dataset import SuperResolutionDataset
from model import SuperResDiffusionUNet, Upsampler, SuperResolutionDiffusion
from losses import HybridLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- hyperparams ---
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    timesteps = 500
    t_recon_max = 400           # only use x0 recon loss for t < this
    inference_steps = 50
    upscale_factor = 5
    recon_weight = 0.05         # weight on x0 reconstruction loss
    grad_clip = 1.0

    # --- wandb init ---
    wandb.init(
        project="superNISP_diffusion",
        config={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "timesteps": timesteps,
            "t_recon_max": t_recon_max,
            "inference_steps": inference_steps,
            "upscale_factor": upscale_factor,
            "recon_weight": recon_weight,
        },
    )

    # --- data ---
    train_ds = SuperResolutionDataset(
        "../data/Nisp_train_cosmos.hdf5",
        "../data/Nircam_train_cosmos.hdf5",
        split="train",
        sample_fraction=0.2,
    )
    val_ds = SuperResolutionDataset(
        "../data/Nisp_train_cosmos.hdf5",
        "../data/Nircam_train_cosmos.hdf5",
        split="test",
        sample_fraction=0.1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # --- model ---
    unet = SuperResDiffusionUNet(
        in_channels=1,
        out_channels=1,
        hidden_dim=64,
        activation_fn=nn.ReLU,
    ).to(device)

    upsampler = Upsampler(
        in_channels=1,
        out_channels=1,
        upscale_factor=upscale_factor,
    ).to(device)

    model = SuperResolutionDiffusion(
        unet_model=unet,
        upsampler=upsampler,
        timesteps=timesteps,
        output_size=[125, 125],
        inference_timesteps=inference_steps,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # only MSE for now in HybridLoss (L1_weight=0)
    recon_loss_fn = HybridLoss(mse_weight=1.0, l1_weight=0.0)

    os.makedirs("checkpoints", exist_ok=True)

    # --- training loop ---
    for epoch in range(epochs):
        model.train()
        train_noise_loss = 0.0
        train_recon_loss = 0.0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            B = lr_batch.size(0)
            t = torch.randint(0, timesteps, (B,), device=device)

            # mask for which samples we apply x0 recon loss
            recon_mask = (t < t_recon_max)

            optimizer.zero_grad()

            # condition on upsampled LR
            upscaled = model.upsampler(lr_batch)
            if hr_batch.shape[2:] != upscaled.shape[2:]:
                hr_resized = F.interpolate(
                    hr_batch,
                    size=upscaled.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                hr_resized = hr_batch

            # forward diffusion: sample x_t and noise
            x_t, noise = model.q_sample(hr_resized, t)

            # predict noise
            pred_noise = model.diffusion(x_t, t, upscaled)
            if pred_noise.shape != noise.shape:
                pred_noise = F.interpolate(
                    pred_noise,
                    size=noise.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            # noise prediction loss
            noise_loss = F.mse_loss(pred_noise, noise)

            # reconstruct x0 from predicted noise
            alpha_bar_t = model.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            x0_pred = (x_t - sqrt_one_minus * pred_noise) / (sqrt_ab + 1e-8)

            # x0 recon loss only for t < t_recon_max
            if recon_mask.any():
                x0_pred_sub = x0_pred[recon_mask]
                hr_sub = hr_resized[recon_mask]

                # clamp x0_pred into sane range
                with torch.no_grad():
                    lo = hr_sub.min()
                    hi = hr_sub.max()
                x0_pred_sub = torch.clamp(x0_pred_sub, lo, hi)

                recon_loss = recon_loss_fn(x0_pred_sub, hr_sub)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            loss = noise_loss + recon_weight * recon_loss

            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_noise_loss += noise_loss.item()
            train_recon_loss += recon_loss.item()

        train_noise_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)

        # --- validation ---
        model.eval()
        val_noise_loss = 0.0
        val_recon_loss = 0.0

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                B = lr_batch.size(0)
                t = torch.randint(0, timesteps, (B,), device=device)
                recon_mask = (t < t_recon_max)

                upscaled = model.upsampler(lr_batch)
                if hr_batch.shape[2:] != upscaled.shape[2:]:
                    hr_resized = F.interpolate(
                        hr_batch,
                        size=upscaled.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    hr_resized = hr_batch

                x_t, noise = model.q_sample(hr_resized, t)
                pred_noise = model.diffusion(x_t, t, upscaled)
                if pred_noise.shape != noise.shape:
                    pred_noise = F.interpolate(
                        pred_noise,
                        size=noise.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )

                noise_loss = F.mse_loss(pred_noise, noise)

                alpha_bar_t = model.alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_ab = torch.sqrt(alpha_bar_t)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
                x0_pred = (x_t - sqrt_one_minus * pred_noise) / (sqrt_ab + 1e-8)

                if recon_mask.any():
                    x0_pred_sub = x0_pred[recon_mask]
                    hr_sub = hr_resized[recon_mask]

                    with torch.no_grad():
                        lo = hr_sub.min()
                        hi = hr_sub.max()
                    x0_pred_sub = torch.clamp(x0_pred_sub, lo, hi)

                    recon_loss = recon_loss_fn(x0_pred_sub, hr_sub)
                else:
                    recon_loss = torch.tensor(0.0, device=device)

                val_noise_loss += noise_loss.item()
                val_recon_loss += recon_loss.item()

        val_noise_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)

        # LR scheduler driven by total val loss
        total_val_loss = val_noise_loss + recon_weight * val_recon_loss
        scheduler.step(total_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| train_noise={train_noise_loss:.4f}, train_recon={train_recon_loss:.4f} "
            f"| val_noise={val_noise_loss:.4f}, val_recon={val_recon_loss:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_noise": train_noise_loss,
                "train_recon": train_recon_loss,
                "val_noise": val_noise_loss,
                "val_recon": val_recon_loss,
                "val_total": total_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # checkpoint
        ckpt_path = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
