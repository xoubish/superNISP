# train.py

import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import SuperResolutionDataset
from model import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler
from losses import HybridLoss

import torchmetrics
from skimage.metrics import structural_similarity as ssim

# === W&B Config Defaults ===================================================

def setup_config_defaults():
    default_config = {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 1e-3,
        "activation_function": "ReLU",
        "optimizer": "AdamW",
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 64,
        "timesteps": 500,
        "mse_weight": 1.0,
        "l1_weight": 0.1,
        "perceptual_weight": 0.01,
        "shape_weight": 0.1,
        "weight_decay": 0.01,
    }
    for key, value in default_config.items():
        if not hasattr(wandb.config, key):
            setattr(wandb.config, key, value)


# === Loss & Metrics Helpers ===============================================

def get_loss_function(config):
    return HybridLoss(
        mse_weight=config.mse_weight,
        l1_weight=config.l1_weight,
        perceptual_weight=config.perceptual_weight,
        shape_weight=config.shape_weight,
    )

def compute_ellipticity_from_moments(img):
    """
    img: (B,1,H,W)
    returns: (B,2) [e1, e2]
    """
    B, _, H, W = img.shape
    device = img.device

    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    x = x[None, None, :, :].expand(B, 1, H, W)
    y = y[None, None, :, :].expand(B, 1, H, W)

    flux = img.sum(dim=[2, 3], keepdim=True)
    x_bar = (img * x).sum(dim=[2, 3], keepdim=True) / (flux + 1e-8)
    y_bar = (img * y).sum(dim=[2, 3], keepdim=True) / (flux + 1e-8)

    dx = x - x_bar
    dy = y - y_bar

    Mxx = (img * dx**2).sum(dim=[2, 3]) / (flux.squeeze() + 1e-8)
    Myy = (img * dy**2).sum(dim=[2, 3]) / (flux.squeeze() + 1e-8)
    Mxy = (img * dx * dy).sum(dim=[2, 3]) / (flux.squeeze() + 1e-8)

    e1 = (Mxx - Myy) / (Mxx + Myy + 1e-8)
    e2 = 2.0 * Mxy / (Mxx + Myy + 1e-8)

    return torch.stack([e1, e2], dim=1)

def compute_psnr(pred, target):
    """
    pred, target: (B,1,H,W)
    """
    psnr = torchmetrics.functional.image.peak_signal_noise_ratio(pred, target)
    return psnr.item()

def compute_ssim(pred, target):
    """
    pred, target: (B,1,H,W) on CPU
    """
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(pred_np, target_np, data_range=target_np.max() - target_np.min())

def compute_average_metrics(model, dataloader, device, timesteps):
    """
    Evaluate PSNR/SSIM over the dataloader using t=0 and no extra noise.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for lr_test, hr_test in dataloader:
            lr_test = lr_test.to(device)
            hr_test = hr_test.to(device)

            B = lr_test.size(0)
            t = torch.zeros(B, dtype=torch.long, device=device)

            # deterministic evaluation: no noise, t=0
            pred_hr = model(lr_test, t, add_noise=False).cpu()

            psnr_value = compute_psnr(pred_hr, hr_test.cpu())
            ssim_value = compute_ssim(pred_hr, hr_test.cpu())

            total_psnr += psnr_value
            total_ssim += ssim_value
            num_batches += 1

    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0.0
    avg_ssim = total_ssim / num_batches if num_batches > 0 else 0.0
    return avg_psnr, avg_ssim


# === Main Training Script ==================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"

    print("Device:", device)

    # --- W&B init ---
    wandb.init(project="super-resolution-diffusion", config={"entity": "your_wandb_entity_name"})
    setup_config_defaults()
    config = wandb.config

    # --- data ---
    train_ds = SuperResolutionDataset(
        "../../data/Nisp_train_cosmos.hdf5",
        "../../data/Nircam_train_cosmos.hdf5",
        split="train",
        sample_fraction=0.2,
    )
    test_ds = SuperResolutionDataset(
        "../../data/Nisp_train_cosmos.hdf5",
        "../../data/Nircam_train_cosmos.hdf5",
        split="test",
        sample_fraction=0.1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- model ---
    activation_fn = getattr(nn, config.activation_function)

    unet = SuperResDiffusionUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_dim=config.hidden_dim,
        activation_fn=activation_fn,
    ).to(device)

    upsampler = Upsampler(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
    ).to(device)

    model = SuperResolutionDiffusion(
        unet_model=unet,
        upsampler=upsampler,
        timesteps=config.timesteps,
    ).to(device)

    # --- loss, optimizer, scheduler ---
    criterion = get_loss_function(config).to(device)
    optimizer_cls = getattr(optim, config.optimizer)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    scaler = GradScaler(device_type)

    best_loss = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    # --- training loop ---
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            B = lr_batch.shape[0]
            t = torch.randint(0, config.timesteps, (B,), device=device)

            optimizer.zero_grad()

            with autocast(device_type):
                # noise-conditioned super-resolution
                output = model(lr_batch, t, add_noise=True)
                loss = criterion(output, hr_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_time": time.time() - start_time,
            }
        )

        print(f"Epoch {epoch+1}/{config.epochs} | train_loss={avg_loss:.4f}")

        # --- visualization + metrics every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # pick a random test sample
                idx = random.randint(0, len(test_ds) - 1)
                lr_img, hr_img = test_ds[idx]
                lr_batch = lr_img.unsqueeze(0).to(device)

                # deterministic SR (no noise, t=0)
                t_test = torch.zeros((1,), dtype=torch.long, device=device)
                sr_img = model(lr_batch, t_test, add_noise=False).cpu().squeeze(0)

                sr_img_tensor = sr_img.unsqueeze(0)
                hr_img_tensor = hr_img.unsqueeze(0)

                e_pred = compute_ellipticity_from_moments(sr_img_tensor)
                e_true = compute_ellipticity_from_moments(hr_img_tensor)
                shape_error = torch.abs(e_pred - e_true).squeeze()

                val_psnr, val_ssim = compute_average_metrics(
                    model, test_loader, device, config.timesteps
                )

                wandb.log(
                    {
                        "low_res": wandb.Image(lr_img, caption=f"Low-Res {idx}"),
                        "super_res": wandb.Image(sr_img, caption=f"Super-Res {idx}"),
                        "high_res": wandb.Image(hr_img, caption=f"High-Res {idx}"),
                        "shape_error_e1": shape_error[0].item(),
                        "shape_error_e2": shape_error[1].item(),
                        "psnr": val_psnr,
                        "ssim": val_ssim,
                    }
                )

        # --- save best model ---
        if avg_loss < best_loss - 1e-3:
            best_loss = avg_loss
            best_model_path = os.path.join(wandb.run.dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated (Loss: {best_loss:.6f})")

        # optional per-epoch checkpoint
        ckpt_path = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)

    # --- final save ---
    final_model_path = os.path.join(wandb.run.dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved.")
    wandb.finish()


if __name__ == "__main__":
    main()
