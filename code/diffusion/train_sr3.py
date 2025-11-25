# train_sr3.py
#
# Training script for SR3SuperResolution using weighted epsilon-loss
# and (optionally) Perceptual Loss on the predicted x0.

import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import wandb
from torchvision.models import vgg16, VGG16_Weights

from dataset import SuperResolutionDataset
from model_sr3 import SR3UNet, SR3SuperResolution


# -----------------------------------------------------------
# PSNR helper (proxy metric)
# -----------------------------------------------------------

def psnr(pred, target):
    """
    Computes a PSNR-like metric using the dynamic range of the target
    in the current batch. This is more appropriate than assuming [-1,1]
    now that we use asinh normalization.
    """
    mse = F.mse_loss(pred, target)
    if mse <= 1e-8:
        return torch.tensor(99.0, device=pred.device)

    data_range = target.max() - target.min()
    if data_range <= 0:
        return torch.tensor(0.0, device=pred.device)

    return 10 * torch.log10((data_range ** 2) / mse)


# -----------------------------------------------------------
# Helper Functions: Gaussian Weight Map and Perceptual Loss
# -----------------------------------------------------------

def gaussian_weight_map_like(x, sigma=0.3):
    """
    Generate a Gaussian weight map centered in the middle of the image,
    on the SAME device and shape as x.

    x: tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    device = x.device

    y, xg = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    d = torch.sqrt(xg**2 + y**2)

    weights = torch.exp(-(d**2) / (2 * sigma**2))
    weights = weights / weights.mean()

    return weights.expand(B, C, H, W)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 for structure preservation on x0_pred.

    Inputs are in asinh-normalized space; we use VGG16 features
    as a relative structural comparator.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # Use features up to relu2_2 (index 9)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # VGG expects 3 channels; repeat grayscale input: (B,1,H,W) -> (B,3,H,W)
        pred_3 = pred.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)

        pred_features = self.feature_extractor(pred_3)
        target_features = self.feature_extractor(target_3)

        return F.mse_loss(pred_features, target_features)


# -----------------------------------------------------------
# Utility: timestep sampling & visualization scaling
# -----------------------------------------------------------

def sample_timesteps(B, timesteps, device):
    """
    Importance sampling over timesteps: slightly higher probability
    near earlier timesteps. You can tune this schedule.
    """
    probs = torch.linspace(1.0, 0.5, timesteps, device=device)
    probs = probs / probs.sum()
    t = torch.multinomial(probs, B, replacement=True)
    return t


def to_viz(x):
    """
    Map a tensor to [0, 1] per image for visualization only,
    using per-image min-max. Works for (C,H,W) or (H,W).
    """
    x = x.detach().clone()
    if x.dim() == 2:
        x2 = x.unsqueeze(0)  # (1,H,W)
    elif x.dim() == 3:
        x2 = x
    else:
        raise ValueError(f"Expected 2D or 3D tensor for viz, got {x.shape}")

    xmin = x2.min()
    xmax = x2.max()
    if xmax > xmin:
        x2 = (x2 - xmin) / (xmax - xmin)
    else:
        x2 = torch.zeros_like(x2)
    return x2  # (C,H,W)


# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=32)

    # --- Loss Arguments ---
    parser.add_argument("--gauss_sigma", type=float, default=0.4,
                        help="Sigma for Gaussian weight map for epsilon loss.")
    parser.add_argument("--lambda_perceptual", type=float, default=0.0,
                        help="Weight for the VGG-based Perceptual Loss on x0_pred.")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")

    # --- Data control ---
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of dataset to use (for debugging/overfitting).")

    # --- Misc / Logging ---
    parser.add_argument("--project", type=str, default="superNISP_sr3")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)  # not used yet
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--inference_steps", type=int, default=100,
                        help="Sampling steps for visualization during training.")
    parser.add_argument("--full_inference_steps", type=int, default=1000,
                        help="Sampling steps for full inference (if different from timesteps).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print("Device:", device)

    # --- W&B ---
    wandb.init(
        project=args.project,
        entity=args.entity,
        config=vars(args),
    )
    config = wandb.config

    # --- Data (41x41 -> 205x205 files, cropped to 21x21 -> 105x105) ---
    train_ds = SuperResolutionDataset(
        lr_path="../../data/euclid_NIR_cosmos_41px_Y_20251124.npy",
        hr_path="../../data/jwst_cosmos_205px_F115W_20251124.npy",
        split="train",
        lr_crop_size=21,
        hr_crop_size=105,
        sample_fraction=config.sample_fraction,
    )

    val_ds = SuperResolutionDataset(
        lr_path="../../data/euclid_NIR_cosmos_41px_Y_20251124.npy",
        hr_path="../../data/jwst_cosmos_205px_F115W_20251124.npy",
        split="test",
        lr_crop_size=21,
        hr_crop_size=105,
        sample_fraction=config.sample_fraction,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- Model & Loss ---
    unet = SR3UNet(
        in_channels=1,
        cond_channels=1,
        base_channels=config.hidden_dim,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
    ).to(device)

    model = SR3SuperResolution(
        unet=unet,
        timesteps=config.timesteps,
        upscale_factor=5,
    ).to(device)

    perceptual_loss_fn = PerceptualLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    os.makedirs("checkpoints_sr3", exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    # --- Training ---
    for epoch in range(config.epochs):
        model.train()
        train_eps_loss_sum = 0.0
        train_perceptual_loss_sum = 0.0

        start = time.time()
        optimizer.zero_grad()

        for step, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            B = lr_batch.size(0)
            t = sample_timesteps(B, config.timesteps, device)

            # model returns pred_eps, true_eps, x0_pred
            pred_eps, true_eps, x0_pred = model(lr_batch, hr_batch, t)

            # --- 1. Weighted Epsilon Loss ---
            weights = gaussian_weight_map_like(pred_eps, sigma=config.gauss_sigma)
            eps_loss = (F.mse_loss(pred_eps, true_eps, reduction='none') * weights).mean()

            # --- 2. Perceptual Loss on x0 prediction (optional) ---
            if config.lambda_perceptual > 0.0:
                perceptual_loss = perceptual_loss_fn(x0_pred, hr_batch)
            else:
                perceptual_loss = torch.tensor(0.0, device=device)

            # --- 3. Total Loss ---
            loss = eps_loss + config.lambda_perceptual * perceptual_loss

            # Gradient accumulation
            loss = loss / config.accumulation_steps
            loss.backward()

            # Step optimizer every N steps or at end of epoch
            if ((step + 1) % config.accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_eps_loss_sum += eps_loss.item()
            train_perceptual_loss_sum += perceptual_loss.item()

        train_eps_loss_avg = train_eps_loss_sum / len(train_loader)
        train_perceptual_loss_avg = train_perceptual_loss_sum / max(len(train_loader), 1)
        train_total_loss_avg = train_eps_loss_avg + config.lambda_perceptual * train_perceptual_loss_avg

        # --- Validation: Hybrid Loss + PSNR at sampled t ---
        model.eval()
        val_eps_loss_sum = 0.0
        val_perceptual_loss_sum = 0.0
        val_psnr_sum = 0.0
        logged_x0 = False  # log x0_pred only once per epoch

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                B = lr_batch.size(0)
                t = torch.randint(
                    low=0,
                    high=config.timesteps,
                    size=(B,),
                    device=device,
                    dtype=torch.long,
                )

                pred_eps, true_eps, x0_pred = model(lr_batch, hr_batch, t)

                # 1. Weighted Epsilon Loss
                weights = gaussian_weight_map_like(pred_eps, sigma=config.gauss_sigma)
                eps_loss = (F.mse_loss(pred_eps, true_eps, reduction='none') * weights).mean()
                val_eps_loss_sum += eps_loss.item()

                # 2. Perceptual Loss
                if config.lambda_perceptual > 0.0:
                    perceptual_loss = perceptual_loss_fn(x0_pred, hr_batch)
                else:
                    perceptual_loss = torch.tensor(0.0, device=device)
                val_perceptual_loss_sum += perceptual_loss.item()

                # 3. PSNR on x0_pred
                val_psnr_sum += psnr(x0_pred, hr_batch).item()

                # Log one x0_pred vs HR example per epoch
                if not logged_x0:
                    x0_vis = to_viz(x0_pred[0].cpu())
                    hr_vis = to_viz(hr_batch[0].cpu())
                    wandb.log({
                        "viz_x0_pred": wandb.Image(x0_vis, caption=f"x0_pred epoch {epoch+1}"),
                        "viz_x0_hr": wandb.Image(hr_vis, caption="HR for x0_pred"),
                    })
                    logged_x0 = True

        val_eps_loss_avg = val_eps_loss_sum / len(val_loader)
        val_perceptual_loss_avg = val_perceptual_loss_sum / max(len(val_loader), 1)
        val_total_loss_avg = val_eps_loss_avg + config.lambda_perceptual * val_perceptual_loss_avg
        val_psnr_avg = val_psnr_sum / len(val_loader)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{config.epochs} "
            f"| train_total_loss={train_total_loss_avg:.4f} "
            f"| val_total_loss={val_total_loss_avg:.4f} "
            f"| val_psnr(x0_pred)={val_psnr_avg:.2f} "
            f"| time={elapsed:.1f}s"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/eps_loss": train_eps_loss_avg,
                "train/perceptual_loss": train_perceptual_loss_avg,
                "train/total_loss": train_total_loss_avg,
                "val/eps_loss": val_eps_loss_avg,
                "val/perceptual_loss": val_perceptual_loss_avg,
                "val/total_loss": val_total_loss_avg,
                "val/psnr_x0": val_psnr_avg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step(val_total_loss_avg)

        # --- Visualization every 10 epochs (full sampling) ---
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                idx = random.randint(0, len(val_ds) - 1)
                lr_img, hr_img = val_ds[idx]  # asinh-normalized tensors
                lr_img = lr_img.unsqueeze(0).to(device)  # (1,1,21,21)

                sr_sample = model.sample(lr_img, num_steps=config.inference_steps)[0].cpu()  # (1,105,105)

                sr_sample_viz = to_viz(sr_sample)      # (1,H,W)
                hr_img_viz = to_viz(hr_img)            # (1,H,W)

                hr_up_interp = F.interpolate(
                    lr_img[0].cpu().unsqueeze(0),       # (1,1,21,21)
                    size=hr_img.shape[1:],              # (H,W)
                    mode="bilinear",
                    align_corners=False,
                )[0]
                hr_up_interp_viz = to_viz(hr_up_interp)

                wandb.log(
                    {
                        "viz_super_res": wandb.Image(sr_sample_viz, caption=f"SR epoch {epoch+1}"),
                        "viz_low_res_interp": wandb.Image(hr_up_interp_viz, caption="LR (Interpolated)"),
                        "viz_high_res": wandb.Image(hr_img_viz, caption="HR (Ground Truth)"),
                    }
                )

        # --- Early stopping & checkpoints ---
        if val_total_loss_avg < best_val_loss - config.early_stop_min_delta:
            best_val_loss = val_total_loss_avg
            patience_counter = 0
            best_path = os.path.join("checkpoints_sr3", "best_sr3.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ New best model saved: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                ckpt_path = os.path.join("checkpoints_sr3", f"sr3_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                break

        # Per-epoch checkpoint (optional)
        ckpt_path = os.path.join("checkpoints_sr3", f"sr3_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
