# train_sr3.py
#
# Training script for SR3SuperResolution using weighted epsilon-loss
# and Perceptual Loss on the predicted x0.

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

# Assuming dataset.py is available
from dataset import SuperResolutionDataset
# Import the main model components
from model_sr3 import SR3UNet, SR3SuperResolution


# -----------------------------------------------------------
# PSNR helper (proxy metric)
# -----------------------------------------------------------

def psnr(pred, target):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR)-like metric.

    Uses the dynamic range of the target in the current batch rather than
    assuming [-1, 1], which is more appropriate now that we use asinh
    normalization instead of strict min-max scaling.
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

def gaussian_weight_map(shape, sigma=0.3):
    """
    Generates a Gaussian weight map centered in the middle of the image.
    Used to weight the epsilon loss, prioritizing the center galaxy.

    shape: (B, C, H, W)
    """
    B, C, H, W = shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create normalized coordinates
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    d = torch.sqrt(x**2 + y**2)

    # Gaussian weights: centered and decays outward
    weights = torch.exp(- (d**2) / (2 * sigma**2))

    # Normalize weights so the loss magnitude remains somewhat stable
    weights = weights / weights.mean()

    return weights.expand(B, C, H, W)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 for structure preservation on x0_pred.

    Inputs are in the asinh-normalized space; we use VGG16 features
    as a relative structural comparator (off ImageNet manifold, but OK
    for "does pred look like target?" in feature space).
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # Use features up to relu2_2 (index 9)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # VGG expects 3 channels; we repeat grayscale input: (B, 1, H, W) -> (B, 3, H, W)
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
    for earlier timesteps. You can tune this schedule.
    """
    probs = torch.linspace(1.0, 0.5, timesteps, device=device)
    probs = probs / probs.sum()
    t = torch.multinomial(probs, B, replacement=True)
    return t


def to_viz(x):
    """
    Map a tensor to [0, 1] per image for visualization only,
    using per-image min-max. Works for (C, H, W) or (H, W).
    """
    x = x.detach().clone()
    if x.dim() == 2:
        x2 = x.unsqueeze(0)  # (1, H, W)
    elif x.dim() == 3:
        x2 = x
    else:
        raise ValueError(f"Expected 2D or 3D tensor for viz, got shape {x.shape}")

    xmin = x2.min()
    xmax = x2.max()
    if xmax > xmin:
        x2 = (x2 - xmin) / (xmax - xmin)
    else:
        x2 = torch.zeros_like(x2)
    return x2  # (C, H, W)


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
    parser.add_argument("--lambda_perceptual", type=float, default=1e-3,
                        help="Weight for the VGG-based Perceptual Loss on x0_pred.")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")

    # --- Misc / Logging ---
    parser.add_argument("--project", type=str, default="superNISP_sr3")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)  # not used yet
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--inference_steps", type=int, default=100,
                        help="Number of sampling steps for visualization during training")
    parser.add_argument("--full_inference_steps", type=int, default=1000,
                        help="Number of sampling steps for full inference (if different from timesteps)")

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

    # --- Data (using 41x41 -> 205x205 files, cropped to 21x21 -> 105x105) ---
    train_ds = SuperResolutionDataset(
        lr_path="../../data/euclid_NIR_cosmos_41px_Y_20251124.npy",
        hr_path="../../data/jwst_cosmos_205px_F115W_20251124.npy",
        split="train",
        lr_crop_size=21,
        hr_crop_size=105,
    )

    val_ds = SuperResolutionDataset(
        lr_path="../../data/euclid_NIR_cosmos_41px_Y_20251124.npy",
        hr_path="../../data/jwst_cosmos_205px_F115W_20251124.npy",
        split="test",
        lr_crop_size=21,
        hr_crop_size=105,
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
            weights = gaussian_weight_map(pred_eps.shape, sigma=config.gauss_sigma).to(device)
            eps_loss = (F.mse_loss(pred_eps, true_eps, reduction='none') * weights).mean()

            # --- 2. Perceptual Loss on x0 prediction ---
            perceptual_loss = perceptual_loss_fn(x0_pred, hr_batch)

            # --- 3. Total Loss ---
            loss = eps_loss + config.lambda_perceptual * perceptual_loss

            # Gradient accumulation
            loss = loss / config.accumulation_steps
            loss.backward()

            # Step optimizer every N steps *or* at end of epoch
            if ((step + 1) % config.accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_eps_loss_sum += eps_loss.item()
            train_perceptual_loss_sum += perceptual_loss.item()

        train_eps_loss_avg = train_eps_loss_sum / len(train_loader)
        train_perceptual_loss_avg = train_perceptual_loss_sum / len(train_loader)
        train_total_loss_avg = train_eps_loss_avg + config.lambda_perceptual * train_perceptual_loss_avg

        # --- Validation: Hybrid Loss + PSNR at t sampled ---
        model.eval()
        val_eps_loss_sum = 0.0
        val_perceptual_loss_sum = 0.0
        val_psnr_sum = 0.0

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
                weights = gaussian_weight_map(pred_eps.shape, sigma=config.gauss_sigma).to(device)
                eps_loss = (F.mse_loss(pred_eps, true_eps, reduction='none') * weights).mean()
                val_eps_loss_sum += eps_loss.item()

                # 2. Perceptual Loss
                perceptual_loss = perceptual_loss_fn(x0_pred, hr_batch)
                val_perceptual_loss_sum += perceptual_loss.item()

                # PSNR on x0_pred
                val_psnr_sum += psnr(x0_pred, hr_batch).item()

        val_eps_loss_avg = val_eps_loss_sum / len(val_loader)
        val_perceptual_loss_avg = val_perceptual_loss_sum / len(val_loader)
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
                "train/eps_loss_weighted": train_eps_loss_avg,
                "train/perceptual_loss_scaled": config.lambda_perceptual * train_perceptual_loss_avg,
                "val/eps_loss": val_eps_loss_avg,
                "val/perceptual_loss": val_perceptual_loss_avg,
                "val/total_loss": val_total_loss_avg,
                "val/psnr_x0": val_psnr_avg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step(val_total_loss_avg)

        # --- Visualization every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                idx = random.randint(0, len(val_ds) - 1)
                lr_img, hr_img = val_ds[idx]  # both are asinh-normalized tensors
                lr_img = lr_img.unsqueeze(0).to(device)  # (1, 1, 21, 21)

                # run a relatively small number of sampling steps for speed
                sr_sample = model.sample(lr_img, num_steps=config.inference_steps)[0].cpu()  # (1, 105, 105)

                # Per-image min-max mapping to [0,1] for viz
                sr_sample_viz = to_viz(sr_sample)           # (1, H, W)
                hr_img_viz = to_viz(hr_img)                 # (1, H, W)

                # Interpolate LR to match HR size for visual comparison
                hr_up_interp = F.interpolate(
                    lr_img[0].cpu().unsqueeze(0),  # (1, 1, 21, 21)
                    size=hr_img.shape[1:],         # (H, W)
                    mode="bilinear",
                    align_corners=False,
                )[0]                               # (1, H, W)
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
