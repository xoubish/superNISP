# train_sr3.py
#
# Training script for SR3SuperResolution using Weighted epsilon-loss
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
# PSNR helper (optional monitoring)
# -----------------------------------------------------------

def psnr(pred, target):
    """Computes Peak Signal-to-Noise Ratio (PSNR)."""
    # Ensure inputs are normalized to [-1, 1] for meaningful MSE/PSNR
    mse = F.mse_loss(pred, target)
    if mse <= 1e-8:
        return torch.tensor(99.0, device=pred.device)
    # Assuming data range is 2.0 (from -1 to 1)
    return 10 * torch.log10(4.0 / mse)


# -----------------------------------------------------------
# Helper Functions: Gaussian Weight Map and Perceptual Loss
# (Adopted from user's provided code for centering loss)
# -----------------------------------------------------------

def gaussian_weight_map(shape, sigma=0.3):
    """
    Generates a Gaussian weight map centered in the middle of the image.
    Used to weight the epsilon loss, prioritizing the center galaxy.
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
    Inputs are expected to be in the [-1, 1] range.
    """
    def __init__(self):
        super().__init__()
        # Use first few layers of VGG16 for low-level feature comparison
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:5]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # VGG expects 3 channels; we'll repeat the single-channel input

    def forward(self, pred, target):
        # VGG requires 3 channels, so we repeat the single channel (1 -> 3)
        # Assuming pred/target are normalized to [-1, 1]
        pred_features = self.feature_extractor(pred.repeat(1, 3, 1, 1))
        target_features = self.feature_extractor(target.repeat(1, 3, 1, 1))
        
        return F.mse_loss(pred_features, target_features)


# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    # --- New Loss Arguments ---
    parser.add_argument("--gauss_sigma", type=float, default=0.4, 
                        help="Sigma for Gaussian weight map for epsilon loss.")
    parser.add_argument("--lambda_perceptual", type=float, default=1e-3, 
                        help="Weight for the VGG-based Perceptual Loss on x0_pred.")
    # --------------------------
    parser.add_argument("--project", type=str, default="superNISP_sr3")
    parser.add_argument("--entity", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- W&B ---
    wandb.init(
        project=args.project,
        entity=args.entity,
        config=vars(args),
    )
    config = wandb.config

    # --- Data ---
    # (Assuming SuperResolutionDataset is available and correctly loads data)
    train_ds = SuperResolutionDataset(
        "../../data/Nisp_train_cosmos.hdf5",
        "../../data/Nircam_train_cosmos.hdf5",
        split="train",
        sample_fraction=0.2,
    )
    val_ds = SuperResolutionDataset(
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

    # Initialize Perceptual Loss
    perceptual_loss_fn = PerceptualLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    os.makedirs("checkpoints_sr3", exist_ok=True)
    best_val_loss = float("inf")

    # Pre-calculate Gaussian weights for the expected HR size (128x128 padded)
    # The UNet pads to 128x128 internally, so we use that size for weights
    # Note: These weights will be used to weight the epsilon loss.
    H_padded, W_padded = 128, 128 
    # The weights must be dynamically calculated inside the loop 
    # or before each forward pass if the batch size changes,
    # but for consistent size (128x128), we can define a template.
    # We will still calculate it inside the loop to get the correct batch size (B).

    # --- Training ---
    for epoch in range(config.epochs):
        model.train()
        train_eps_loss_sum = 0.0
        train_perceptual_loss_sum = 0.0

        start = time.time()
        for lr_batch, hr_batch in train_loader:
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

            optimizer.zero_grad()

            # The model now returns pred_eps, true_eps, and the predicted x0
            pred_eps, true_eps, x0_pred = model(lr_batch, hr_batch, t)
            
            # --- 1. Weighted Epsilon Loss ---
            # Get Gaussian weights matching the batch size and padded HR size (128x128)
            weights = gaussian_weight_map(pred_eps.shape, sigma=config.gauss_sigma).to(device)
            
            # Weighted MSE loss: L_eps = mean(weights * (pred_eps - true_eps)^2)
            eps_loss = (F.mse_loss(pred_eps, true_eps, reduction='none') * weights).mean()
            
            # --- 2. Perceptual Loss on x0 prediction ---
            # hr_batch is the true x0 (ground truth HR)
            perceptual_loss = perceptual_loss_fn(x0_pred, hr_batch)
            
            # --- 3. Total Loss ---
            loss = eps_loss + config.lambda_perceptual * perceptual_loss
            
            loss.backward()
            optimizer.step()

            train_eps_loss_sum += eps_loss.item()
            train_perceptual_loss_sum += perceptual_loss.item()

        train_eps_loss_avg = train_eps_loss_sum / len(train_loader)
        train_perceptual_loss_avg = train_perceptual_loss_sum / len(train_loader)
        train_total_loss_avg = train_eps_loss_avg + config.lambda_perceptual * train_perceptual_loss_avg


        # --- Validation: Hybrid Loss + PSNR at t=0 ---
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
                
                # Retrieve all outputs
                pred_eps, true_eps, x0_pred = model(lr_batch, hr_batch, t)
                
                # 1. Weighted Epsilon Loss (using the same weights logic)
                weights = gaussian_weight_map(pred_eps.shape, sigma=config.gauss_sigma).to(device)
                eps_loss = (F.mse_loss(pred_eps, true_eps, reduction='none') * weights).mean()
                val_eps_loss_sum += eps_loss.item()

                # 2. Perceptual Loss
                perceptual_loss = perceptual_loss_fn(x0_pred, hr_batch)
                val_perceptual_loss_sum += perceptual_loss.item()
                
                # --- PSNR Calculation (x0_pred on validation batch) ---
                # Note: We can reuse x0_pred calculated from the forward pass
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
                "val/eps_loss": val_eps_loss_avg,
                "val/perceptual_loss": val_perceptual_loss_avg,
                "val/total_loss": val_total_loss_avg,
                "val/psnr_x0": val_psnr_avg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # visualize sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                idx = random.randint(0, len(val_ds) - 1)
                lr_img, hr_img = val_ds[idx]
                lr_img = lr_img.unsqueeze(0).to(device)

                # run a relatively small number of sampling steps for speed
                sr_sample = model.sample(lr_img, num_steps=100)[0].cpu()

                # Interpolate LR to match HR size for visual comparison
                hr_up_interp = F.interpolate(
                    lr_img[0].cpu().unsqueeze(0), 
                    size=hr_img.shape[1:], 
                    mode="bilinear", 
                    align_corners=False
                )[0]

                wandb.log(
                    {
                        "viz_super_res": wandb.Image(sr_sample, caption=f"SR epoch {epoch+1}"),
                        "viz_low_res_interp": wandb.Image(hr_up_interp, caption="LR (Interpolated)"),
                        "viz_high_res": wandb.Image(hr_img, caption="HR (Ground Truth)"),
                    }
                )

        # save best model based on total validation loss
        if val_total_loss_avg < best_val_loss:
            best_val_loss = val_total_loss_avg
            best_path = os.path.join("checkpoints_sr3", "best_sr3.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ New best model saved: {best_path}")

        # per-epoch checkpoint (optional)
        ckpt_path = os.path.join("checkpoints_sr3", f"sr3_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()