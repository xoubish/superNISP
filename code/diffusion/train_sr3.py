# train_sr3.py
#
# Training script for SR3SuperResolution using epsilon-loss only.
# Uses SuperResolutionDataset(lr_path, hr_path, split=..., sample_fraction=...).

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

from dataset import SuperResolutionDataset
from model_sr3 import SR3UNet, SR3SuperResolution


# -----------------------------------------------------------
# Simple PSNR helper (optional monitoring)
# -----------------------------------------------------------

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse <= 0:
        return torch.tensor(99.0, device=pred.device)
    return 10 * torch.log10(1.0 / mse)


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

    # --- Model ---
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    os.makedirs("checkpoints_sr3", exist_ok=True)
    best_val_loss = float("inf")

    # --- Training ---
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

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

            pred_eps, true_eps = model(lr_batch, hr_batch, t)
            loss = F.mse_loss(pred_eps, true_eps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation: epsilon loss + optionally PSNR at t=0 ---
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
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
                pred_eps, true_eps = model(lr_batch, hr_batch, t)
                loss = F.mse_loss(pred_eps, true_eps)
                val_loss += loss.item()

                # quick deterministic estimate at t=0 (x0 prediction)
                t0 = torch.zeros(B, dtype=torch.long, device=device)
                cond = F.interpolate(
                    lr_batch, scale_factor=5, mode="bilinear", align_corners=False
                )
                if hr_batch.shape[2:] != cond.shape[2:]:
                    hr_up = F.interpolate(
                        hr_batch,
                        size=cond.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    hr_up = hr_batch

                x0, eps0 = model.q_sample(hr_up, t0)
                pred_eps0 = model.unet(x0, t0, cond)
                if pred_eps0.shape[2:] != eps0.shape[2:]:
                    pred_eps0 = F.interpolate(
                        pred_eps0,
                        size=eps0.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                sqrt_ā_0 = model.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
                sqrt_one_minus_0 = model.sqrt_one_minus_alphas_cumprod[t0].view(
                    -1, 1, 1, 1
                )
                x0_pred = (x0 - sqrt_one_minus_0 * pred_eps0) / (sqrt_ā_0 + 1e-8)
                val_psnr += psnr(x0_pred, hr_up).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{config.epochs} "
            f"| train_eps_loss={train_loss:.4f} "
            f"| val_eps_loss={val_loss:.4f} "
            f"| val_psnr(x0_pred)={val_psnr:.2f} "
            f"| time={elapsed:.1f}s"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_eps_loss": train_loss,
                "val_eps_loss": val_loss,
                "val_psnr_x0": val_psnr,
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

                wandb.log(
                    {
                        "viz_super_res": wandb.Image(sr_sample, caption=f"SR epoch {epoch+1}"),
                        "viz_low_res": wandb.Image(lr_img[0].cpu(), caption="LR"),
                        "viz_high_res": wandb.Image(hr_img, caption="HR"),
                    }
                )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join("checkpoints_sr3", "best_sr3.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ New best model saved: {best_path}")

        # per-epoch checkpoint (optional)
        ckpt_path = os.path.join("checkpoints_sr3", f"sr3_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
