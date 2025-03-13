import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset  
from model_attention import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler  
import random
from losses import HybridLoss

# Initialize Weights & Biases
wandb.init(project="super-resolution-diffusion", config={
    "epochs": 200,
    "learning_rate": 3e-4,
    "batch_size": 32,
    "optimizer": "AdamW",
    "loss_function": "Hybrid Loss (Weighted MSE + Perceptual Loss)",
    "in_channels": 1,
    "out_channels": 1,
    "hidden_dim": 64,
    "activation_function": "ReLU",
    "num_heads": 4,
    "timesteps": 500,
    "upscale_factor": 2
})

# Enable CuDNN Optimization
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saving models
os.makedirs("checkpoints", exist_ok=True)
# Configure activation function dynamically based on a string
activation_fn = getattr(nn, wandb.config["activation_function"])

# Instantiate & Move Model to GPU
unet = SuperResDiffusionUNet(
    in_channels=wandb.config["in_channels"],
    out_channels=wandb.config["out_channels"],
    hidden_dim=wandb.config["hidden_dim"],
    num_heads=wandb.config["num_heads"],  # Correctly pass num_heads
    activation_fn=activation_fn  # Pass the class, not an instance
).to(device)


upsampler = Upsampler(
    in_channels=wandb.config["in_channels"],
    out_channels=wandb.config["out_channels"],
    upscale_factor=wandb.config["upscale_factor"]
).to(device)

model = SuperResolutionDiffusion(unet, upsampler).to(device)

# Define loss function & optimizer
criterion = HybridLoss(mse_weight=1.0, l1_weight=0.01, perceptual_weight=0.01).to(device)
optimizer = optim.AdamW(model.parameters(), lr=wandb.config["learning_rate"], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Load Data
train_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train.hdf5", "../data/Nircam_train.hdf5", split="train", sample_fraction=0.1),
    batch_size=wandb.config["batch_size"],
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train.hdf5", "../data/Nircam_train.hdf5", split="test", sample_fraction=0.1),
    batch_size=wandb.config["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# Enable Mixed Precision Training
scaler = torch.amp.GradScaler()

# Best Loss Tracker
best_loss = float("inf")

### Training Loop
num_epochs = wandb.config["epochs"]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
        # Free memory only if needed
        if torch.cuda.memory_allocated(device) > 0.95 * torch.cuda.get_device_properties(device).total_memory:
            torch.cuda.empty_cache()

        lr_batch, hr_batch = lr_batch.to(device, non_blocking=True), hr_batch.to(device, non_blocking=True)
        t = torch.randint(0, model.diffusion.timesteps, (lr_batch.shape[0],), device=device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):

            output = model(lr_batch, t)
            loss = criterion(output, hr_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    elapsed_time = time.time() - start_time

    print(f"✅ Epoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds. Loss: {avg_loss:.6f}")

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    scheduler.step(avg_loss)

    # Save model & log images every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"📌 Model saved at {checkpoint_path}")

        # Evaluate and log images
        model.eval()
        with torch.no_grad():
            random_idx = random.randint(0, len(test_loader.dataset) - 1)
            lr_img, hr_img = test_loader.dataset[random_idx]
            lr_img = lr_img.unsqueeze(0).to(device)

            t_test = torch.zeros((1,), dtype=torch.long, device=device)
            sr_img = model(lr_img, t_test).cpu().squeeze(0)

            # Log images to WandB
            wandb.log({
                "low_res": wandb.Image(lr_img, caption=f"Low-Res {random_idx}"),
                "super_res": wandb.Image(sr_img, caption=f"Super-Res {random_idx}"),
                "high_res": wandb.Image(hr_img, caption=f"High-Res {random_idx}"),
            })

        model.train()

    # Save best model only if significantly better
    if avg_loss < best_loss - 0.001:  # Avoid saving for tiny fluctuations
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"🏆 Best model updated (Loss: {best_loss:.6f})")

torch.save(model.state_dict(), "checkpoints/final_model.pth")
print("✅ Final model saved.")
