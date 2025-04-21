import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset
from model import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler
from losses import HybridLoss
import random
import time
from torchvision import transforms

def setup_config_defaults():
    default_config = {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "activation_function": "ReLU",
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 64,
        "timesteps": 500,
        "upscale_factor": 2,
        "mse_weight": 1.0,
        "l1_weight": 0.1,
        "perceptual_weight": 0.01,
        "shape_weight": 0.1,
        "weight_decay": 0.01
    }
    for key, value in default_config.items():
        if not hasattr(wandb.config, key):
            setattr(wandb.config, key, value)

def get_loss_function(config):
    return HybridLoss(
        mse_weight=config.mse_weight,
        l1_weight=config.l1_weight,
        perceptual_weight=config.perceptual_weight,
        shape_weight=config.shape_weight
    )

def compute_ellipticity_from_moments(img):
    B, _, H, W = img.shape
    device = img.device
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
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
    e2 = 2 * Mxy / (Mxx + Myy + 1e-8)
    return torch.stack([e1, e2], dim=1)

# Initialize Weights & Biases
wandb.init(project="super-resolution-diffusion", config={"entity": "your_wandb_entity_name"})
setup_config_defaults()
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
activation_fn = getattr(nn, config.activation_function)

unet = SuperResDiffusionUNet(
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    hidden_dim=config.hidden_dim,
    activation_fn=activation_fn
).to(device)

upsampler = Upsampler(
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    upscale_factor=config.upscale_factor
).to(device)

model = SuperResolutionDiffusion(unet, upsampler).to(device)

criterion = get_loss_function(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train_cosmos.hdf5", "../data/Nircam_train_cosmos.hdf5", split="train", sample_fraction=0.2),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train_cosmos.hdf5", "../data/Nircam_train_cosmos.hdf5", split="test", sample_fraction=0.1),
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

scaler = torch.amp.GradScaler()
best_loss = float("inf")

# Training loop
for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for lr_batch, hr_batch in train_loader:
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
        t = torch.randint(0, config.timesteps, (lr_batch.shape[0],), device=device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(lr_batch, t)
            loss = criterion(output, hr_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_time": time.time() - start_time
    })

    if (epoch + 1) % 20 == 0:
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            random_idx = random.randint(0, len(test_loader.dataset) - 1)
            lr_img, hr_img = test_loader.dataset[random_idx]
            lr_img = lr_img.unsqueeze(0).to(device)
            t_test = torch.zeros((1,), dtype=torch.long, device=device)
            sr_img = model(lr_img, t_test).cpu().squeeze(0)

            sr_img_tensor = sr_img.unsqueeze(0)
            hr_img_tensor = hr_img.unsqueeze(0)
            e_pred = compute_ellipticity_from_moments(sr_img_tensor)
            e_true = compute_ellipticity_from_moments(hr_img_tensor)
            shape_error = torch.abs(e_pred - e_true).squeeze()

            wandb.log({
                "low_res": wandb.Image(lr_img, caption=f"Low-Res {random_idx}"),
                "super_res": wandb.Image(sr_img, caption=f"Super-Res {random_idx}"),
                "high_res": wandb.Image(hr_img, caption=f"High-Res {random_idx}"),
                "shape_error_e1": shape_error[0].item(),
                "shape_error_e2": shape_error[1].item(),
            })

        model.train()

    if avg_loss < best_loss - 0.001:
        best_loss = avg_loss
        # Save best model inside wandb run directory
        best_model_path = os.path.join(wandb.run.dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated (Loss: {best_loss:.6f})")

final_model_path = os.path.join(wandb.run.dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Final model saved.")
wandb.finish()