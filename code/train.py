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
        "loss_function": "Hybrid Loss (Weighted MSE + Perceptual Loss)",
        "mse_weight": 1.0,
        "l1_weight": 0.1,
        "perceptual_weight": 0.01,
        "weight_decay": 0.01
    }
    # Set defaults only if they are not already set by wandb
    for key, value in default_config.items():
        if not hasattr(wandb.config, key):
            setattr(wandb.config, key, value)

def get_loss_function(config):
    if config.loss_function == "Hybrid Loss (Weighted MSE + Perceptual Loss)":
        return HybridLoss(
            mse_weight=config.mse_weight, 
            l1_weight=config.l1_weight, 
            perceptual_weight=config.perceptual_weight
        )
    elif config.loss_function == "MSELoss":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function}")

# Initialize Weights & Biases with dynamic configuration setup
wandb.init(project="super-resolution-diffusion", config={"entity": "your_wandb_entity_name"})
setup_config_defaults()  # Set default values
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
activation_fn = getattr(nn, config.activation_function)  # Get activation function dynamically

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
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust these values based on your data analysis
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
        print(f"📌 Model saved at {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            random_idx = random.randint(0, len(test_loader.dataset) - 1)
            lr_img, hr_img = test_loader.dataset[random_idx]
            lr_img = lr_img.unsqueeze(0).to(device)
            t_test = torch.zeros((1,), dtype=torch.long, device=device)
            sr_img = model(lr_img, t_test).cpu().squeeze(0)

            wandb.log({
                "low_res": wandb.Image(lr_img, caption=f"Low-Res {random_idx}"),
                "super_res": wandb.Image(sr_img, caption=f"Super-Res {random_idx}"),
                "high_res": wandb.Image(hr_img, caption=f"High-Res {random_idx}"),
            })

        model.train()

    if avg_loss < best_loss - 0.001:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"🏆 Best model updated (Loss: {best_loss:.6f})")

torch.save(model.state_dict(), "checkpoints/final_model.pth")
print("✅ Final model saved.")
