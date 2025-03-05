import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset  
from model import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler  

# Initialize Weights & Biases
wandb.init(
    project="super-resolution-diffusion",
    config={
        "epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 64,  # Increased for better GPU utilization
        "optimizer": "AdamW",
        "loss_function": "MSELoss"
    }
)

# Enable CuDNN Optimization
torch.backends.cudnn.benchmark = True  

# Define loss function & optimizer
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate & Move Model to GPU
unet = SuperResDiffusionUNet(in_channels=1, out_channels=1).to(device)
upsampler = Upsampler().to(device)
model = SuperResolutionDiffusion(unet, upsampler).to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=wandb.config["learning_rate"], weight_decay=1e-5)

# Optimized Training DataLoader (Restored `num_workers=8`)
train_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train.hdf5", "../data/Nircam_train.hdf5", split="train", sample_fraction=0.1),
    batch_size=wandb.config["batch_size"],
    shuffle=True,
    num_workers=8,  # ✅ Restored 8 workers for fast training
    pin_memory=True, 
    persistent_workers=True  # ✅ Keeps workers alive for speed
)

# Optimized Test DataLoader (Restored `persistent_workers=True`)
test_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train.hdf5", "../data/Nircam_train.hdf5", split="test", sample_fraction=0.1),
    batch_size=wandb.config["batch_size"],
    shuffle=False,
    num_workers=2,  # ✅ Test set is smaller, 2 workers is enough
    pin_memory=True, 
    persistent_workers=True  # ✅ Restored to prevent slow reloading
)

# Enable Mixed Precision Training
scaler = torch.amp.GradScaler(device='cuda')  

# Training loop
num_epochs = wandb.config["epochs"]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
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

    # ✅ Print epoch info
    print(f"✅ Epoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds. Loss: {avg_loss:.6f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "time_per_epoch": elapsed_time,
        "gpu_usage": torch.cuda.memory_allocated(device) / 1e9,
        "gpu_max_allocated": torch.cuda.max_memory_allocated(device) / 1e9,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    # Log Images Every 5 Epochs (Restored direct batch fetching)
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            lr_batch, hr_batch = next(iter(test_loader))  # ✅ Restored direct batch fetching
            lr_batch = lr_batch.to(device)
            t_test = torch.zeros((lr_batch.shape[0],), dtype=torch.long, device=device)
            sr_batch = model(lr_batch, t_test).cpu()

        def normalize_image(img):
            img = img.squeeze().cpu().numpy()
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            return (img * 255).astype("uint8")

        wandb.log({
            "low_res": wandb.Image(normalize_image(lr_batch[0]), caption="Low-Res"),
            "super_res": wandb.Image(normalize_image(sr_batch[0]), caption="Super-Res"),
            "high_res": wandb.Image(normalize_image(hr_batch[0]), caption="High-Res"),
        })
        model.train()
