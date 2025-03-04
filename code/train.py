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
        "epochs": 5,
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

# Verify if model parameters are on GPU
for name, param in model.named_parameters():
    print(f"{name} is on {param.device}")  # Ensure all say "cuda:0"

# Train-Test Dataloaders
train_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train.hdf5", "../data/Nircam_train.hdf5", split="train"),
    batch_size=wandb.config["batch_size"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True
)
test_loader = DataLoader(
    SuperResolutionDataset("../data/Nisp_train.hdf5", "../data/Nircam_train.hdf5", split="test"),
    batch_size=wandb.config["batch_size"], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True
)

# Enable Mixed Precision Training
scaler = torch.amp.GradScaler(device='cuda')  

# Training loop
num_epochs = wandb.config["epochs"]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    print(f"Starting Epoch {epoch+1}...")

    for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
        start_batch_time = time.time()  # Measure batch processing time

        lr_batch, hr_batch = lr_batch.to(device, non_blocking=True), hr_batch.to(device, non_blocking=True)
        t = torch.randint(0, model.diffusion.timesteps, (lr_batch.shape[0],), device=device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):  # Updated autocast
            output = model(lr_batch, t)
            loss = criterion(output, hr_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()  # Forces GPU execution immediately

        batch_time = time.time() - start_batch_time

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    elapsed_time = time.time() - start_time

    # Log to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "time_per_epoch": elapsed_time
    })

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")

wandb.finish()

# Save trained model
torch.save(model.state_dict(), "super_resolution_model.pth")
print("Model saved successfully!")
