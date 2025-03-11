import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset  
from model import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler  
import random
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F

# Initialize Weights & Biases
wandb.init(
    project="super-resolution-diffusion",
    config={
        "epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 32,  # Reduce from 64 → 32 to avoid OOM
        "optimizer": "AdamW",
        "loss_function": "Hybrid Loss (Weighted MSE + Perceptual Loss + TV)"
    }
)

# ✅ Enable CuDNN Optimization
torch.backends.cudnn.benchmark = True  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### **🔹 Loss Functions for Galaxy Image Training**
def gaussian_weight_map(shape, sigma=0.3):
    """
    Generates a Gaussian weight map centered in the middle of the image.
    This prevents hard cut-offs in loss weighting and allows surroundings to be learned.
    """
    B, C, H, W = shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device='cuda'), torch.linspace(-1, 1, W, device='cuda'))
    d = torch.sqrt(x**2 + y**2)  # Distance from center
    weights = torch.exp(- (d**2) / (2 * sigma**2))  # Gaussian decay
    return weights.expand(B, C, H, W)

def weighted_mse_loss(pred, target, base_weight=1.0, center_boost=3.0):
    """
    Weighted MSE loss with a Gaussian weight map to emphasize the galaxy smoothly.
    """
    weight_map = gaussian_weight_map(target.shape, sigma=0.4)
    weight_map = base_weight + (center_boost - base_weight) * weight_map  # Rescale
    
    loss = weight_map * (pred - target) ** 2
    return loss.mean()

def total_variation_loss(img, weight=0.05):
    """
    Computes total variation loss to encourage smooth outputs.
    Reduces checkerboard artifacts.
    """
    loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return weight * loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features to preserve galaxy structures.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features  # ✅ Fixes deprecated warning
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:5]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # ✅ Freezes VGG weights

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred.repeat(1, 3, 1, 1))  # Convert grayscale to 3-channel
        target_features = self.feature_extractor(target.repeat(1, 3, 1, 1))
        return F.mse_loss(pred_features, target_features)


class HybridGalaxyLoss(nn.Module):
    """
    Hybrid loss that combines weighted MSE, perceptual loss, and total variation loss.
    """
    def __init__(self, center_boost=3.0, base_weight=1.0, perceptual_weight=0.05, tv_weight=0.05):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.center_boost = center_boost
        self.base_weight = base_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight

    def forward(self, pred, target):
        mse_loss = weighted_mse_loss(pred, target, base_weight=self.base_weight, center_boost=self.center_boost)
        perceptual_loss = self.perceptual_loss(pred, target)
        tv_loss = total_variation_loss(pred, self.tv_weight)
        return mse_loss + self.perceptual_weight * perceptual_loss + tv_loss


# ✅ Instantiate & Move Model to GPU
unet = SuperResDiffusionUNet(in_channels=1, out_channels=1).to(device)
upsampler = Upsampler().to(device)  # ✅ Comes from model.py
model = SuperResolutionDiffusion(unet, upsampler).to(device)

# ✅ Define loss function & optimizer
criterion = HybridGalaxyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=wandb.config["learning_rate"], weight_decay=1e-5)


### **🔹 Data Loaders**
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

# ✅ Enable Mixed Precision Training
scaler = torch.amp.GradScaler(device='cuda')  


### **🔹 Training Loop**
num_epochs = wandb.config["epochs"]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
        torch.cuda.empty_cache()  # ✅ **Free GPU memory before every batch**
        
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


    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "time_per_epoch": elapsed_time,
        "gpu_usage": torch.cuda.memory_allocated(device) / 1e9,
        "gpu_max_allocated": torch.cuda.max_memory_allocated(device) / 1e9,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            # Select a random index from the dataset directly
            random_idx = random.randint(0, len(test_loader.dataset) - 1)    
            lr_img, hr_img = test_loader.dataset[random_idx]
            lr_img = lr_img.unsqueeze(0).to(device)  # Add batch dimension

            #Generate super-resolution image
            t_test = torch.zeros((1,), dtype=torch.long, device=device)
            sr_img = model(lr_img, t_test).cpu().squeeze(0)

            # Normalize images
            lr_img = normalize_image(lr_img)
            sr_img = normalize_image(sr_img)
            hr_img = normalize_image(hr_img)

            # Log to WandB
            wandb.log({
                "low_res": wandb.Image(lr_img, caption=f"Low-Res {random_idx}"),
                "super_res": wandb.Image(sr_img, caption=f"Super-Res {random_idx}"),
                "high_res": wandb.Image(hr_img, caption=f"High-Res {random_idx}"),
            })
        model.train()