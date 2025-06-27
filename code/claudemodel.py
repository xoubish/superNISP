import os
import json
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from dataset import SuperResolutionDataset
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

# Set these at the beginning of your script
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
torch.backends.cudnn.allow_tf32 = True

def resize_to_match(img, target_shape):
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def compute_psnr_ssim(pred, target):
    pred_np = pred.squeeze().numpy()
    target_np = target.squeeze().numpy()
    psnr_val = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    ssim_val = structural_similarity(target_np, pred_np, data_range=1.0)
    return psnr_val, ssim_val

# Use channels_last memory format for better performance
def convert_to_channels_last(model):
    return model.to(memory_format=torch.channels_last)

def get_data_splits(dataset, val_split=0.2, test_split=0.1, seed=42, split_save_path="splits.json"):
    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_test = int(test_split * n_total)
    n_train = n_total - n_val - n_test

    if os.path.exists(split_save_path):
        with open(split_save_path, "r") as f:
            indices = json.load(f)
        print(f"Loaded split indices from {split_save_path}")
    else:
        torch.manual_seed(seed)
        all_indices = torch.randperm(n_total).tolist()
        indices = {
            "train": all_indices[:n_train],
            "val":   all_indices[n_train:n_train + n_val],
            "test":  all_indices[n_train + n_val:]
        }
        with open(split_save_path, "w") as f:
            json.dump(indices, f)
        print(f"Saved split indices to {split_save_path}")

    return {
        "train": Subset(dataset, indices["train"]),
        "val":   Subset(dataset, indices["val"]),
        "test":  Subset(dataset, indices["test"])
    }

class ResidualDenseBlock(nn.Module):
    """Enhanced Residual Dense Block with local feature fusion"""
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5  # Residual connection

class DetailEnhancementModule(nn.Module):
    """Advanced detail enhancement module"""
    def __init__(self, channels):
        super(DetailEnhancementModule, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)  # Changed to 1x1 conv
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels//4), kernel_size=1),  # Ensure at least 1 channel
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(1, channels//4), channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Register edge detection kernel as buffer
        edge_kernel = torch.tensor([[-1, -1, -1], 
                                   [-1, 8, -1], 
                                   [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer('edge_kernel', edge_kernel.reshape(1, 1, 3, 3))
        
    def forward(self, x):
        # Edge-aware feature extraction
        edge_kernel = self.edge_kernel.repeat(x.size(1), 1, 1, 1)
        edges = F.conv2d(x, edge_kernel, padding=1, groups=x.size(1))
        
        # Feature enhancement
        enhanced = self.edge_conv(x + edges)
        
        # Channel attention
        attention = self.attention(enhanced)
        
        return enhanced * attention

class EuclidToJWSTSuperResolution(nn.Module):
    def __init__(self, num_rrdb=8, features=64):  # Reduced from 16 to 8 blocks
        super(EuclidToJWSTSuperResolution, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, features, kernel_size=3, padding=1)
        
        # Fewer but more efficient residual blocks
        self.rrdb_blocks = nn.ModuleList([ResidualDenseBlock(features, growth_rate=24) for _ in range(num_rrdb)])  # Reduced growth rate
        
        # Trunk convolution
        self.trunk_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Simplified detail enhancement
        self.detail_enhancer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=1)  # 1x1 conv is faster
        )
        
        # More efficient upsampling - direct to target size
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * 25, kernel_size=3, padding=1),  # 25 = 5x5 for 5x upsampling
            nn.PixelShuffle(5),  # Direct 5x upsampling (41*5 = 205)
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features//2, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # Initial features
        features = self.conv_first(x)
        
        # Residual processing
        trunk = features
        for block in self.rrdb_blocks:
            trunk = block(trunk)
        
        features = features + self.trunk_conv(trunk)
        
        # Detail enhancement
        features = self.detail_enhancer(features)
        
        # Direct upsampling to target size
        features = self.upsample(features)
        
        # Final output
        output = self.final_conv(features)
        
        return output

# SSIM Loss Implementation
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

def train_two_stage(
    lr_hdf5_path,
    hr_hdf5_path,
    val_split=0.2,
    batch_size=8,
    num_epochs_stage1=50,
    num_epochs_stage2=50,
    use_amp=True,
    sample_fraction=1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    full_dataset = SuperResolutionDataset(
        lr_hdf5_path=lr_hdf5_path,
        hr_hdf5_path=hr_hdf5_path,
        split='train',
        sample_fraction=sample_fraction,
        inference_mode=False
    )

    torch.manual_seed(42)
    n = len(full_dataset)
    n_val = int(val_split * n)
    n_train = n - n_val

    splits = get_data_splits(full_dataset, val_split=val_split, test_split=0.1)
    train_ds = splits["train"]
    val_ds   = splits["val"]

    #train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    print(f"Dataset split: {n_train} train, {n_val} val samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = EuclidToJWSTSuperResolution(num_rrdb=6, features=48).to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    model = convert_to_channels_last(model)

    wandb.watch(model, log="all", log_freq=100)

    def log_wandb_image_triplet_batch(lr_imgs, hr_imgs, out_imgs, stage_name, step, num_samples=4):
        """
        Log a few (num_samples) LR/SR/HR image triplets as a wandb image grid.
        """
        images = []
        for i in range(min(num_samples, lr_imgs.size(0))):
            lr = lr_imgs[i].detach().cpu().squeeze().numpy()
            hr = hr_imgs[i].detach().cpu().squeeze().numpy()
            sr = out_imgs[i].detach().cpu().squeeze().numpy()


            # Resize LR to match HR shape
            lr_resized = resize_to_match(lr, hr.shape)
            
            psnr_val, ssim_val = compute_psnr_ssim(torch.tensor(sr), torch.tensor(hr))
    
            caption = f"Sample {i} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}"
            images.append(wandb.Image(np.concatenate([lr_resized, sr, hr], axis=1), caption=caption))
    
        wandb.log({f"{stage_name}/ImageTriplets": images}, step=step)


    # --- Stage 1 ---
    print("\n=== Stage 1: Initial Training ===")
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(opt1, max_lr=1e-3, epochs=num_epochs_stage1, steps_per_epoch=len(train_loader))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    best1 = float('inf')

    for ep in range(num_epochs_stage1):
        model.train()
        running = 0.0
        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device, memory_format=torch.channels_last, non_blocking=True)
            hr_img = hr_img.to(device, memory_format=torch.channels_last, non_blocking=True)

            opt1.zero_grad()
            if scaler:
                with autocast():
                    out = model(lr_img)
                    loss = 0.7*l1(out, hr_img) + 0.3*mse(out, hr_img)
                scaler.scale(loss).backward()
                scaler.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt1)
                scaler.update()
            else:
                out = model(lr_img)
                loss = 0.7*l1(out, hr_img) + 0.3*mse(out, hr_img)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt1.step()

            running += loss.item()
        sched1.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                out = model(lr_img)
                val_loss += (0.7*l1(out, hr_img) + 0.3*mse(out, hr_img)).item()

        avg_train = running/len(train_loader)
        avg_val   = val_loss/len(val_loader)
        print(f"Stage1 Ep{ep+1}: train={avg_train:.4e}, val={avg_val:.4e}")
        wandb.log({"Stage1/train_loss": avg_train, "Stage1/val_loss": avg_val}, step=ep)

        if ep % 10 == 0:
            log_wandb_image_triplet_batch(lr_img, hr_img, out, "Stage1", ep)


        if avg_val < best1:
            best1 = avg_val
            torch.save(model.state_dict(), 'stage1_best.pth')

    # --- Stage 2 ---
    print("\n=== Stage 2: Fine-tuning ===")
    model.load_state_dict(torch.load('stage1_best.pth'))
    opt2 = torch.optim.Adam(model.parameters(), lr=5e-5)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=num_epochs_stage2, eta_min=1e-6)
    ssim = SSIMLoss()
    best2 = float('inf')

    for ep in range(num_epochs_stage2):
        model.train()
        running = 0.0
        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            opt2.zero_grad()
            out = model(lr_img)
            loss = 0.5*l1(out, hr_img) + 0.3*mse(out, hr_img) + 0.2*ssim(out, hr_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt2.step()
            running += loss.item()
        sched2.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                out = model(lr_img)
                val_loss += (0.5*l1(out, hr_img) + 0.3*mse(out, hr_img) + 0.2*ssim(out, hr_img)).item()

        avg_train = running/len(train_loader)
        avg_val   = val_loss/len(val_loader)
        print(f"Stage2 Ep{ep+1}: train={avg_train:.4e}, val={avg_val:.4e}")
        wandb.log({"Stage2/train_loss": avg_train, "Stage2/val_loss": avg_val}, step=ep)

        if ep % 5 == 0:
            log_wandb_image_triplet_batch(lr_img, hr_img, out, "Stage2", ep)


        if avg_val < best2:
            best2 = avg_val
            torch.save(model.state_dict(), 'final_best.pth')

    model.load_state_dict(torch.load('final_best.pth'))
    return model
