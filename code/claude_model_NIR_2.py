import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import math
import wandb
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import cv2

from claude_sweep import sweep_config

# Global variable to track best overall performance
GLOBAL_BEST_LOSS = float('inf')
GLOBAL_BEST_CONFIG = None
GLOBAL_BEST_MODEL_PATH = None

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
    def __init__(self, num_rrdb=16, features=64):
        super(EuclidToJWSTSuperResolution, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, features, kernel_size=3, padding=1)
        
        # Residual Dense Blocks
        self.rrdb_blocks = nn.ModuleList([ResidualDenseBlock(features) for _ in range(num_rrdb)])
        
        # Trunk convolution after RRDBs
        self.trunk_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Detail enhancement module
        self.detail_enhancer = DetailEnhancementModule(features)
        
        # Upsampling pathway - Fixed for proper scaling
        # 41x41 -> 82x82 -> 164x164 -> then interpolate to 205x205
        self.upsampling_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        
        # Final reconstruction layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Initial feature extraction
        features = self.conv_first(x)
        
        # Residual dense blocks
        trunk = features
        for block in self.rrdb_blocks:
            trunk = block(trunk)
        
        # Combine initial and trunk features
        features = features + self.trunk_conv(trunk)
        
        # Detail enhancement
        features = self.detail_enhancer(features)
        
        # Upsampling (41x41 -> 82x82 -> 164x164)
        for upsampling_block in self.upsampling_blocks:
            features = upsampling_block(features)
        
        # Final reconstruction
        output = self.final_conv(features)
        
        # Interpolate to exact target size (164x164 -> 205x205)
        output = F.interpolate(output, size=(205, 205), mode='bilinear', align_corners=False)
        
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

# Flux conservation implementation
class FluxConservationLoss(nn.Module):
    """Loss to enforce flux conservation"""
    def __init__(self, weight=1.0):
        super(FluxConservationLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        # Calculate total flux for each image in the batch
        pred_flux = torch.sum(pred.view(pred.size(0), -1), dim=1)
        target_flux = torch.sum(target.view(target.size(0), -1), dim=1)
        
        # L1 loss on flux difference
        flux_loss = torch.mean(torch.abs(pred_flux - target_flux))
        
        return self.weight * flux_loss

# Adds center weighting to the model
class CenterWeightedLoss(nn.Module):
    """Loss that weights the center of the image more heavily"""
    def __init__(self, center_weight=3.0, base_loss=nn.L1Loss()):
        super(CenterWeightedLoss, self).__init__()
        self.center_weight = center_weight
        self.base_loss = base_loss
        self.weight_mask = None
    
    def create_weight_mask(self, height, width, device):
        """Create a mask that weights center pixels more heavily"""
        if self.weight_mask is not None and self.weight_mask.shape[-2:] == (height, width):
            return self.weight_mask.to(device)
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        
        # Calculate distance from center
        distance = torch.sqrt(x**2 + y**2)
        
        # Create weight mask (higher weight for center, lower for edges)
        # Use gaussian-like weighting
        weight_mask = torch.exp(-distance**2 / 0.5)  # Adjust 0.5 to control falloff
        
        # Normalize so average weight is 1.0
        weight_mask = weight_mask / weight_mask.mean()
        
        # Apply center weighting
        weight_mask = 1.0 + (self.center_weight - 1.0) * weight_mask
        
        # Add batch and channel dimensions
        self.weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)
        
        return self.weight_mask.to(device)
    
    def forward(self, pred, target):
        batch_size, channels, height, width = pred.shape
        
        # Create weight mask
        weight_mask = self.create_weight_mask(height, width, pred.device)
        weight_mask = weight_mask.expand(batch_size, channels, -1, -1)
        
        # Calculate weighted loss
        if isinstance(self.base_loss, nn.L1Loss):
            loss_map = torch.abs(pred - target)
        elif isinstance(self.base_loss, nn.MSELoss):
            loss_map = (pred - target) ** 2
        else:
            # Fallback to standard loss
            return self.base_loss(pred, target)
        
        # Apply weights
        weighted_loss = loss_map * weight_mask
        
        return weighted_loss.mean()

# Dataset class
class EuclidToJWSTDataset(Dataset):
    def __init__(self, euclid_path, jwst_path, normalize_method='flux_preserving'):
        # Load data
        self.euclid_data = np.load(euclid_path)
        self.jwst_data = np.load(jwst_path)
        
        # Verify shapes
        assert self.euclid_data.shape[0] == self.jwst_data.shape[0], "Number of Euclid and JWST images must match"
        assert self.euclid_data.shape[1:] == (41, 41), f"Euclid images should be 41x41, got {self.euclid_data.shape[1:]}"
        assert self.jwst_data.shape[1:] == (205, 205), f"JWST images should be 205x205, got {self.jwst_data.shape[1:]}"
        
        self.normalize_method = normalize_method
        self.transform = None  # Will be set externally if needed
        
    def normalize_data(self, euclid_img, jwst_img):
        """Truly flux-preserving normalization"""
        
        # Calculate total flux for both images
        euclid_flux = np.sum(euclid_img)
        jwst_flux = np.sum(jwst_img)
        
        # Store original flux ratio for later restoration
        flux_ratio = jwst_flux / (euclid_flux + 1e-10)
        
        # Normalize both images to [0, 1] based on their individual ranges
        # but preserve the flux relationship
        euclid_min, euclid_max = np.min(euclid_img), np.max(euclid_img)
        jwst_min, jwst_max = np.min(jwst_img), np.max(jwst_img)
        
        # Avoid division by zero
        euclid_range = euclid_max - euclid_min if euclid_max > euclid_min else 1.0
        jwst_range = jwst_max - jwst_min if jwst_max > jwst_min else 1.0
        
        # Normalize to [0, 1] while preserving relative intensities
        euclid_norm = (euclid_img - euclid_min) / euclid_range
        jwst_norm = (jwst_img - jwst_min) / jwst_range
        
        # Scale to preserve flux relationship
        # The key insight: don't artificially boost the target
        scale_factor = min(1.0, flux_ratio)  # Don't amplify beyond original
        jwst_norm = jwst_norm * scale_factor
        
        return euclid_norm, jwst_norm, {
            'method': 'flux_preserving',
            'euclid_min': euclid_min,
            'euclid_max': euclid_max,
            'jwst_min': jwst_min,
            'jwst_max': jwst_max,
            'flux_ratio': flux_ratio,
            'scale_factor': scale_factor
        }

    def adaptive_histogram_normalization(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Adaptive histogram equalization for astronomical images
        """
        # Convert to uint16 for CLAHE (OpenCV requirement)
        img_min, img_max = np.percentile(image, [0.5, self.clip_percentile])
        image_clipped = np.clip(image, img_min, img_max)
        
        # Normalize to 0-65535 range
        if img_max > img_min:
            image_norm = ((image_clipped - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
        else:
            image_norm = np.zeros_like(image_clipped, dtype=np.uint16)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image_clahe = clahe.apply(image_norm)
        
        # Convert back to float and normalize to [-1, 1]
        image_final = (image_clahe.astype(np.float32) / 32767.5) - 1.0
        
        return image_final
    
    def flux_preserving_normalization(self, image):
        """
        Flux-preserving normalization that maintains relative intensities
        """
        # Robust statistics to handle outliers
        img_median = np.median(image)
        img_mad = np.median(np.abs(image - img_median))  # Median Absolute Deviation
        
        # Use MAD-based scaling (more robust than std)
        if img_mad > 0:
            # Scale by MAD but preserve flux ratios
            image_norm = (image - img_median) / (img_mad * 1.4826)  # 1.4826 makes MAD comparable to std
            
            # Soft clipping to preserve dynamic range
            image_norm = np.tanh(image_norm / 3.0) * 3.0
        else:
            # Fallback for constant images
            image_norm = image - img_median
            
        return image_norm.astype(np.float32)
    
    def __len__(self):
        return self.euclid_data.shape[0]
    
    def __getitem__(self, idx):
        euclid_img = self.euclid_data[idx].astype(np.float32)
        jwst_img = self.jwst_data[idx].astype(np.float32)
        
        # Normalize
        euclid_norm, jwst_norm, norm_params = self.normalize_data(euclid_img, jwst_img)
        
        # Convert to tensors
        euclid_tensor = torch.from_numpy(euclid_norm).unsqueeze(0)
        jwst_tensor = torch.from_numpy(jwst_norm).unsqueeze(0)
        
        # Apply transforms if available (only to input, not target)
        if self.transform is not None:
            # For paired data, we need to apply the same transform to both
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            euclid_tensor = self.transform(euclid_tensor)
            torch.manual_seed(seed)
            jwst_tensor = self.transform(jwst_tensor)
        
        return euclid_tensor, jwst_tensor, norm_params

def calculate_metrics(pred, target):
    """Calculate various image quality metrics"""
    # Convert tensors to numpy if needed
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
    else:
        pred_np = pred
        target_np = target
    
    metrics = {}
    
    # Calculate metrics for each image in batch
    batch_psnr = []
    batch_ssim = []
    batch_mae = []
    batch_mse = []
    
    for i in range(pred_np.shape[0]):
        # Remove channel dimension for metric calculation
        pred_img = pred_np[i, 0] if pred_np.ndim == 4 else pred_np[i]
        target_img = target_np[i, 0] if target_np.ndim == 4 else target_np[i]
        
        # PSNR
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=target_img.max() - target_img.min())
        batch_psnr.append(psnr)
        
        # SSIM
        ssim = structural_similarity(target_img, pred_img, data_range=target_img.max() - target_img.min())
        batch_ssim.append(ssim)
        
        # MAE
        mae = np.mean(np.abs(pred_img - target_img))
        batch_mae.append(mae)
        
        # MSE
        mse = np.mean((pred_img - target_img) ** 2)
        batch_mse.append(mse)
    
    metrics['psnr'] = np.mean(batch_psnr)
    metrics['ssim'] = np.mean(batch_ssim)
    metrics['mae'] = np.mean(batch_mae)
    metrics['mse'] = np.mean(batch_mse)
    
    return metrics

def create_comparison_figure(euclid_imgs, pred_imgs, jwst_imgs, num_samples=4):
    """Create a comparison figure showing Euclid, Predicted, and JWST images"""
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(min(num_samples, euclid_imgs.shape[0])):
        # Convert tensors to numpy and remove channel dimension
        euclid_np = euclid_imgs[i, 0].detach().cpu().numpy() if torch.is_tensor(euclid_imgs) else euclid_imgs[i, 0]
        pred_np = pred_imgs[i, 0].detach().cpu().numpy() if torch.is_tensor(pred_imgs) else pred_imgs[i, 0]
        jwst_np = jwst_imgs[i, 0].detach().cpu().numpy() if torch.is_tensor(jwst_imgs) else jwst_imgs[i, 0]
        
        # Calculate metrics for this sample
        psnr = peak_signal_noise_ratio(jwst_np, pred_np, data_range=jwst_np.max() - jwst_np.min())
        ssim = structural_similarity(jwst_np, pred_np, data_range=jwst_np.max() - jwst_np.min())
        
        # Euclid input
        im1 = axes[0, i].imshow(euclid_np, cmap='viridis')
        axes[0, i].set_title(f'Euclid Input {i+1}\n(41×41)')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
        
        # Predicted output
        im2 = axes[1, i].imshow(pred_np, cmap='viridis')
        axes[1, i].set_title(f'Super-Resolved {i+1}\n(205×205)\nPSNR: {psnr:.2f}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
        
        # JWST target
        im3 = axes[2, i].imshow(jwst_np, cmap='viridis')
        axes[2, i].set_title(f'JWST Target {i+1}\n(205×205)\nSSIM: {ssim:.3f}')
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i], fraction=0.046)
    
    plt.tight_layout()
    return fig

def log_sample_predictions(model, val_loader, device, stage_name, epoch, num_samples=4):
    """Log sample predictions to W&B"""
    model.eval()
    
    with torch.no_grad():
        for euclid_imgs, jwst_imgs, _ in val_loader:
            euclid_imgs = euclid_imgs.to(device)
            jwst_imgs = jwst_imgs.to(device)
            
            # Get predictions
            pred_imgs = model(euclid_imgs)
            
            # Create comparison figure
            fig = create_comparison_figure(euclid_imgs, pred_imgs, jwst_imgs, num_samples)
            
            # Log to W&B
            wandb.log({
                f"final/sample_comparisons_{stage_name}": wandb.Image(fig),
                "epoch": epoch
            })
            
            plt.close(fig)
            break  # Only log first batch

def get_scheduler(optimizer, scheduler_type, **kwargs):
    """Get scheduler based on type"""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get('T_max', 50), eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 20), gamma=kwargs.get('gamma', 0.5)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=kwargs.get('patience', 10), factor=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_with_config_subset():
    """Complete training function for hyperparameter sweep with data subset"""
    wandb.init()
    config = wandb.config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create full dataset
    full_dataset = EuclidToJWSTDataset(
        euclid_path='../data/euclid_NIR_cosmos_41px_Y.npy',
        jwst_path='../data/jwst_cosmos_205px_F115W.npy',
        normalize_method='flux_preserving'  # Use your preferred normalization
    )
    
    # SUBSET THE DATA FOR SWEEP
    total_samples = len(full_dataset)
    subset_size = min(8000, total_samples)  # Use max 8000 samples for sweep
    
    print(f"Full dataset size: {total_samples}, using subset: {subset_size}")
    
    # Create reproducible subset
    torch.manual_seed(42)
    all_indices = torch.randperm(total_samples)[:subset_size]
    subset_dataset = torch.utils.data.Subset(full_dataset, all_indices)
    
    # Split subset into train/val
    val_size = int(0.2 * subset_size)
    train_size = subset_size - val_size
    
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
    
    print(f"Sweep training on: {train_size} train, {val_size} val samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Initialize model with config parameters
    model = EuclidToJWSTSuperResolution(
        num_rrdb=config.get('num_rrdb', 8),
        features=config.get('features', 64)
    ).to(device)
    
    # Watch model for W&B
    wandb.watch(model, log_freq=100)
    
    # Loss functions
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    flux_criterion = FluxConservationLoss(weight=0.5)
    center_weighted_l1 = CenterWeightedLoss(center_weight=3.0, base_loss=nn.L1Loss())
    
    # REDUCED EPOCHS FOR SWEEP
    num_epochs_stage1 = 30
    num_epochs_stage2 = 20
    
    # =================================================================
    # STAGE 1: Initial Training
    # =================================================================
    print(f"\n=== Stage 1: Initial Training ({num_epochs_stage1} epochs) ===")
    
    stage1_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('lr_stage1', 0.0001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    stage1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage1_optimizer, 
        T_max=num_epochs_stage1, 
        eta_min=1e-6
    )
    
    best_val_loss_stage1 = float('inf')
    
    for epoch in range(num_epochs_stage1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_l1_loss = 0.0
        train_mse_loss = 0.0
        train_flux_loss = 0.0
        train_center_loss = 0.0
        
        for batch_data in train_loader:
            # Handle both old and new dataset formats
            if len(batch_data) == 3:
                euclid_images, jwst_images, _ = batch_data
            else:
                euclid_images, jwst_images = batch_data
            
            euclid_images = euclid_images.to(device)
            jwst_images = jwst_images.to(device)
            
            stage1_optimizer.zero_grad()
            
            outputs = model(euclid_images)
            
            # Compute individual losses
            l1_loss = l1_criterion(outputs, jwst_images)
            mse_loss = mse_criterion(outputs, jwst_images)
            flux_loss = flux_criterion(outputs, jwst_images)
            center_loss = center_weighted_l1(outputs, jwst_images)
            
            # Combine losses with configurable weights
            l1_weight = config.get('l1_weight_stage1', 0.4)
            loss = (l1_weight * l1_loss + 
                    0.2 * mse_loss + 
                    0.2 * flux_loss + 
                    0.2 * center_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.get('gradient_clip_norm', 0.5)
            )
            stage1_optimizer.step()
            
            train_loss += loss.item()
            train_l1_loss += l1_loss.item()
            train_mse_loss += mse_loss.item()
            train_flux_loss += flux_loss.item()
            train_center_loss += center_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_mse_loss = 0.0
        val_flux_loss = 0.0
        val_center_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    euclid_images, jwst_images, _ = batch_data
                else:
                    euclid_images, jwst_images = batch_data
                
                euclid_images = euclid_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(euclid_images)
                
                l1_loss = l1_criterion(outputs, jwst_images)
                mse_loss = mse_criterion(outputs, jwst_images)
                flux_loss = flux_criterion(outputs, jwst_images)
                center_loss = center_weighted_l1(outputs, jwst_images)
                
                l1_weight = config.get('l1_weight_stage1', 0.4)
                loss = (l1_weight * l1_loss + 
                        0.2 * mse_loss + 
                        0.2 * flux_loss + 
                        0.2 * center_loss)
                
                val_loss += loss.item()
                val_l1_loss += l1_loss.item()
                val_mse_loss += mse_loss.item()
                val_flux_loss += flux_loss.item()
                val_center_loss += center_loss.item()
        
        # Update learning rate
        stage1_scheduler.step()
        
        # Compute averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to W&B
        wandb.log({
            'stage1/epoch': epoch + 1,
            'stage1/train_loss': avg_train_loss,
            'stage1/val_loss': avg_val_loss,
            'stage1/train_l1_loss': train_l1_loss / len(train_loader),
            'stage1/val_l1_loss': val_l1_loss / len(val_loader),
            'stage1/train_flux_loss': train_flux_loss / len(train_loader),
            'stage1/val_flux_loss': val_flux_loss / len(val_loader),
            'stage1/learning_rate': stage1_optimizer.param_groups[0]['lr'],
        })
        
        # Track best validation loss
        if avg_val_loss < best_val_loss_stage1:
            best_val_loss_stage1 = avg_val_loss
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Stage 1 - Epoch {epoch+1}/{num_epochs_stage1}: '
                  f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # =================================================================
    # STAGE 2: Fine-tuning with Advanced Losses
    # =================================================================
    print(f"\n=== Stage 2: Fine-tuning ({num_epochs_stage2} epochs) ===")
    
    # Add SSIM loss for stage 2
    ssim_criterion = SSIMLoss()
    
    stage2_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('lr_stage2', 0.00005),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage2_optimizer, 
        T_max=num_epochs_stage2, 
        eta_min=1e-6
    )
    
    best_val_loss_stage2 = float('inf')
    
    for epoch in range(num_epochs_stage2):
        # Training phase
        model.train()
        train_loss = 0.0
        train_l1_loss = 0.0
        train_mse_loss = 0.0
        train_ssim_loss = 0.0
        train_flux_loss = 0.0
        train_center_loss = 0.0
        
        for batch_data in train_loader:
            if len(batch_data) == 3:
                euclid_images, jwst_images, _ = batch_data
            else:
                euclid_images, jwst_images = batch_data
            
            euclid_images = euclid_images.to(device)
            jwst_images = jwst_images.to(device)
            
            stage2_optimizer.zero_grad()
            
            outputs = model(euclid_images)
            
            # Compute individual losses
            l1_loss = l1_criterion(outputs, jwst_images)
            mse_loss = mse_criterion(outputs, jwst_images)
            ssim_loss = ssim_criterion(outputs, jwst_images)
            flux_loss = flux_criterion(outputs, jwst_images)
            center_loss = center_weighted_l1(outputs, jwst_images)
            
            # Combine losses with configurable weights
            l1_weight = config.get('l1_weight_stage2', 0.3)
            mse_weight = config.get('mse_weight_stage2', 0.2)
            ssim_weight = config.get('ssim_weight_stage2', 0.1)
            
            # Normalize weights
            total_weight = l1_weight + mse_weight + ssim_weight + 0.2 + 0.2  # +flux +center
            l1_w = l1_weight / total_weight
            mse_w = mse_weight / total_weight
            ssim_w = ssim_weight / total_weight
            flux_w = 0.2 / total_weight
            center_w = 0.2 / total_weight
            
            loss = (l1_w * l1_loss + 
                    mse_w * mse_loss + 
                    ssim_w * ssim_loss + 
                    flux_w * flux_loss + 
                    center_w * center_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.get('gradient_clip_norm', 0.5)
            )
            stage2_optimizer.step()
            
            train_loss += loss.item()
            train_l1_loss += l1_loss.item()
            train_mse_loss += mse_loss.item()
            train_ssim_loss += ssim_loss.item()
            train_flux_loss += flux_loss.item()
            train_center_loss += center_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_mse_loss = 0.0
        val_ssim_loss = 0.0
        val_flux_loss = 0.0
        val_center_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    euclid_images, jwst_images, _ = batch_data
                else:
                    euclid_images, jwst_images = batch_data
                
                euclid_images = euclid_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(euclid_images)
                
                l1_loss = l1_criterion(outputs, jwst_images)
                mse_loss = mse_criterion(outputs, jwst_images)
                ssim_loss = ssim_criterion(outputs, jwst_images)
                flux_loss = flux_criterion(outputs, jwst_images)
                center_loss = center_weighted_l1(outputs, jwst_images)
                
                # Use same weights as training
                loss = (l1_w * l1_loss + 
                        mse_w * mse_loss + 
                        ssim_w * ssim_loss + 
                        flux_w * flux_loss + 
                        center_w * center_loss)
                
                val_loss += loss.item()
                val_l1_loss += l1_loss.item()
                val_mse_loss += mse_loss.item()
                val_ssim_loss += ssim_loss.item()
                val_flux_loss += flux_loss.item()
                val_center_loss += center_loss.item()
        
        # Update learning rate
        stage2_scheduler.step()
        
        # Compute averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to W&B
        wandb.log({
            'stage2/epoch': epoch + 1,
            'stage2/train_loss': avg_train_loss,
            'stage2/val_loss': avg_val_loss,
            'stage2/train_l1_loss': train_l1_loss / len(train_loader),
            'stage2/val_l1_loss': val_l1_loss / len(val_loader),
            'stage2/train_ssim_loss': train_ssim_loss / len(train_loader),
            'stage2/val_ssim_loss': val_ssim_loss / len(val_loader),
            'stage2/train_flux_loss': train_flux_loss / len(train_loader),
            'stage2/val_flux_loss': val_flux_loss / len(val_loader),
            'stage2/learning_rate': stage2_optimizer.param_groups[0]['lr'],
        })
        
        # Track best validation loss
        if avg_val_loss < best_val_loss_stage2:
            best_val_loss_stage2 = avg_val_loss
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Stage 2 - Epoch {epoch+1}/{num_epochs_stage2}: '
                  f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # =================================================================
    # Final Logging for Sweep Optimization
    # =================================================================
    
    # Log final metrics for sweep to optimize
    final_metrics = {
        'final_val_loss': best_val_loss_stage2,  # Primary metric for sweep
        'stage1_best_val_loss': best_val_loss_stage1,
        'stage2_best_val_loss': best_val_loss_stage2,
        'combined_val_loss': best_val_loss_stage1 + best_val_loss_stage2,
        'total_epochs': num_epochs_stage1 + num_epochs_stage2,
        'subset_size': subset_size
    }
    
    wandb.log(final_metrics)
    
    print(f"\nSweep run completed!")
    print(f"Stage 1 best val loss: {best_val_loss_stage1:.6f}")
    print(f"Stage 2 best val loss: {best_val_loss_stage2:.6f}")
    print(f"Final val loss (sweep metric): {best_val_loss_stage2:.6f}")
    
    # Clean up
    wandb.finish()

# Create and run the sweep
def run_sweep():
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="euclid-jwst-hyperparameter-sweep"
    )
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_config_subset, count=50)  # Run 50 experiments

if __name__ == "__main__":
    run_sweep()