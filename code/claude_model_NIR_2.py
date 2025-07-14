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

class ConfigurableResidualDenseBlock(nn.Module):
    """RDB with configurable growth rate"""
    def __init__(self, channels, growth_rate=32):
        super(ConfigurableResidualDenseBlock, self).__init__()
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
        return x + x5

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

class ConfigurableEuclidToJWSTSuperResolution(nn.Module):
    def __init__(self, num_rrdb=16, features=64, growth_rate=32):
        super(ConfigurableEuclidToJWSTSuperResolution, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, features, kernel_size=3, padding=1)
        
        # Residual Dense Blocks with configurable parameters
        self.rrdb_blocks = nn.ModuleList([
            ConfigurableResidualDenseBlock(features, growth_rate) 
            for _ in range(num_rrdb)
        ])
        
        # Trunk convolution after RRDBs
        self.trunk_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Detail enhancement module
        self.detail_enhancer = DetailEnhancementModule(features)
        
        # Upsampling pathway
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
        
        # Upsampling
        for upsampling_block in self.upsampling_blocks:
            features = upsampling_block(features)
        
        # Final reconstruction
        output = self.final_conv(features)
        
        # Interpolate to exact target size
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

# Dataset class
class ConfigurableEuclidToJWSTDataset(Dataset):
    def __init__(self, euclid_path, jwst_path, normalize_method='flux_preserving', 
                 augmentation_prob=0.5):
        # Load data
        self.euclid_data = np.load(euclid_path)
        self.jwst_data = np.load(jwst_path)
        
        # Verify shapes
        assert self.euclid_data.shape[0] == self.jwst_data.shape[0], "Number of images must match"
        assert self.euclid_data.shape[1:] == (41, 41), f"Euclid images should be 41x41"
        assert self.jwst_data.shape[1:] == (205, 205), f"JWST images should be 205x205"
        
        self.normalize_method = normalize_method
        self.augmentation_prob = augmentation_prob
        
    def normalize_data(self, euclid_img, jwst_img):
        if self.normalize_method == 'flux_preserving':
            return self._flux_preserving_norm(euclid_img), self._flux_preserving_norm(jwst_img), {'method': 'flux_preserving'}
        elif self.normalize_method == 'adaptive_hist':
            return self._adaptive_hist_norm(euclid_img), self._adaptive_hist_norm(jwst_img), {'method': 'adaptive_hist'}
        elif self.normalize_method == 'percentile':
            return self._percentile_norm(euclid_img, jwst_img)
        elif self.normalize_method == 'z_score':
            return self._z_score_norm(euclid_img, jwst_img)
        elif self.normalize_method == 'none':
            return euclid_img, jwst_img
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
    
    def _flux_preserving_norm(self, image):
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
    
    def _adaptive_hist_norm(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
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
    
    def _percentile_norm(self, euclid_img, jwst_img):
        # Percentile-based normalization
        p_low, p_high = 1, 99
        
        euclid_p_low, euclid_p_high = np.percentile(euclid_img, [p_low, p_high])
        jwst_p_low, jwst_p_high = np.percentile(jwst_img, [p_low, p_high])
        
        euclid_norm = np.clip((euclid_img - euclid_p_low) / (euclid_p_high - euclid_p_low), 0, 1)
        jwst_norm = np.clip((jwst_img - jwst_p_low) / (jwst_p_high - jwst_p_low), 0, 1)
        
        return euclid_norm, jwst_norm, {'method': 'percentile'}
    
    def _z_score_norm(self, euclid_img, jwst_img):
        # Z-score normalization
        euclid_norm = (euclid_img - np.mean(euclid_img)) / (np.std(euclid_img) + 1e-8)
        jwst_norm = (jwst_img - np.mean(jwst_img)) / (np.std(jwst_img) + 1e-8)
        
        return euclid_norm, jwst_norm, {'method': 'z_score'}
    
    def apply_augmentation(self, euclid_img, jwst_img):
        """Apply data augmentation with configurable probability"""
        if np.random.random() > self.augmentation_prob:
            return euclid_img, jwst_img
        
        # Random rotation
        if np.random.random() > 0.5:
            k = np.random.choice([1, 2, 3])
            euclid_img = np.rot90(euclid_img, k).copy()
            jwst_img = np.rot90(jwst_img, k).copy()
        
        # Random flips
        if np.random.random() > 0.5:
            euclid_img = np.fliplr(euclid_img).copy()
            jwst_img = np.fliplr(jwst_img).copy()
        
        if np.random.random() > 0.5:
            euclid_img = np.flipud(euclid_img).copy()
            jwst_img = np.flipud(jwst_img).copy()
        
        return euclid_img, jwst_img
    
    def __len__(self):
        return self.euclid_data.shape[0]
    
    def __getitem__(self, idx):
        euclid_img = self.euclid_data[idx].astype(np.float32)
        jwst_img = self.jwst_data[idx].astype(np.float32)
        
        # Apply augmentation
        euclid_img, jwst_img = self.apply_augmentation(euclid_img, jwst_img)
        
        # Normalize
        euclid_norm, jwst_norm, norm_params = self.normalize_data(euclid_img, jwst_img)
        
        # Convert to tensors
        euclid_tensor = torch.from_numpy(euclid_norm).unsqueeze(0)
        jwst_tensor = torch.from_numpy(jwst_norm).unsqueeze(0)
        
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

def train_with_config():
    """Training function that uses W&B config"""
    # Initialize W&B run
    wandb.init()
    config = wandb.config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create full dataset
    full_dataset = ConfigurableEuclidToJWSTDataset(
        euclid_path='../data/euclid_NIR_cosmos_41px_Y.npy',
        jwst_path='../data/jwst_cosmos_205px_F115W.npy',
        normalize_method=config.normalization_method,
        augmentation_prob=config.augmentation_prob
    )
    
    # SUBSET THE DATA FOR SWEEP
    total_samples = len(full_dataset)
    subset_size = min(8000, total_samples)  # Use max 8000 samples for sweep
    
    # Create subset indices
    torch.manual_seed(42)  # Reproducible subset
    all_indices = torch.randperm(total_samples)[:subset_size]
    
    # Create subset dataset
    subset_dataset = torch.utils.data.Subset(full_dataset, all_indices)
    
    # Split subset into train/val
    val_size = int(0.2 * subset_size)  # 20% for validation
    train_size = subset_size - val_size
    
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
    
    print(f"Sweep using subset: {train_size} train, {val_size} val samples")
    
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
    
    # Initialize model
    model = ConfigurableEuclidToJWSTSuperResolution(
        num_rrdb=config.num_rrdb,
        features=config.features,
        growth_rate=config.growth_rate
    ).to(device)
    
    # Watch model
    wandb.watch(model, log_freq=100)
    
    # Loss functions
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    ssim_criterion = SSIMLoss()
    
    # Stage 1 Training
    stage1_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr_stage1,
        weight_decay=config.weight_decay
    )
    
    stage1_scheduler = get_scheduler(
        stage1_optimizer, 
        config.scheduler_type,
        T_max=30,  # Reduced epochs for sweep
        patience=config.scheduler_patience
    )
    
    # Reduced epochs for hyperparameter search
    num_epochs_stage1 = 30
    num_epochs_stage2 = 20
    
    best_val_psnr = 0.0
    
    # Stage 1 training loop (abbreviated)
    for epoch in range(num_epochs_stage1):
        model.train()
        train_loss = 0.0
        
        for euclid_images, jwst_images, _ in train_loader:
            euclid_images = euclid_images.to(device)
            jwst_images = jwst_images.to(device)
            
            stage1_optimizer.zero_grad()
            outputs = model(euclid_images)
            
            # Configurable loss weights
            l1_loss = l1_criterion(outputs, jwst_images)
            mse_loss = mse_criterion(outputs, jwst_images)
            loss = config.l1_weight_stage1 * l1_loss + (1 - config.l1_weight_stage1) * mse_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            stage1_optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            for euclid_images, jwst_images, _ in val_loader:
                euclid_images = euclid_images.to(device)
                jwst_images = jwst_images.to(device)
                outputs = model(euclid_images)
                
                l1_loss = l1_criterion(outputs, jwst_images)
                mse_loss = mse_criterion(outputs, jwst_images)
                loss = config.l1_weight_stage1 * l1_loss + (1 - config.l1_weight_stage1) * mse_loss
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = calculate_metrics(outputs, jwst_images)
                val_metrics['psnr'] += batch_metrics['psnr']
                val_metrics['ssim'] += batch_metrics['ssim']
        
        # Average metrics
        avg_val_psnr = val_metrics['psnr'] / len(val_loader)
        avg_val_ssim = val_metrics['ssim'] / len(val_loader)
        
        # Update scheduler
        if config.scheduler_type == 'plateau':
            stage1_scheduler.step(val_loss / len(val_loader))
        else:
            stage1_scheduler.step()
        
        # Log metrics
        wandb.log({
            'stage1/epoch': epoch + 1,
            'stage1/train_loss': train_loss / len(train_loader),
            'stage1/val_loss': val_loss / len(val_loader),
            'stage1/val_psnr': avg_val_psnr,
            'stage1/val_ssim': avg_val_ssim,
            'stage1/learning_rate': stage1_optimizer.param_groups[0]['lr']
        })
        
        best_val_psnr = max(best_val_psnr, avg_val_psnr)
    
    # Stage 2 Training (similar structure with configurable parameters)
    stage2_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr_stage2,
        weight_decay=config.weight_decay
    )
    
    stage2_scheduler = get_scheduler(
        stage2_optimizer, 
        config.scheduler_type,
        T_max=num_epochs_stage2,
        patience=config.scheduler_patience
    )
    
    for epoch in range(num_epochs_stage2):
        model.train()
        train_loss = 0.0
        
        for euclid_images, jwst_images, _ in train_loader:
            euclid_images = euclid_images.to(device)
            jwst_images = jwst_images.to(device)
            
            stage2_optimizer.zero_grad()
            outputs = model(euclid_images)
            
            # Configurable loss combination
            l1_loss = l1_criterion(outputs, jwst_images)
            mse_loss = mse_criterion(outputs, jwst_images)
            ssim_loss = ssim_criterion(outputs, jwst_images)
            
            # Normalize weights
            total_weight = config.l1_weight_stage2 + config.mse_weight_stage2 + config.ssim_weight_stage2
            l1_w = config.l1_weight_stage2 / total_weight
            mse_w = config.mse_weight_stage2 / total_weight
            ssim_w = config.ssim_weight_stage2 / total_weight
            
            loss = l1_w * l1_loss + mse_w * mse_loss + ssim_w * ssim_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            stage2_optimizer.step()
            
            train_loss += loss.item()
        
        # Validation (similar to stage 1)
        model.eval()
        val_loss = 0.0
        val_metrics = {'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            for euclid_images, jwst_images, _ in val_loader:
                euclid_images = euclid_images.to(device)
                jwst_images = jwst_images.to(device)
                outputs = model(euclid_images)
                
                l1_loss = l1_criterion(outputs, jwst_images)
                mse_loss = mse_criterion(outputs, jwst_images)
                ssim_loss = ssim_criterion(outputs, jwst_images)
                
                loss = l1_w * l1_loss + mse_w * mse_loss + ssim_w * ssim_loss
                val_loss += loss.item()
                
                batch_metrics = calculate_metrics(outputs, jwst_images)
                val_metrics['psnr'] += batch_metrics['psnr']
                val_metrics['ssim'] += batch_metrics['ssim']
        
        avg_val_psnr = val_metrics['psnr'] / len(val_loader)
        avg_val_ssim = val_metrics['ssim'] / len(val_loader)
        
        # Update scheduler
        if config.scheduler_type == 'plateau':
            stage2_scheduler.step(val_loss / len(val_loader))
        else:
            stage2_scheduler.step()
        
        # Log metrics
        wandb.log({
            'stage2/epoch': epoch + 1,
            'stage2/train_loss': train_loss / len(train_loader),
            'stage2/val_loss': val_loss / len(val_loader),
            'stage2/val_psnr': avg_val_psnr,
            'stage2/val_ssim': avg_val_ssim,
            'stage2/learning_rate': stage2_optimizer.param_groups[0]['lr']
        })
        
        # best_val_psnr = max(best_val_psnr, avg_val_psnr)
    
    # Log final best metric
    wandb.log({'final_best_val_loss': val_loss / len(val_loader)})

# Create and run the sweep
def run_sweep():
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="euclid-jwst-hyperparameter-sweep"
    )
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_config, count=50)  # Run 50 experiments

if __name__ == "__main__":
    run_sweep()