import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np
import math

# Set these at the beginning of your script
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
torch.backends.cudnn.allow_tf32 = True

# Use channels_last memory format for better performance
def convert_to_channels_last(model):
    return model.to(memory_format=torch.channels_last)

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

# Dataset class
class EuclidToJWSTDataset(Dataset):
    def __init__(self, euclid_path, jwst_path, normalize_method='flux_preserving', preload=True):
        # Load and preprocess all data at initialization
        print("Loading and preprocessing data...")
        self.euclid_data = np.load(euclid_path).astype(np.float32)
        self.jwst_data = np.load(jwst_path).astype(np.float32)
        
        # Verify shapes
        assert self.euclid_data.shape[0] == self.jwst_data.shape[0]
        assert self.euclid_data.shape[1:] == (41, 41)
        assert self.jwst_data.shape[1:] == (205, 205)
        
        if preload:
            # Preprocess all data
            self.preprocessed_data = []
            for i in range(len(self.euclid_data)):
                euclid_norm, jwst_norm, _ = self.normalize_data(
                    self.euclid_data[i], self.jwst_data[i]
                )
                self.preprocessed_data.append((
                    torch.from_numpy(euclid_norm).unsqueeze(0),
                    torch.from_numpy(jwst_norm).unsqueeze(0)
                ))
            print(f"Preprocessed {len(self.preprocessed_data)} samples")
        
        self.preload = preload
        self.normalize_method = normalize_method
        
    def normalize_data(self, euclid_img, jwst_img):
        # Simplified normalization for speed
        euclid_norm = (euclid_img - euclid_img.min()) / (euclid_img.max() - euclid_img.min() + 1e-8)
        jwst_norm = (jwst_img - jwst_img.min()) / (jwst_img.max() - jwst_img.min() + 1e-8)
        
        return euclid_norm, jwst_norm, {}
    
    def __len__(self):
        return len(self.euclid_data)
    
    def __getitem__(self, idx):
        if self.preload:
            return self.preprocessed_data[idx][0], self.preprocessed_data[idx][1], {}
        else:
            # Original processing
            euclid_img = self.euclid_data[idx]
            jwst_img = self.jwst_data[idx]
            euclid_norm, jwst_norm, norm_params = self.normalize_data(euclid_img, jwst_img)
            return torch.from_numpy(euclid_norm).unsqueeze(0), torch.from_numpy(jwst_norm).unsqueeze(0), norm_params

def train_two_stage(euclid_path, jwst_path, val_split=0.2, batch_size=8, 
                    num_epochs_stage1=50, num_epochs_stage2=50, use_amp=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize mixed precision scaler
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # Create dataset
    full_dataset = EuclidToJWSTDataset(
        euclid_path, 
        jwst_path, 
        preload=True
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    torch.manual_seed(42)  # For reproducible splits
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create dataloaders
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Adjust based on your CPU cores
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Use the efficient model
    model = EuclidToJWSTSuperResolution(num_rrdb=6, features=48).to(device)  # Smaller model

    # After model initialization
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled for faster execution")
    model = convert_to_channels_last(model)
    
    # Stage 1: Initial Training with Basic Loss Functions
    print("\n=== Stage 1: Initial Training ===")
    # More aggressive learning rate schedule
    stage1_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Higher LR, AdamW
    stage1_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        stage1_optimizer, 
        max_lr=0.001,
        epochs=num_epochs_stage1,
        steps_per_epoch=len(train_loader)
    )
    
    # Loss functions
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    best_val_loss_stage1 = float('inf')
    
    # Stage 1 training loop modification:
    for epoch in range(num_epochs_stage1):
        model.train()
        train_loss = 0.0
        
        for euclid_images, jwst_images, _ in train_loader:
            euclid_images = euclid_images.to(device, memory_format=torch.channels_last, non_blocking=True)
            jwst_images = jwst_images.to(device, memory_format=torch.channels_last, non_blocking=True)
            
            stage1_optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp and scaler is not None:
                with autocast('cuda'):
                    outputs = model(euclid_images)
                    loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                
                scaler.scale(loss).backward()
                scaler.unscale_(stage1_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(stage1_optimizer)
                scaler.update()
            else:
                outputs = model(euclid_images)
                loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                stage1_optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for euclid_images, jwst_images, _ in val_loader:
                euclid_images = euclid_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(euclid_images)
                loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                val_loss += loss.item()
        
        # Update learning rate
        stage1_scheduler.step()
        
        # Compute average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch statistics
        print(f'Stage 1 - Epoch {epoch+1}/{num_epochs_stage1}:')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Save best model from stage 1
        if avg_val_loss < best_val_loss_stage1:
            best_val_loss_stage1 = avg_val_loss
            torch.save(model.state_dict(), 'stage1_best_model.pth')
    
    # Stage 2: Fine-tuning with Advanced Losses
    print("\n=== Stage 2: Fine-tuning ===")
    
    # Load best model from stage 1
    model.load_state_dict(torch.load('stage1_best_model.pth'))
    
    # Advanced loss terms
    ssim_criterion = SSIMLoss()
    
    # Stage 2 optimizer with lower learning rate
    stage2_optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
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
        
        for euclid_images, jwst_images, _ in train_loader:
            # Move data to device
            euclid_images = euclid_images.to(device)
            jwst_images = jwst_images.to(device)
            
            # Zero the parameter gradients
            stage2_optimizer.zero_grad()
            
            # Forward pass
            outputs = model(euclid_images)
            
            # Compute advanced multi-component loss
            l1_loss = l1_criterion(outputs, jwst_images)
            mse_loss = mse_criterion(outputs, jwst_images)
            ssim_loss = ssim_criterion(outputs, jwst_images)
            
            # Combine losses with weights
            loss = (0.5 * l1_loss + 
                    0.3 * mse_loss + 
                    0.2 * ssim_loss)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            stage2_optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for euclid_images, jwst_images, _ in val_loader:
                euclid_images = euclid_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(euclid_images)
                
                # Compute validation loss similarly
                l1_loss = l1_criterion(outputs, jwst_images)
                mse_loss = mse_criterion(outputs, jwst_images)
                ssim_loss = ssim_criterion(outputs, jwst_images)
                
                loss = (0.5 * l1_loss + 
                        0.3 * mse_loss + 
                        0.2 * ssim_loss)
                
                val_loss += loss.item()
        
        # Update learning rate
        stage2_scheduler.step()
        
        # Compute average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch statistics
        print(f'Stage 2 - Epoch {epoch+1}/{num_epochs_stage2}:')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Save best model from stage 2
        if avg_val_loss < best_val_loss_stage2:
            best_val_loss_stage2 = avg_val_loss
            torch.save(model.state_dict(), 'final_best_model.pth')
    
    # Load and return the best model
    model.load_state_dict(torch.load('final_best_model.pth'))
    return model