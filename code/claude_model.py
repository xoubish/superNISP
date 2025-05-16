import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import math

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Lightweight localization network for small offsets
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # Regressor for the 2x3 affine matrix (primarily for translation)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 41 * 41, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )
        
        # Initialize to identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 41 * 41)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, channels):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels)
        )
        
    def forward(self, x):
        return x + self.body(x)

class MER2JWSTSuperResolution(nn.Module):
    def __init__(self, num_blocks=8, features=64):
        super(MER2JWSTSuperResolution, self).__init__()
        
        # Spatial transformer for fine alignment
        self.transformer = SpatialTransformer()
        
        # Initial feature extraction with noise handling
        self.initial = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual Channel Attention Blocks
        blocks = [RCAB(features) for _ in range(num_blocks)]
        self.residual_blocks = nn.Sequential(*blocks)
        
        # Feature reconstruction
        self.feature_reconstruction = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Upsampling (MER 41×41 → JWST 69×69, scale factor ~1.68)
        # We'll use a custom approach with pixel shuffle and precise resizing
        self.upsampling = nn.Sequential(
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2x upsampling to 82x82
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final reconstruction
        self.final = nn.Conv2d(features, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Apply spatial transformer for fine alignment
        x = self.transformer(x)
        
        # Initial feature extraction
        feat1 = self.initial(x)
        
        # Deep feature extraction with residual blocks
        feat2 = self.residual_blocks(feat1)
        
        # Global residual learning
        feat3 = self.feature_reconstruction(feat2) + feat1
        
        # Upsampling
        upsampled = self.upsampling(feat3)
        
        # Final convolution
        out = self.final(upsampled)
        
        # Resize to exactly 69x69 (JWST size)
        out = F.interpolate(out, size=(69, 69), mode='bicubic', align_corners=False)
        
        return out

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # L1 loss
    l1_criterion = nn.L1Loss()
    
    # Fixed SSIM implementation
    def gaussian(window_size, sigma):
        """Generate a 1D Gaussian kernel."""
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel=1):
        """Create a 2D Gaussian window."""
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(img1, img2, window_size=11, size_average=True):
        """Calculate SSIM with fixed padding (using integers)."""
        # Check shapes
        if img1.shape != img2.shape:
            raise ValueError(f"Input images must have the same dimensions, got {img1.shape} and {img2.shape}")
            
        # Get batch and channel dimensions
        if len(img1.shape) == 4:
            _, channel, _, _ = img1.shape
        else:
            channel = 1
            img1 = img1.unsqueeze(1)
            img2 = img2.unsqueeze(1)
            
        # Create window
        window = create_window(window_size, channel=channel).to(img1.device)
        
        # Calculate SSIM
        # Using integer padding (window_size//2) instead of float (window_size/2)
        padding = window_size // 2
        
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def ssim_loss(pred, target):
        """SSIM loss function using our fixed implementation."""
        return 1 - ssim(pred, target)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for mer_images, jwst_images in train_loader:
            mer_images = mer_images.to(device)
            jwst_images = jwst_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(mer_images)
            
            # Combined loss
            loss = 0.8 * l1_criterion(outputs, jwst_images)
            if epoch > 10:  # Start using SSIM loss after initial convergence
                loss += 0.2 * ssim_loss(outputs, jwst_images)
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mer_images, jwst_images in val_loader:
                mer_images = mer_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(mer_images)
                loss = l1_criterion(outputs, jwst_images)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_mer2jwst_model.pth')
    
    return model