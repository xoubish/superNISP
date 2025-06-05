# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_ssim
# import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with local feature fusion"""
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.beta + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels, growth_rate=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_rate)
        self.rdb2 = ResidualDenseBlock(channels, growth_rate)
        self.rdb3 = ResidualDenseBlock(channels, growth_rate)
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.beta + x

class HighFrequencyAttention(nn.Module):
    """Attention module for high-frequency details"""
    def __init__(self, channels):
        super(HighFrequencyAttention, self).__init__()
        self.conv_edge = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Extract edge information using Laplacian kernel
        edge_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                   dtype=torch.float32).reshape(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1).to(x.device)
        edge_map = F.conv2d(x, edge_filter, padding=1, groups=x.size(1))
        
        # Process edge information
        edge_attention = self.sigmoid(self.conv_edge(edge_map))
        
        # Apply attention
        return x * (1 + edge_attention)

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Lightweight localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Regressor for the 2x3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 10 * 10, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 2 * 3)
        )
        
        # Initialize to identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 10 * 10)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x

class EnhancedSuperResolution(nn.Module):
    def __init__(self, num_rrdb=8, features=64):
        super(EnhancedSuperResolution, self).__init__()
        
        # Spatial transformer for alignment
        self.transformer = SpatialTransformer()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, features, 3, padding=1)
        
        # RRDB blocks
        rrdb_blocks = [RRDB(features) for _ in range(num_rrdb)]
        self.body = nn.Sequential(*rrdb_blocks)
        
        # Trunk conv after RRDBs
        self.trunk_conv = nn.Conv2d(features, features, 3, padding=1)
        
        # High-frequency attention
        self.hf_attention = HighFrequencyAttention(features)
        
        # Upsampling (to 82x82, then resize to 69x69)
        self.upconv1 = nn.Conv2d(features, features * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Final convolutions with stronger focus on details
        self.detail_enhancer = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 1, 3, padding=1)
        )
        
    def forward(self, x):
        # Apply spatial transformer for alignment
        x = self.transformer(x)
        
        # Initial feature extraction
        fea = self.conv_first(x)
        
        # RRDBs
        trunk = self.trunk_conv(self.body(fea))
        
        # Add feature from trunk to initial features (global residual learning)
        fea = fea + trunk
        
        # Apply high-frequency attention
        fea = self.hf_attention(fea)
        
        # Upsampling
        fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        
        # Detail enhancement and final output
        out = self.detail_enhancer(fea)
        
        # Resize to 69x69 (JWST size)
        out = F.interpolate(out, size=(69, 69), mode='bicubic', align_corners=False)
        
        return out

def train_model_enhanced(model, train_loader, val_loader, num_epochs=200, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # L1 loss (pixel-wise)
    l1_criterion = nn.L1Loss()
    
    # MSE loss (pixel-wise)
    mse_criterion = nn.MSELoss()
    
    # SSIM loss implementation
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel=1):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(img1, img2, window_size=11, window=None, size_average=True):
        if window is None:
            window = create_window(window_size, img1.size(1)).to(img1.device)
        
        padding = window_size // 2
        mu1 = F.conv2d(img1, window, padding=padding, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=padding, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=img1.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
    
    def ssim_loss(pred, target):
        return 1 - ssim(pred, target)
    
    # High-frequency loss function (emphasize edges and details)
    def high_frequency_loss(pred, target):
        # Laplacian edge detector
        edge_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                  dtype=torch.float32).reshape(1, 1, 3, 3).to(pred.device)
        
        # Apply to both prediction and target
        pred_edges = F.conv2d(pred, edge_filter, padding=1)
        target_edges = F.conv2d(target, edge_filter, padding=1)
        
        # Compare edges using L1 loss
        return F.l1_loss(pred_edges, target_edges)
    
    # Histogram matching loss (to match overall intensity distributions)
    def histogram_loss(pred, target, bins=64):
        # Flatten the images
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        loss = 0
        for i in range(pred.size(0)):  # Loop through batch
            # Create histograms (normalize to 0-1 first)
            p_min, p_max = pred_flat[i].min(), pred_flat[i].max()
            t_min, t_max = target_flat[i].min(), target_flat[i].max()
            
            # Avoid division by zero
            p_range = p_max - p_min
            t_range = t_max - t_min
            if p_range == 0: p_range = 1
            if t_range == 0: t_range = 1
            
            p_norm = (pred_flat[i] - p_min) / p_range
            t_norm = (target_flat[i] - t_min) / t_range
            
            # Create histograms
            p_hist = torch.histc(p_norm, bins=bins, min=0, max=1)
            t_hist = torch.histc(t_norm, bins=bins, min=0, max=1)
            
            # Normalize histograms
            p_hist = p_hist / p_hist.sum()
            t_hist = t_hist / t_hist.sum()
            
            # Calculate histogram difference (Earth Mover's Distance approximation)
            p_cdf = torch.cumsum(p_hist, dim=0)
            t_cdf = torch.cumsum(t_hist, dim=0)
            
            loss += torch.mean((p_cdf - t_cdf).abs())
            
        return loss / pred.size(0)
    
    # OptimizerS
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/100)
    
    best_val_loss = float('inf')
    window = create_window(11, 1).to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for mer_images, jwst_images in train_loader:
            mer_images = mer_images.to(device)
            jwst_images = jwst_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(mer_images)
            
            # Multi-component loss
            # Start with basic pixel losses
            loss = 0.5 * l1_criterion(outputs, jwst_images) + 0.2 * mse_criterion(outputs, jwst_images)
            
            # Add SSIM loss for structural similarity
            loss += 0.2 * ssim_loss(outputs, jwst_images)
            
            # Add high-frequency loss for details
            loss += 0.3 * high_frequency_loss(outputs, jwst_images)
            
            # Add histogram loss for matching overall intensity distribution
            loss += 0.1 * histogram_loss(outputs, jwst_images)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
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
                
                # Use combined loss for validation too
                loss = 0.5 * l1_criterion(outputs, jwst_images) + 0.2 * mse_criterion(outputs, jwst_images)
                loss += 0.2 * ssim_loss(outputs, jwst_images)
                loss += 0.3 * high_frequency_loss(outputs, jwst_images)
                
                val_loss += loss.item()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f'LR: {scheduler.get_last_lr()[0]:.8f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_enhanced_jwst_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.6f}")
    
    return model