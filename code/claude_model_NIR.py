import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImprovedSpatialTransformer(nn.Module):
    """Enhanced STN for sub-pixel alignment with astronomical data"""
    def __init__(self, input_channels=1, max_translation=3.0):
        super(ImprovedSpatialTransformer, self).__init__()
        self.max_translation = max_translation
        
        # Localization network optimized for small astronomical features
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Regression head with dropout for better generalization
        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # tx, ty
        )
        
        # Initialize to identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract transformation parameters
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        translation = self.fc_loc(xs)
        
        # Constrain translation to reasonable range (few pixels)
        translation = torch.tanh(translation) * self.max_translation
        
        # Create affine transformation matrix
        theta = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = translation / (x.size(-1) / 2.0)  # Normalize for grid_sample
        
        # Apply spatial transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, mode='bilinear', 
                                    padding_mode='reflection', align_corners=False)
        
        return x_transformed, translation

class EnhancedResidualBlock(nn.Module):
    """Improved residual block with better feature extraction"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(EnhancedResidualBlock, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply channel attention
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        return self.relu(out)

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for better detail preservation"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2)
        )
        
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.relu(self.fusion(out))
        
        return out

class EuclidJWSTSuperResolution(nn.Module):
    def __init__(self, scale_factor=5, num_residual_blocks=20, num_features=64):
        super(EuclidJWSTSuperResolution, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Spatial transformer for alignment
        self.stn = ImprovedSpatialTransformer(input_channels=1, max_translation=3.0)
        
        # Multi-scale initial feature extraction
        self.initial_conv = MultiScaleFeatureExtractor(1, num_features)
        
        # Dense residual blocks with varying dilations for multi-scale features
        self.residual_blocks = nn.ModuleList()
        dilations = [1, 1, 2, 1, 2, 1, 1, 2, 1, 2] * (num_residual_blocks // 10 + 1)
        
        for i in range(num_residual_blocks):
            self.residual_blocks.append(
                EnhancedResidualBlock(num_features, dilation=dilations[i])
            )
        
        # Feature fusion with 1x1 conv
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling for better quality
        self.upsample1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 2x upsampling: 41x41 -> 82x82
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 2x upsampling: 82x82 -> 164x164
            nn.ReLU(inplace=True)
        )
        
        # Final refinement and size adjustment
        self.refinement = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Final output layer
        self.final_conv = nn.Conv2d(num_features // 2, 1, 3, padding=1)
        
    def forward(self, x):
        # Spatial alignment
        x_aligned, translation = self.stn(x)
        
        # Initial feature extraction
        features = self.initial_conv(x_aligned)
        residual_features = features.clone()
        
        # Deep feature extraction
        for res_block in self.residual_blocks:
            features = res_block(features)
        
        # Global residual connection
        features = self.feature_fusion(features) + residual_features
        
        # Progressive upsampling
        features = self.upsample1(features)  # 41x41 -> 82x82
        features = self.upsample2(features)  # 82x82 -> 164x164
        
        # Refinement
        features = self.refinement(features)
        
        # Final output
        output = self.final_conv(features)
        
        # Resize to exact target size (164x164 -> 205x205)
        output = F.interpolate(output, size=(205, 205), mode='bilinear', align_corners=False)
        
        return output, translation

class AstronomicalLoss(nn.Module):
    """Loss function tailored for astronomical super-resolution"""
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.1, delta=0.05):
        super(AstronomicalLoss, self).__init__()
        self.alpha = alpha    # L1 reconstruction loss
        self.beta = beta      # High-frequency detail loss
        self.gamma = gamma    # Perceptual loss (gradient-based)
        self.delta = delta    # Translation regularization
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target, translation):
        # Main reconstruction loss
        l1_loss = self.l1_loss(pred, target)
        
        # High-frequency detail preservation
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        gradient_loss = (self.l1_loss(pred_grad_x, target_grad_x) + 
                        self.l1_loss(pred_grad_y, target_grad_y))
        
        # Second-order gradients for fine details
        pred_grad_xx = pred_grad_x[:, :, :, 1:] - pred_grad_x[:, :, :, :-1]
        pred_grad_yy = pred_grad_y[:, :, 1:, :] - pred_grad_y[:, :, :-1, :]
        target_grad_xx = target_grad_x[:, :, :, 1:] - target_grad_x[:, :, :, :-1]
        target_grad_yy = target_grad_y[:, :, 1:, :] - target_grad_y[:, :, :-1, :]
        
        second_order_loss = (self.l1_loss(pred_grad_xx, target_grad_xx) + 
                           self.l1_loss(pred_grad_yy, target_grad_yy))
        
        # Translation regularization (prefer minimal shifts)
        translation_loss = torch.mean(torch.sum(translation**2, dim=1))
        
        total_loss = (self.alpha * l1_loss + 
                     self.beta * gradient_loss + 
                     self.gamma * second_order_loss + 
                     self.delta * translation_loss)
        
        return total_loss, {
            'l1_loss': l1_loss.item(),
            'gradient_loss': gradient_loss.item(),
            'second_order_loss': second_order_loss.item(),
            'translation_loss': translation_loss.item(),
            'total_loss': total_loss.item()
        }