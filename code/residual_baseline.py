"""Residual baseline isolated from legacy loaders, sweeps, and imports."""
import torch
from torch import nn
from torch.nn import functional as F


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

class EuclidToJWSTSuperResolution(nn.Module):
    def __init__(self, num_rrdb=8, features=64, head="legacy"):
        super(EuclidToJWSTSuperResolution, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, features, kernel_size=3, padding=1)
        
        # Single residual dense blocks (historical attribute name retained)
        self.rrdb_blocks = nn.ModuleList([ResidualDenseBlock(features) for _ in range(num_rrdb)])
        
        # Trunk convolution after dense blocks
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
        
        if head not in {"legacy", "direct5"}:
            raise ValueError(head)
        self.head = head
        if head == "direct5":
            self.upsampling_blocks = nn.ModuleList([nn.Sequential(
                nn.Conv2d(features, features * 25, 3, padding=1),
                nn.PixelShuffle(5), nn.LeakyReLU(0.2, inplace=True),
            )])

        # Final reconstruction layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Initial feature extraction
        features = self.conv_first(x)
        
        # Single residual dense blocks
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
        if self.head == 'legacy':
            output = F.interpolate(output, size=(205, 205), mode='bilinear', align_corners=False)
        
        return output
