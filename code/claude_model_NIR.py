import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import math

# More conservative optimization flags
torch.backends.cudnn.benchmark = True
# Remove TF32 flags as they can cause instability
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

class ResidualDenseBlock(nn.Module):
    """Stable Residual Dense Block"""
    def __init__(self, channels, growth_rate=24):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * 0.2  # Scale residual connection

class StableDetailEnhancement(nn.Module):
    """Stable detail enhancement"""
    def __init__(self, channels):
        super(StableDetailEnhancement, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels//4), kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(1, channels//4), channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        enhanced = self.enhance(x)
        attention = self.attention(enhanced)
        return enhanced * attention

class StableEuclidToJWSTSuperResolution(nn.Module):
    """Stable optimized model"""
    def __init__(self, num_rrdb=6, features=48):
        super(StableEuclidToJWSTSuperResolution, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.rrdb_blocks = nn.ModuleList([
            ResidualDenseBlock(features, growth_rate=24) for _ in range(num_rrdb)
        ])
        
        # Trunk convolution
        self.trunk_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Detail enhancement
        self.detail_enhancer = StableDetailEnhancement(features)
        
        # Conservative upsampling - back to progressive approach
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
        
        # Final reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features//2, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Add output activation to prevent extreme values
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Clamp input to prevent extreme values
        x = torch.clamp(x, 0, 1)
        
        # Initial features
        features = self.conv_first(x)
        
        # Residual processing
        trunk = features
        for block in self.rrdb_blocks:
            trunk = block(trunk)
        
        features = features + self.trunk_conv(trunk) * 0.2  # Scale trunk connection
        
        # Detail enhancement
        features = self.detail_enhancer(features)
        
        # Progressive upsampling
        for upsampling_block in self.upsampling_blocks:
            features = upsampling_block(features)
        
        # Final output
        output = self.final_conv(features)
        
        # Interpolate to exact size and scale output
        output = F.interpolate(output, size=(205, 205), mode='bilinear', align_corners=False)
        output = (output + 1) / 2  # Convert from [-1,1] to [0,1]
        
        return torch.clamp(output, 0, 1)  # Ensure output is in valid range

# Stable Dataset with better normalization
class StableFastEuclidToJWSTDataset(Dataset):
    def __init__(self, euclid_path, jwst_path, augment=False, preload=True):
        print("Loading and preprocessing data...")
        self.euclid_data = np.load(euclid_path).astype(np.float32)
        self.jwst_data = np.load(jwst_path).astype(np.float32)
        
        # Verify shapes
        assert self.euclid_data.shape[0] == self.jwst_data.shape[0]
        assert self.euclid_data.shape[1:] == (25, 25)
        assert self.jwst_data.shape[1:] == (125, 125)

        
        self.augment = augment
        
        # Compute global statistics for stable normalization
        print("Computing global statistics...")
        self.euclid_mean = np.mean(self.euclid_data)
        self.euclid_std = np.std(self.euclid_data) + 1e-8
        self.jwst_mean = np.mean(self.jwst_data)
        self.jwst_std = np.std(self.jwst_data) + 1e-8
        
        # Use percentile-based normalization to handle outliers
        self.euclid_p1 = np.percentile(self.euclid_data, 1)
        self.euclid_p99 = np.percentile(self.euclid_data, 99)
        self.jwst_p1 = np.percentile(self.jwst_data, 1)
        self.jwst_p99 = np.percentile(self.jwst_data, 99)
        
        if preload:
            self.preprocessed_data = []
            for i in range(len(self.euclid_data)):
                euclid_norm, jwst_norm = self.stable_normalize(
                    self.euclid_data[i], self.jwst_data[i]
                )
                self.preprocessed_data.append((
                    torch.from_numpy(euclid_norm).unsqueeze(0),
                    torch.from_numpy(jwst_norm).unsqueeze(0)
                ))
            print(f"Preprocessed {len(self.preprocessed_data)} samples")
        
        self.preload = preload
        
    def stable_normalize(self, euclid_img, jwst_img):
        """Stable normalization using percentiles"""
        # Clip outliers
        euclid_clipped = np.clip(euclid_img, self.euclid_p1, self.euclid_p99)
        jwst_clipped = np.clip(jwst_img, self.jwst_p1, self.jwst_p99)
        
        # Normalize to [0, 1]
        euclid_norm = (euclid_clipped - self.euclid_p1) / (self.euclid_p99 - self.euclid_p1 + 1e-8)
        jwst_norm = (jwst_clipped - self.jwst_p1) / (self.jwst_p99 - self.jwst_p1 + 1e-8)
        
        # Ensure values are in [0, 1]
        euclid_norm = np.clip(euclid_norm, 0, 1)
        jwst_norm = np.clip(jwst_norm, 0, 1)
        
        return euclid_norm, jwst_norm
    
    def apply_augmentation(self, euclid_tensor, jwst_tensor):
        """Conservative augmentation"""
        if not self.augment:
            return euclid_tensor, jwst_tensor
            
        import random
        
        # Only apply one augmentation at a time to reduce complexity
        aug_type = random.randint(0, 2)
        
        if aug_type == 0:  # Horizontal flip
            euclid_tensor = torch.flip(euclid_tensor, [-1])
            jwst_tensor = torch.flip(jwst_tensor, [-1])
        elif aug_type == 1:  # Vertical flip
            euclid_tensor = torch.flip(euclid_tensor, [-2])
            jwst_tensor = torch.flip(jwst_tensor, [-2])
        elif aug_type == 2:  # 90-degree rotation
            k = random.randint(1, 3)
            euclid_tensor = torch.rot90(euclid_tensor, k, [-2, -1])
            jwst_tensor = torch.rot90(jwst_tensor, k, [-2, -1])
            
        return euclid_tensor, jwst_tensor
    
    def __len__(self):
        return len(self.euclid_data)
    
    def __getitem__(self, idx):
        if self.preload:
            euclid_tensor, jwst_tensor = self.preprocessed_data[idx]
        else:
            euclid_img = self.euclid_data[idx]
            jwst_img = self.jwst_data[idx]
            euclid_norm, jwst_norm = self.stable_normalize(euclid_img, jwst_img)
            euclid_tensor = torch.from_numpy(euclid_norm).unsqueeze(0)
            jwst_tensor = torch.from_numpy(jwst_norm).unsqueeze(0)
        
        # Apply augmentation
        euclid_tensor, jwst_tensor = self.apply_augmentation(euclid_tensor, jwst_tensor)
        
        return euclid_tensor, jwst_tensor, {}

# Stable Training Function
def train_two_stage_stable(euclid_path, jwst_path, val_split=0.2, batch_size=12,  # Reduced batch size
                          num_epochs_stage1=30, num_epochs_stage2=20, use_amp=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mixed precision training: {use_amp and device.type == 'cuda'}")

    # Initialize mixed precision scaler with more conservative settings
    scaler = GradScaler(init_scale=2**10) if use_amp and device.type == 'cuda' else None  # Lower initial scale

    # Create stable datasets
    train_dataset_full = StableFastEuclidToJWSTDataset(
        euclid_path, 
        jwst_path, 
        augment=True,
        preload=True
    )
    
    # Calculate split sizes
    dataset_size = len(train_dataset_full)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    
    # Create validation dataset without augmentation
    val_dataset_clean = StableFastEuclidToJWSTDataset(
        euclid_path, 
        jwst_path, 
        augment=False,
        preload=True
    )
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Data loaders with conservative settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced workers
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize stable model
    model = StableEuclidToJWSTSuperResolution(num_rrdb=6, features=48).to(device)
    
    # Don't compile initially - add it later if training is stable
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    #     print("Model compiled for faster execution")
    
    # Stage 1: Conservative Training
    print("\n=== Stage 1: Stable Initial Training ===")
    stage1_optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # Lower learning rate
        weight_decay=1e-4,
        eps=1e-8  # Prevent division by zero
    )
    
    stage1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        stage1_optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Loss functions
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    best_val_loss_stage1 = float('inf')
    
    for epoch in range(num_epochs_stage1):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (euclid_images, jwst_images, _) in enumerate(train_loader):
            # Check for NaN in input data
            if torch.isnan(euclid_images).any() or torch.isnan(jwst_images).any():
                print(f"NaN detected in input data at batch {batch_idx}")
                continue
            
            euclid_images = euclid_images.to(device, non_blocking=True)
            jwst_images = jwst_images.to(device, non_blocking=True)
            
            stage1_optimizer.zero_grad()
            
            # Mixed precision forward pass with error checking
            if use_amp and scaler is not None:
                with autocast():
                    outputs = model(euclid_images)
                    
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        print(f"NaN detected in model output at batch {batch_idx}")
                        continue
                    
                    loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                    
                    # Check for NaN in loss
                    if torch.isnan(loss):
                        print(f"NaN loss detected at batch {batch_idx}")
                        continue
                
                scaler.scale(loss).backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"NaN gradient detected at batch {batch_idx}")
                    stage1_optimizer.zero_grad()
                    continue
                
                scaler.unscale_(stage1_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # More aggressive clipping
                scaler.step(stage1_optimizer)
                scaler.update()
            else:
                outputs = model(euclid_images)
                
                if torch.isnan(outputs).any():
                    print(f"NaN detected in model output at batch {batch_idx}")
                    continue
                
                loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                stage1_optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        if num_batches == 0:
            print("No valid batches processed - stopping training")
            break
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for euclid_images, jwst_images, _ in val_loader:
                if torch.isnan(euclid_images).any() or torch.isnan(jwst_images).any():
                    continue
                    
                euclid_images = euclid_images.to(device, non_blocking=True)
                jwst_images = jwst_images.to(device, non_blocking=True)
                
                if use_amp and scaler is not None:
                    with autocast():
                        outputs = model(euclid_images)
                        if torch.isnan(outputs).any():
                            continue
                        loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                else:
                    outputs = model(euclid_images)
                    if torch.isnan(outputs).any():
                        continue
                    loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
        
        if val_batches == 0:
            print("No valid validation batches - stopping training")
            break
        
        # Compute average losses
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches
        
        # Update scheduler
        stage1_scheduler.step(avg_val_loss)
        
        # Print epoch statistics
        print(f'Stage 1 - Epoch {epoch+1}/{num_epochs_stage1}:')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss_stage1:
            best_val_loss_stage1 = avg_val_loss
            torch.save(model.state_dict(), 'stage1_best_model_stable.pth')
            print(f'New best model saved with val loss: {avg_val_loss:.6f}')
    
    print("Stage 1 completed successfully!")
    
    # Load and return the best model
    model.load_state_dict(torch.load('final_best_model.pth'))
    return model
