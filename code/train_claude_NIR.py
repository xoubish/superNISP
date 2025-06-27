import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import cv2
from scipy import ndimage
import os
import argparse
from tqdm import tqdm
from claude_model_NIR import EuclidJWSTSuperResolution, AstronomicalLoss

class AstronomicalDataset(Dataset):
    """Enhanced dataset class with multiple normalization options - FIXED VERSION"""
    def __init__(self, euclid_images, jwst_images, transform=None, augment=True, 
                 normalization='adaptive_hist', clip_percentile=99.5):
        self.euclid_images = euclid_images  # Shape: (N, 41, 41)
        self.jwst_images = jwst_images      # Shape: (N, 205, 205)
        self.transform = transform
        self.augment = augment
        self.normalization = normalization
        self.clip_percentile = clip_percentile
        
        print(f"Dataset initialized with {len(self.euclid_images)} image pairs")
        print(f"Normalization method: {normalization}")
        
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
    
    def percentile_normalization(self, image, lower_percentile=1, upper_percentile=99):
        """
        Simple percentile-based normalization
        """
        p_low, p_high = np.percentile(image, [lower_percentile, upper_percentile])
        
        if p_high > p_low:
            image_norm = (image - p_low) / (p_high - p_low)
            image_norm = np.clip(image_norm, 0, 1) * 2 - 1  # Scale to [-1, 1]
        else:
            image_norm = np.zeros_like(image)
            
        return image_norm.astype(np.float32)
    
    def z_score_normalization(self, image):
        """
        Standard z-score normalization with robust statistics
        """
        # Use robust statistics
        img_mean = np.mean(image)
        img_std = np.std(image)
        
        if img_std > 0:
            image_norm = (image - img_mean) / img_std
            # Soft clipping to prevent extreme values
            image_norm = np.tanh(image_norm / 3.0) * 3.0
        else:
            image_norm = image - img_mean
            
        return image_norm.astype(np.float32)
    
    def normalize_image(self, image):
        """Apply the selected normalization method"""
        if self.normalization == 'adaptive_hist':
            return self.adaptive_histogram_normalization(image)
        elif self.normalization == 'flux_preserving':
            return self.flux_preserving_normalization(image)
        elif self.normalization == 'percentile':
            return self.percentile_normalization(image)
        elif self.normalization == 'z_score':
            return self.z_score_normalization(image)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
    
    def __len__(self):
        return len(self.euclid_images)
    
    def __getitem__(self, idx):
        euclid_img = self.euclid_images[idx].astype(np.float32)
        jwst_img = self.jwst_images[idx].astype(np.float32)
        
        # Apply normalization
        euclid_img = self.normalize_image(euclid_img)
        jwst_img = self.normalize_image(jwst_img)
        
        # Data augmentation for astronomical images
        if self.augment:
            # Random rotation (90, 180, 270 degrees)
            if np.random.random() > 0.5:
                k = np.random.choice([1, 2, 3])
                euclid_img = np.rot90(euclid_img, k).copy()  # FIXED: Added .copy()
                jwst_img = np.rot90(jwst_img, k).copy()      # FIXED: Added .copy()
            
            # Random flips
            if np.random.random() > 0.5:
                euclid_img = np.fliplr(euclid_img).copy()    # FIXED: Added .copy()
                jwst_img = np.fliplr(jwst_img).copy()        # FIXED: Added .copy()
            if np.random.random() > 0.5:
                euclid_img = np.flipud(euclid_img).copy()    # FIXED: Added .copy()
                jwst_img = np.flipud(jwst_img).copy()        # FIXED: Added .copy()
            
            # Small random brightness/contrast adjustments
            if np.random.random() > 0.7:
                # Brightness
                brightness_factor = np.random.uniform(0.9, 1.1)
                euclid_img = euclid_img * brightness_factor
                jwst_img = jwst_img * brightness_factor
                
                # Contrast
                contrast_factor = np.random.uniform(0.95, 1.05)
                euclid_mean = np.mean(euclid_img)
                jwst_mean = np.mean(jwst_img)
                euclid_img = (euclid_img - euclid_mean) * contrast_factor + euclid_mean
                jwst_img = (jwst_img - jwst_mean) * contrast_factor + jwst_mean
        
        # Ensure arrays are contiguous before converting to tensors
        euclid_img = np.ascontiguousarray(euclid_img)  # FIXED: Ensure contiguous
        jwst_img = np.ascontiguousarray(jwst_img)      # FIXED: Ensure contiguous
        
        # Add channel dimension
        euclid_img = euclid_img[np.newaxis, ...]  # (1, 41, 41)
        jwst_img = jwst_img[np.newaxis, ...]      # (1, 205, 205)
        
        return torch.from_numpy(euclid_img), torch.from_numpy(jwst_img)

# [Include all the model classes from the previous code here - ImprovedSpatialTransformer, 
#  EnhancedResidualBlock, MultiScaleFeatureExtractor, EuclidJWSTSuperResolution, AstronomicalLoss]

def create_data_loaders(euclid_images, jwst_images, train_split=0.8, val_split=0.1, 
                       batch_size=16, normalization='adaptive_hist', num_workers=4):
    """
    Create train/validation/test data loaders
    """
    n_samples = len(euclid_images)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val
    
    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Create random split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create datasets
    train_dataset = AstronomicalDataset(
        euclid_images[train_indices], 
        jwst_images[train_indices], 
        augment=True,
        normalization=normalization
    )
    
    val_dataset = AstronomicalDataset(
        euclid_images[val_indices], 
        jwst_images[val_indices], 
        augment=False,
        normalization=normalization
    )
    
    test_dataset = AstronomicalDataset(
        euclid_images[test_indices], 
        jwst_images[test_indices], 
        augment=False,
        normalization=normalization
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader

def save_sample_images(model, val_loader, device, save_dir='sample_outputs', num_samples=5):
    """Save sample outputs for visual inspection"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, (euclid_imgs, jwst_imgs) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            euclid_imgs = euclid_imgs.to(device)
            jwst_imgs = jwst_imgs.to(device)
            
            pred_imgs, translation = model(euclid_imgs)
            
            # Take first image from batch
            euclid_np = euclid_imgs[0, 0].cpu().numpy()
            jwst_np = jwst_imgs[0, 0].cpu().numpy()
            pred_np = pred_imgs[0, 0].cpu().numpy()
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(euclid_np, cmap='viridis')
            axes[0].set_title(f'Euclid Input (41x41)')
            axes[0].axis('off')
            
            axes[1].imshow(pred_np, cmap='viridis')
            axes[1].set_title(f'Predicted (205x205)\nTranslation: ({translation[0,0]:.2f}, {translation[0,1]:.2f})')
            axes[1].axis('off')
            
            axes[2].imshow(jwst_np, cmap='viridis')
            axes[2].set_title('JWST Target (205x205)')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{i:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Sample images saved to {save_dir}/")

def train_model(model, train_loader, val_loader, num_epochs=150, lr=2e-4, 
                save_dir='checkpoints', device=None):
    """Enhanced training function with comprehensive logging - FIXED VERSION"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # FIXED: Use parameter IDs instead of tensor comparison
    stn_param_ids = set(id(p) for p in model.stn.parameters())
    
    # Create parameter groups
    stn_params = []
    other_params = []
    
    for p in model.parameters():
        if id(p) in stn_param_ids:
            stn_params.append(p)
        else:
            other_params.append(p)
    
    # Optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': stn_params, 'lr': lr * 0.1},  # Lower LR for STN
        {'params': other_params, 'lr': lr}
    ], weight_decay=1e-4)
    
    # Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
    )
    
    criterion = AstronomicalLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"STN parameters: {len(stn_params)}")
    print(f"Other parameters: {len(other_params)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {'l1_loss': 0, 'gradient_loss': 0, 'translation_loss': 0}
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (euclid_imgs, jwst_imgs) in enumerate(train_pbar):
            euclid_imgs = euclid_imgs.to(device, non_blocking=True)
            jwst_imgs = jwst_imgs.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            pred_imgs, translation = model(euclid_imgs)
            loss, loss_dict = criterion(pred_imgs, jwst_imgs, translation)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_metrics:
                if key in loss_dict:
                    train_metrics[key] += loss_dict[key]
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Trans': f'{translation.abs().mean().item():.3f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'l1_loss': 0, 'gradient_loss': 0, 'translation_loss': 0}
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for euclid_imgs, jwst_imgs in val_pbar:
                euclid_imgs = euclid_imgs.to(device, non_blocking=True)
                jwst_imgs = jwst_imgs.to(device, non_blocking=True)
                
                pred_imgs, translation = model(euclid_imgs)
                loss, loss_dict = criterion(pred_imgs, jwst_imgs, translation)
                
                val_loss += loss.item()
                for key in val_metrics:
                    if key in loss_dict:
                        val_metrics[key] += loss_dict[key]
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}')
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_sample_images(model, val_loader, device, 
                             save_dir=f'{save_dir}/samples_epoch_{epoch+1}')
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, f'{save_dir}/best_model.pth')
            
            print(f'✓ New best model saved with validation loss: {best_val_loss:.6f}')
        else:
            patience_counter += 1
            
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return train_losses, val_losses

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train Euclid-JWST Super-Resolution Model')
    parser.add_argument('--euclid_data', type=str, required=True, 
                       help='Path to Euclid images (.npy file)')
    parser.add_argument('--jwst_data', type=str, required=True,
                       help='Path to JWST images (.npy file)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--normalization', type=str, default='adaptive_hist',
                       choices=['adaptive_hist', 'flux_preserving', 'percentile', 'z_score'],
                       help='Normalization method')
    parser.add_argument('--save_dir', type=str, default='training_output',
                       help='Directory to save outputs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading data...")
    euclid_images = np.load(args.euclid_data)
    jwst_images = np.load(args.jwst_data)
    
    print(f"Euclid images shape: {euclid_images.shape}")
    print(f"JWST images shape: {jwst_images.shape}")
    
    # Validate data shapes
    assert euclid_images.shape[1:] == (41, 41), f"Expected Euclid shape (N, 41, 41), got {euclid_images.shape}"
    assert jwst_images.shape[1:] == (205, 205), f"Expected JWST shape (N, 205, 205), got {jwst_images.shape}"
    assert len(euclid_images) == len(jwst_images), "Number of Euclid and JWST images must match"
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        euclid_images, jwst_images,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        normalization=args.normalization,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = EuclidJWSTSuperResolution(
        scale_factor=5, 
        num_residual_blocks=20, 
        num_features=64
    )
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save configuration
    config = {
        'model_params': {
            'scale_factor': 5,
            'num_residual_blocks': 20,
            'num_features': 64
        },
        'training_params': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'normalization': args.normalization,
            'train_split': args.train_split,
            'val_split': args.val_split
        },
        'data_info': {
            'total_samples': len(euclid_images),
            'euclid_shape': euclid_images.shape,
            'jwst_shape': jwst_images.shape
        }
    }
    
    import json
    with open(f'{args.save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )
    
    print("Training completed!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()