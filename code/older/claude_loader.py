from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import torch
from claude_model import EnhancedSuperResolution
import skimage.exposure as exposure
import skimage.util as util

class EnhancedAstroSRDataset(Dataset):
    def __init__(self, mer_path, jwst_path, transform=None, normalize_method='flux_preserving'):
        # Load numpy arrays
        self.mer_data = np.load(mer_path)
        self.jwst_data = np.load(jwst_path)
        
        # Verify shapes
        assert self.mer_data.shape[0] == self.jwst_data.shape[0], "Number of MER and JWST images must match"
        assert self.mer_data.shape[1:] == (41, 41), f"MER images should be 41x41, got {self.mer_data.shape[1:]}"
        assert self.jwst_data.shape[1:] == (69, 69), f"JWST images should be 69x69, got {self.jwst_data.shape[1:]}"
        
        self.transform = transform
        self.normalize_method = normalize_method
        self.num_samples = self.mer_data.shape[0]
    
    def normalize_data(self, mer_img, jwst_img):
        """Apply advanced normalization to both images."""
        
        if self.normalize_method == 'none':
            return mer_img, jwst_img, {'method': 'none'}
            
        elif self.normalize_method == 'adaptive_hist':
            try:
                # Normalize images to [0, 1] range first
                mer_norm = (mer_img - mer_img.min()) / (mer_img.max() - mer_img.min())
                jwst_norm = (jwst_img - jwst_img.min()) / (jwst_img.max() - jwst_img.min())
                
                # Convert to uint8 for histogram equalization
                mer_eq = exposure.equalize_adapthist(
                    util.img_as_ubyte(mer_norm), 
                    clip_limit=0.03
                )
                jwst_eq = exposure.equalize_adapthist(
                    util.img_as_ubyte(jwst_norm), 
                    clip_limit=0.03
                )
                
                # Convert back to float
                mer_eq = mer_eq.astype(np.float32)
                jwst_eq = jwst_eq.astype(np.float32)
                
                return mer_eq, jwst_eq, {
                    'method': 'adaptive_hist',
                    'mer_min': mer_img.min(), 'mer_max': mer_img.max(),
                    'jwst_min': jwst_img.min(), 'jwst_max': jwst_img.max()
                }
            except Exception as e:
                print(f"Adaptive histogram equalization failed: {e}")
                # Fallback to flux preserving
                self.normalize_method = 'flux_preserving'
        
        # Flux preserving normalization (existing implementation)
        if self.normalize_method == 'flux_preserving':
            # Calculate total flux in each image
            mer_flux = np.sum(mer_img)
            jwst_flux = np.sum(jwst_img)
            
            # Scale factor for JWST to match MER's total flux
            scale_factor = mer_flux / (jwst_flux + 1e-10)
            
            # Apply scaling to JWST image
            jwst_scaled = jwst_img * scale_factor
            
            # Normalize both to [0,1] range while preserving their relative scaling
            max_val = max(np.max(mer_img), np.max(jwst_scaled))
            min_val = min(np.min(mer_img), np.min(jwst_scaled))
            
            # Avoid division by zero
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1
                
            mer_norm = (mer_img - min_val) / range_val
            jwst_norm = (jwst_scaled - min_val) / range_val
            
            return mer_norm, jwst_norm, {
                'method': 'flux_preserving',
                'min_val': min_val, 'max_val': max_val,
                'mer_min': np.min(mer_img), 'mer_max': np.max(mer_img),
                'jwst_min': np.min(jwst_img), 'jwst_max': np.max(jwst_img),
                'scale_factor': scale_factor
            }
        
        # Fallback to min-max normalization
        mer_min, mer_max = np.min(mer_img), np.max(mer_img)
        jwst_min, jwst_max = np.min(jwst_img), np.max(jwst_img)
        
        mer_norm = (mer_img - mer_min) / (mer_max - mer_min + 1e-10)
        jwst_norm = (jwst_img - jwst_min) / (jwst_max - jwst_min + 1e-10)
        
        return mer_norm, jwst_norm, {
            'method': 'minmax',
            'mer_min': mer_min, 'mer_max': mer_max,
            'jwst_min': jwst_min, 'jwst_max': jwst_max
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get the image pair at index idx
        mer_img = self.mer_data[idx].astype(np.float32)
        jwst_img = self.jwst_data[idx].astype(np.float32)
        
        # Apply normalization
        mer_norm, jwst_norm, norm_params = self.normalize_data(mer_img, jwst_img)
        
        # Convert to tensors and add channel dimension
        mer_tensor = torch.from_numpy(mer_norm).unsqueeze(0)  # Shape: [1, 41, 41]
        jwst_tensor = torch.from_numpy(jwst_norm).unsqueeze(0)  # Shape: [1, 69, 69]
        
        # Apply transforms if any
        if self.transform:
            # Create a paired transform to ensure both images are transformed the same way
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            mer_tensor = self.transform(mer_tensor)
            
            torch.manual_seed(seed)
            jwst_tensor = self.transform(jwst_tensor)
            
        return mer_tensor, jwst_tensor, norm_params

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import math
import numpy as np

def train_two_stage(mer_path, jwst_path, val_split=0.2, batch_size=8, num_epochs_stage1=100, num_epochs_stage2=100):
    """
    Two-stage training process for enhanced JWST-like super-resolution.
    
    Args:
        mer_path (str): Path to MER training data numpy file
        jwst_path (str): Path to JWST training data numpy file
        val_split (float): Proportion of data to use for validation
        batch_size (int): Batch size for training
        num_epochs_stage1 (int): Number of epochs for initial training
        num_epochs_stage2 (int): Number of epochs for fine-tuning
    
    Returns:
        Trained PyTorch model
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset with adaptive histogram normalization
    full_dataset = EnhancedAstroSRDataset(
        mer_path, 
        jwst_path, 
        normalize_method='adaptive_hist'
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    torch.manual_seed(42)  # For reproducible splits
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20)  # Add rotation for more variety
    ])
    
    # Add transforms only to training set
    train_dataset.dataset.transform = train_transform
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = EnhancedSuperResolution(num_rrdb=8, features=64).to(device)
    
    # Stage 1: Initial Training with Basic Loss Functions
    print("\n=== Stage 1: Initial Training ===")
    stage1_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    stage1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage1_optimizer, 
        T_max=num_epochs_stage1, 
        eta_min=1e-6
    )
    
    # Loss functions
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    best_val_loss_stage1 = float('inf')
    
    for epoch in range(num_epochs_stage1):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for mer_images, jwst_images, _ in train_loader:
            # Move data to device
            mer_images = mer_images.to(device)
            jwst_images = jwst_images.to(device)
            
            # Zero the parameter gradients
            stage1_optimizer.zero_grad()
            
            # Forward pass
            outputs = model(mer_images)
            
            # Compute loss
            loss = 0.7 * l1_criterion(outputs, jwst_images) + 0.3 * mse_criterion(outputs, jwst_images)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            stage1_optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for mer_images, jwst_images, _ in val_loader:
                mer_images = mer_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(mer_images)
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
    ssim_criterion = SSIMLoss()  # You'll need to define this custom SSIM loss
    
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
        
        for mer_images, jwst_images, _ in train_loader:
            # Move data to device
            mer_images = mer_images.to(device)
            jwst_images = jwst_images.to(device)
            
            # Zero the parameter gradients
            stage2_optimizer.zero_grad()
            
            # Forward pass
            outputs = model(mer_images)
            
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
            for mer_images, jwst_images, _ in val_loader:
                mer_images = mer_images.to(device)
                jwst_images = jwst_images.to(device)
                
                outputs = model(mer_images)
                
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
    
    # Return the final trained model
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss for image super-resolution.
    
    This implementation provides a loss function based on the SSIM metric,
    which measures the structural similarity between two images.
    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        """
        Args:
            window_size (int): Size of the gaussian window
            size_average (bool): Whether to average the loss over batch
            val_range (float): Dynamic range of the images (default: None)
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Gaussian window creation
        self.window = self._create_window(window_size)

    def _create_window(self, window_size, channel=1):
        """
        Create a 2D gaussian window for SSIM calculation.
        
        Args:
            window_size (int): Size of the window
            channel (int): Number of channels (default: 1)
        
        Returns:
            torch.Tensor: Gaussian window tensor
        """
        def gaussian(window_size, sigma):
            """Generate a 1D Gaussian kernel."""
            gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                                   for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, val_range=None):
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            img1 (torch.Tensor): First input image
            img2 (torch.Tensor): Second input image
            window (torch.Tensor): Gaussian window
            window_size (int): Size of the window
            channel (int): Number of channels
            val_range (float): Dynamic range of the images
        
        Returns:
            torch.Tensor: SSIM map
        """
        # Check if val_range is provided, otherwise determine dynamically
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1
            
            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        # Compute mean
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variance and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        # SSIM constants
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Return SSIM
        return ssim_map.mean() if self.size_average else ssim_map

    def forward(self, img1, img2):
        """
        Compute SSIM loss
        
        Args:
            img1 (torch.Tensor): First input image
            img2 (torch.Tensor): Second input image
        
        Returns:
            torch.Tensor: SSIM loss value
        """
        # Ensure images are in the correct shape and on the same device
        (_, channel, _, _) = img1.size()
        
        # Create window if not already created or on different device
        if self.window.dtype != img1.dtype or self.window.device != img1.device:
            self.window = self._create_window(self.window_size, channel).to(img1.device).type_as(img1)
        
        # Compute SSIM
        ssim_value = self._ssim(
            img1, img2, 
            window=self.window, 
            window_size=self.window_size, 
            channel=channel, 
            val_range=self.val_range
        )
        
        # Return SSIM loss (1 - SSIM)
        return 1 - ssim_value

# Optional: Perceptual SSIM Loss with Multi-Scale Support
class MultiScaleSSIMLoss(SSIMLoss):
    """
    Multi-scale SSIM loss to capture details at different scales
    """
    def __init__(self, window_sizes=[11, 17, 23]):
        """
        Args:
            window_sizes (list): List of window sizes for multi-scale SSIM
        """
        super().__init__(window_size=window_sizes[0])
        self.window_sizes = window_sizes
    
    def forward(self, img1, img2):
        """
        Compute multi-scale SSIM loss
        
        Args:
            img1 (torch.Tensor): First input image
            img2 (torch.Tensor): Second input image
        
        Returns:
            torch.Tensor: Multi-scale SSIM loss value
        """
        # Compute SSIM at multiple scales
        total_ssim = 0
        for window_size in self.window_sizes:
            # Create a new SSIMLoss for each window size
            scale_ssim_loss = SSIMLoss(window_size=window_size)
            total_ssim += scale_ssim_loss(img1, img2)
        
        # Average the SSIM loss across scales
        return total_ssim / len(self.window_sizes)