from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import torch

class AstroSRDataset(Dataset):
    def __init__(self, mer_path, jwst_path, transform=None, normalize_method='flux_preserving'):
        """
        Dataset for astronomical super-resolution with numpy arrays.
        
        Args:
            mer_path (str): Path to the MER numpy file (.npy) with shape [N, 41, 41]
            jwst_path (str): Path to the JWST numpy file (.npy) with shape [N, 69, 69]
            transform: Optional transforms to apply to the data
            normalize_method (str): Normalization method to use ('flux_preserving', 
                                    'minmax', 'zscore', or 'none')
        """
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
        
    def __len__(self):
        return self.num_samples
    
    def normalize_data(self, mer_img, jwst_img):
        """Apply the selected normalization method to both images."""
        
        if self.normalize_method == 'none':
            return mer_img, jwst_img
            
        elif self.normalize_method == 'flux_preserving':
            # Calculate total flux in each image
            mer_flux = np.sum(mer_img)
            jwst_flux = np.sum(jwst_img)
            
            # Scale factor for JWST to match MER's total flux
            # This preserves the flux ratio between input and output
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
            
            return mer_norm, jwst_norm
            
        elif self.normalize_method == 'minmax':
            # Simple min-max normalization for each image separately
            def minmax_norm(img):
                min_val = np.min(img)
                max_val = np.max(img)
                range_val = max_val - min_val
                if range_val == 0:
                    return np.zeros_like(img)
                return (img - min_val) / range_val
            
            return minmax_norm(mer_img), minmax_norm(jwst_img)
            
        elif self.normalize_method == 'zscore':
            # Z-score normalization (standardization)
            def zscore_norm(img):
                mean = np.mean(img)
                std = np.std(img)
                if std == 0:
                    return np.zeros_like(img)
                return (img - mean) / std
            
            return zscore_norm(mer_img), zscore_norm(jwst_img)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
    
    def __getitem__(self, idx):
        # Get the image pair at index idx
        mer_img = self.mer_data[idx].astype(np.float32)
        jwst_img = self.jwst_data[idx].astype(np.float32)
        
        # Apply normalization
        mer_norm, jwst_norm = self.normalize_data(mer_img, jwst_img)
        
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
            
        return mer_tensor, jwst_tensor

def create_data_loaders(mer_path, jwst_path, val_split=0.2, batch_size=8, normalize_method='flux_preserving', seed=42):
    """
    Create train and validation data loaders with a specified split.
    
    Args:
        mer_path (str): Path to MER numpy file
        jwst_path (str): Path to JWST numpy file
        val_split (float): Fraction of data to use for validation (0.0-1.0)
        batch_size (int): Batch size for data loaders
        normalize_method (str): Normalization method to use
        seed (int): Random seed for reproducibility
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    # Create the full dataset
    full_dataset = AstroSRDataset(mer_path, jwst_path, normalize_method=normalize_method)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    torch.manual_seed(seed)  # For reproducible splits
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Add transforms only to training set
    train_dataset.dataset.transform = train_transform
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader