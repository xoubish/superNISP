import torch
from torch.utils.data import Dataset
import h5py # Keep h5py for HDF5 support
import numpy as np # Keep numpy for .npy support
import os # Useful for checking file existence

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_path, hr_path=None, transform=None, inference_mode=False, split="train", sample_fraction=1.0):
        
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.transform = transform
        self.inference_mode = inference_mode
        self.split = split
        self.sample_fraction = sample_fraction
        
        self.has_hr = hr_path is not None
        
        # --- DYNAMIC FORMAT CHECK ---
        
        # Determine format based on file extension
        if lr_path.lower().endswith(('.h5', '.hdf5')):
            self.format = 'HDF5'
        elif lr_path.lower().endswith('.npy'):
            self.format = 'NUMPY'
        else:
            raise ValueError(f"Unsupported file format: {lr_path}. Must be .h5, .hdf5, or .npy")
            
        print(f"Loading data in {self.format} format for split '{self.split}'...")

        # --- HDF5 Loading Logic (Lazy) ---
        if self.format == 'HDF5':
            # Defer actual HDF5 file open to __getitem__, but load keys now
            try:
                with h5py.File(self.lr_path, 'r') as f:
                    # Assumes HDF5 structure with keys/indices stored under f"{split}_keys"
                    self.keys = list(f[f"{split}_keys"][:])
            except (KeyError, FileNotFoundError) as e:
                print(f"Error loading HDF5 keys: {e}. Check file path and internal dataset structure.")
                raise e
                
            if sample_fraction < 1.0:
                num_samples = int(len(self.keys) * sample_fraction)
                self.keys = np.random.choice(self.keys, num_samples, replace=False)

            self.lr_file = None # Will be opened lazily
            self.hr_file = None # Will be opened lazily
            self.length = len(self.keys)

        # --- NUMPY Loading Logic (Eager) ---
        elif self.format == 'NUMPY':
            # Load the entire NumPy arrays into memory
            self.lr_data = np.load(self.lr_path)
            
            if self.has_hr and not self.inference_mode:
                self.hr_data = np.load(self.hr_path)
            else:
                self.hr_data = None

            # Determine indices for train/test split (assuming data is pre-shuffled)
            total_len = len(self.lr_data)
            split_idx = int(total_len * 0.8) # Assuming 80/20 split
            
            if split == "train":
                indices = np.arange(split_idx)
            elif split == "test":
                indices = np.arange(split_idx, total_len)
            else:
                raise ValueError(f"Unknown split: {split}")

            # Apply sampling fraction
            num_samples = int(len(indices) * sample_fraction)
            self.indices = indices[:num_samples]
            
            self.length = len(self.indices)

        # Compute global statistics for normalization (if not already computed)
        self.normalize_stats = None
        if not inference_mode and hasattr(self, 'lr_data'):
            # Compute 2nd and 98th percentiles across entire dataset
            lr_flat = self.lr_data.flatten()
            self.lr_p2 = np.percentile(lr_flat, 2)
            self.lr_p98 = np.percentile(lr_flat, 98)
            if self.has_hr:
                hr_flat = self.hr_data.flatten()
                self.hr_p2 = np.percentile(hr_flat, 2)
                self.hr_p98 = np.percentile(hr_flat, 98)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        # --- HDF5 Retrieval Logic ---
        if self.format == 'HDF5':
            # Lazily open files per worker (only once per process)
            if self.lr_file is None:
                self.lr_file = h5py.File(self.lr_path, 'r', swmr=True)
            if self.hr_file is None and self.has_hr and not self.inference_mode:
                self.hr_file = h5py.File(self.hr_path, 'r', swmr=True)

            key_idx = idx # Corresponds to index into the self.keys list
            lr_image = torch.from_numpy(self.lr_file[f"{self.split}_img"][key_idx]).float()

            if self.inference_mode or not self.has_hr:
                # Add channel dim if it's (H, W) -> (1, H, W)
                if lr_image.dim() == 2: lr_image = lr_image.unsqueeze(0)
                return lr_image

            hr_image = torch.from_numpy(self.hr_file[f"{self.split}_img"][key_idx]).float()
        
        # --- NUMPY Retrieval Logic ---
        elif self.format == 'NUMPY':
            global_idx = self.indices[idx]
            
            lr_image_np = self.lr_data[global_idx]
            lr_image = torch.from_numpy(lr_image_np).float()
            
            if self.inference_mode or not self.has_hr:
                # Add channel dim if it's (H, W) -> (1, H, W)
                if lr_image.dim() == 2: lr_image = lr_image.unsqueeze(0)
                return lr_image

            hr_image_np = self.hr_data[global_idx]
            hr_image = torch.from_numpy(hr_image_np).float()
            
        # Ensure channel dimension is present for both formats: (C, H, W)
        if lr_image.dim() == 2:
            lr_image = lr_image.unsqueeze(0)
        if not self.inference_mode and self.has_hr:
            if hr_image.dim() == 2:
                hr_image = hr_image.unsqueeze(0)
            
        # Normalize to [-1, 1] range using percentile-based normalization
        # Now lr_image and hr_image are guaranteed to be (C, H, W)
        lr_p2 = torch.quantile(lr_image, 0.02)
        lr_p98 = torch.quantile(lr_image, 0.98)
        if lr_p98 > lr_p2:
            lr_image = (lr_image - lr_p2) / (lr_p98 - lr_p2) * 2.0 - 1.0
            lr_image = torch.clamp(lr_image, -1.0, 1.0)
        else:
            lr_image = torch.zeros_like(lr_image)
        
        # Same for hr_image
        if not self.inference_mode and self.has_hr:
            hr_p2 = torch.quantile(hr_image, 0.02)
            hr_p98 = torch.quantile(hr_image, 0.98)
            if hr_p98 > hr_p2:
                hr_image = (hr_image - hr_p2) / (hr_p98 - hr_p2) * 2.0 - 1.0
                hr_image = torch.clamp(hr_image, -1.0, 1.0)
            else:
                hr_image = torch.zeros_like(hr_image)
        
        return lr_image, hr_image