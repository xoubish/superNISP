import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_hdf5_path, hr_hdf5_path=None, transform=None, inference_mode=False, split="train", sample_fraction=1.0):
        """
        Dataset that loads LR images and optionally HR images.

        :param lr_hdf5_path: Path to Low-Resolution HDF5 file
        :param hr_hdf5_path: Path to High-Resolution HDF5 file (optional)
        :param transform: Optional torchvision transforms
        :param inference_mode: If True, returns only LR images (for super-res inference)
        :param split: "train" or "test" - determines which dataset to load
        :param sample_fraction: Fraction of the dataset to use (e.g., 0.1 = 10% of data)
        """
        self.lr_hdf5_path = lr_hdf5_path
        self.hr_hdf5_path = hr_hdf5_path  # Can be None
        self.transform = transform
        self.inference_mode = inference_mode  # If True, load only LR
        self.split = split  # Either "train" or "test"

        # Load keys from LR file
        with h5py.File(self.lr_hdf5_path, 'r') as lr_hdf5:
            self.keys = list(lr_hdf5[f"{self.split}_keys"][:])  # Use "train_keys" or "test_keys"

        # ✅ Subsample the dataset
        if sample_fraction < 1.0:
            num_samples = int(len(self.keys) * sample_fraction)
            self.keys = np.random.choice(self.keys, num_samples, replace=False)

        # Check if HR exists
        self.has_hr = hr_hdf5_path is not None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = str(int(self.keys[idx]))  # Ensure key is string format

        # Open HDF5 files inside __getitem__ (for multiprocessing)
        with h5py.File(self.lr_hdf5_path, 'r', swmr=True) as lr_hdf5:
            lr_image = lr_hdf5[f"{self.split}_img"][idx, ...]

        # Use torch.from_numpy() for faster conversion
        lr_image = torch.from_numpy(lr_image).float()

        # If in inference mode OR no HR data, return only LR image
        if self.inference_mode or not self.has_hr:
            return lr_image

        # Load HR image only if available
        with h5py.File(self.hr_hdf5_path, 'r', swmr=True) as hr_hdf5:
            hr_image = hr_hdf5[f"{self.split}_img"][idx, ...]

        hr_image = torch.from_numpy(hr_image).float()  # Faster conversion

        return lr_image, hr_image  # Training case (LR, HR)
