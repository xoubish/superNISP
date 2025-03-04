import torch
from torch.utils.data import Dataset
import h5py

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_hdf5_path, hr_hdf5_path=None, transform=None, inference_mode=False, split="train"):
        """
        Dataset that loads LR images and optionally HR images.

        :param lr_hdf5_path: Path to Low-Resolution HDF5 file
        :param hr_hdf5_path: Path to High-Resolution HDF5 file (optional)
        :param transform: Optional torchvision transforms
        :param inference_mode: If True, returns only LR images (for super-res inference)
        :param split: "train" or "test" - determines which dataset to load
        """
        self.lr_hdf5_path = lr_hdf5_path
        self.hr_hdf5_path = hr_hdf5_path  # Can be None
        self.transform = transform
        self.inference_mode = inference_mode  # If True, load only LR
        self.split = split  # Either "train" or "test"

        # Load keys from LR file
        with h5py.File(self.lr_hdf5_path, 'r') as lr_hdf5:
            self.keys = list(lr_hdf5[f"{self.split}_keys"][:])  # Use "train_keys" or "test_keys"

        # Check if HR exists
        self.has_hr = hr_hdf5_path is not None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = str(int(self.keys[idx]))  # Ensure key is string format

        # Open HDF5 files inside __getitem__ (for multiprocessing)
        with h5py.File(self.lr_hdf5_path, 'r') as lr_hdf5:
            lr_image = lr_hdf5[f"{self.split}_img"][idx, ...]  # "train_img" or "test_img"

        # Convert to tensor
        lr_image = torch.tensor(lr_image, dtype=torch.float32)

        # If in inference mode OR no HR data, return only LR image
        if self.inference_mode or not self.has_hr:
            return lr_image

        # Load HR image only if available
        with h5py.File(self.hr_hdf5_path, 'r') as hr_hdf5:
            hr_image = hr_hdf5[f"{self.split}_img"][idx, ...]  # "train_img" or "test_img"
            hr_image = torch.tensor(hr_image, dtype=torch.float32)
            return lr_image, hr_image  # Training case (LR, HR)
