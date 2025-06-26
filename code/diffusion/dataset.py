import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_hdf5_path, hr_hdf5_path=None, transform=None, inference_mode=False, split="train", sample_fraction=1.0):
        self.lr_hdf5_path = lr_hdf5_path
        self.hr_hdf5_path = hr_hdf5_path
        self.transform = transform
        self.inference_mode = inference_mode
        self.split = split
        self.sample_fraction = sample_fraction

        # Defer actual HDF5 file open to __getitem__, but load keys now
        with h5py.File(self.lr_hdf5_path, 'r') as f:
            self.keys = list(f[f"{split}_keys"][:])
        if sample_fraction < 1.0:
            num_samples = int(len(self.keys) * sample_fraction)
            self.keys = np.random.choice(self.keys, num_samples, replace=False)

        self.has_hr = hr_hdf5_path is not None
        self.lr_file = None
        self.hr_file = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Lazily open files per worker (only once per process)
        if self.lr_file is None:
            self.lr_file = h5py.File(self.lr_hdf5_path, 'r', swmr=True)
        if self.hr_file is None and self.has_hr and not self.inference_mode:
            self.hr_file = h5py.File(self.hr_hdf5_path, 'r', swmr=True)

        key_idx = idx  # Your keys are shuffled indices, not HDF5 keys
        lr_image = torch.from_numpy(self.lr_file[f"{self.split}_img"][key_idx]).float()

        if self.inference_mode or not self.has_hr:
            return lr_image

        hr_image = torch.from_numpy(self.hr_file[f"{self.split}_img"][key_idx]).float()
        return lr_image, hr_image
