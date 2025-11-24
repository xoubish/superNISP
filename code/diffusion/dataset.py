import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from astropy.stats import sigma_clipped_stats


# ------------------------------------------------------------
# Asinh-based global normalizer
# ------------------------------------------------------------

class AsinhNormalizer:
    """
    Global, invertible, asinh-based normalizer.

    image  ->  (I - bkg)/sigma_sky -> asinh(x/alpha) / s99
    """

    def __init__(self, alpha=3.0):
        self.alpha = alpha
        self.sigma_sky = None
        self.s99 = None

    def fit(self, cutouts, sample_frac=0.2, max_cuts=2000):
        """
        Estimate global sigma_sky and s99 from a list/array of 2D images.

        cutouts: either a 3D ndarray (N, H, W) or a list of 2D ndarrays.
        """
        # Make a list-like interface
        if isinstance(cutouts, np.ndarray) and cutouts.ndim == 3:
            n_total = cutouts.shape[0]
            get_img = lambda i: cutouts[i]
        else:
            # assume list-like
            n_total = len(cutouts)
            get_img = lambda i: cutouts[i]

        if n_total == 0:
            raise ValueError("No cutouts given to fit normalizer.")

        n_use = int(n_total * sample_frac)
        n_use = max(1, min(n_use, max_cuts, n_total))
        indices = np.random.choice(n_total, size=n_use, replace=False)

        pixel_samples = []
        for idx in indices:
            img = np.asarray(get_img(idx), dtype=float)
            _, med, _ = sigma_clipped_stats(img, sigma=3.0)
            img0 = img - med
            pixel_samples.append(img0.ravel())

        pixel_samples = np.concatenate(pixel_samples)

        # Global sky sigma
        _, _, sigma_sky = sigma_clipped_stats(pixel_samples, sigma=3.0)
        self.sigma_sky = float(sigma_sky)

        # Asinh transform of distribution
        X = pixel_samples / self.sigma_sky
        Y = np.arcsinh(X / self.alpha)

        # 99.7th percentile (~3σ in asinh-space)
        self.s99 = float(np.percentile(np.abs(Y), 99.7))

    def normalize(self, image):
        """
        Normalize a single 2D image.

        Returns:
            Z (float32 array), bkg (float)
        """
        if self.sigma_sky is None or self.s99 is None:
            raise RuntimeError("AsinhNormalizer must be fit() before use.")

        img = np.asarray(image, dtype=float)
        _, bkg, _ = sigma_clipped_stats(img, sigma=3.0)
        img0 = img - bkg

        X = img0 / self.sigma_sky
        Y = np.arcsinh(X / self.alpha)
        Z = Y / self.s99

        return Z.astype(np.float32), float(bkg)

    def denormalize(self, Z, bkg):
        """
        Invert normalization. Z is a 2D array in normalized space.
        """
        if self.sigma_sky is None or self.s99 is None:
            raise RuntimeError("AsinhNormalizer must be fit() before use.")

        Y = Z * self.s99
        X = np.sinh(Y) * self.alpha
        img = X * self.sigma_sky + bkg
        return img


# ------------------------------------------------------------
# Center-crop helper
# ------------------------------------------------------------

def _center_crop_tensor(img: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Center crop a (C, H, W) or (H, W) tensor to (C, crop_size, crop_size).
    """
    if crop_size is None:
        return img

    if img.dim() == 2:
        H, W = img.shape
    elif img.dim() == 3:
        C, H, W = img.shape
    else:
        raise ValueError(f"Unexpected tensor dim {img.dim()} for center crop")

    if crop_size > H or crop_size > W:
        raise ValueError(f"crop_size={crop_size} larger than image size {(H, W)}")

    y0 = H // 2 - crop_size // 2
    x0 = W // 2 - crop_size // 2
    y1 = y0 + crop_size
    x1 = x0 + crop_size

    if img.dim() == 2:
        return img[y0:y1, x0:x1]
    else:
        return img[:, y0:y1, x0:x1]


# ------------------------------------------------------------
# Super-resolution Dataset
# ------------------------------------------------------------

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_path, hr_path=None, transform=None,
                 inference_mode=False, split="train", sample_fraction=1.0,
                 lr_crop_size=None, hr_crop_size=None):

        self.lr_path = lr_path
        self.hr_path = hr_path
        self.transform = transform
        self.inference_mode = inference_mode
        self.split = split
        self.sample_fraction = sample_fraction
        self.has_hr = hr_path is not None

        # 21 for NISP, 105 for JWST, for example
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = hr_crop_size if hr_crop_size is not None else lr_crop_size

        # --- DYNAMIC FORMAT CHECK ---
        if lr_path.lower().endswith(('.h5', '.hdf5')):
            self.format = 'HDF5'
        elif lr_path.lower().endswith('.npy'):
            self.format = 'NUMPY'
        else:
            raise ValueError(f"Unsupported file format: {lr_path}. Must be .h5, .hdf5, or .npy")

        print(f"Loading data in {self.format} format for split '{self.split}'...")

        # ----------------------------------------------------
        # HDF5 Loading Logic (Lazy)
        # ----------------------------------------------------
        if self.format == 'HDF5':
            try:
                with h5py.File(self.lr_path, 'r') as f:
                    self.keys = list(f[f"{split}_keys"][:])
            except (KeyError, FileNotFoundError) as e:
                print(f"Error loading HDF5 keys: {e}. Check file path and internal dataset structure.")
                raise e

            if sample_fraction < 1.0:
                num_samples = int(len(self.keys) * sample_fraction)
                self.keys = np.random.choice(self.keys, num_samples, replace=False)

            self.lr_file = None
            self.hr_file = None
            self.length = len(self.keys)

            # eager arrays are not present for HDF5
            self.lr_data = None
            self.hr_data = None

        # ----------------------------------------------------
        # NUMPY Loading Logic (Eager)
        # ----------------------------------------------------
        elif self.format == 'NUMPY':
            self.lr_data = np.load(self.lr_path)

            if self.has_hr and not self.inference_mode:
                self.hr_data = np.load(self.hr_path)
            else:
                self.hr_data = None

            total_len = len(self.lr_data)
            split_idx = int(total_len * 0.8)

            if split == "train":
                indices = np.arange(split_idx)
            elif split == "test":
                indices = np.arange(split_idx, total_len)
            else:
                raise ValueError(f"Unknown split: {split}")

            num_samples = int(len(indices) * sample_fraction)
            self.indices = indices[:num_samples]

            self.length = len(self.indices)

        # ----------------------------------------------------
        # Asinh Normalizers (global per instrument)
        # ----------------------------------------------------
        self.lr_norm = AsinhNormalizer(alpha=3.0)
        self.hr_norm = AsinhNormalizer(alpha=3.0) if self.has_hr else None

        if not inference_mode:
            if self.format == "NUMPY":
                print("Fitting LR normalizer on NUMPY data...")
                self.lr_norm.fit(self.lr_data)

                if self.has_hr:
                    print("Fitting HR normalizer on NUMPY data...")
                    self.hr_norm.fit(self.hr_data)

            elif self.format == "HDF5":
                # For HDF5, sample a subset of images from disk
                with h5py.File(self.lr_path, 'r') as f:
                    imgs = f[f"{self.split}_img"]
                    n_total = imgs.shape[0]
                    n_use = min(max(1, int(n_total * 0.2)), 2000)
                    idxs = np.random.choice(n_total, size=n_use, replace=False)
                    lr_samples = [imgs[i] for i in idxs]
                print("Fitting LR normalizer on HDF5 data...")
                self.lr_norm.fit(lr_samples, sample_frac=1.0)

                if self.has_hr:
                    with h5py.File(self.hr_path, 'r') as f:
                        imgs = f[f"{self.split}_img"]
                        n_total = imgs.shape[0]
                        n_use = min(max(1, int(n_total * 0.2)), 2000)
                        idxs = np.random.choice(n_total, size=n_use, replace=False)
                        hr_samples = [imgs[i] for i in idxs]
                    print("Fitting HR normalizer on HDF5 data...")
                    self.hr_norm.fit(hr_samples, sample_frac=1.0)

    # --------------------------------------------------------

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            - if inference_mode or no HR: lr_image (1, H, W)
            - else: (lr_image, hr_image)
        Both are normalized using the asinh normalizer if it was fit.
        """
        lr_image = None
        hr_image = None

        # ----------------------------------------------------
        # HDF5 Retrieval Logic
        # ----------------------------------------------------
        if self.format == 'HDF5':
            if self.lr_file is None:
                self.lr_file = h5py.File(self.lr_path, 'r', swmr=True)
            if self.hr_file is None and self.has_hr and not self.inference_mode:
                self.hr_file = h5py.File(self.hr_path, 'r', swmr=True)

            key_idx = idx  # using positional index; adjust if you actually index via self.keys
            lr_np = self.lr_file[f"{self.split}_img"][key_idx]
            lr_image = torch.from_numpy(lr_np).float()

            if self.has_hr and not self.inference_mode:
                hr_np = self.hr_file[f"{self.split}_img"][key_idx]
                hr_image = torch.from_numpy(hr_np).float()

        # ----------------------------------------------------
        # NUMPY Retrieval Logic
        # ----------------------------------------------------
        elif self.format == 'NUMPY':
            global_idx = self.indices[idx]
            lr_np = self.lr_data[global_idx]
            lr_image = torch.from_numpy(lr_np).float()

            if self.has_hr and not self.inference_mode:
                hr_np = self.hr_data[global_idx]
                hr_image = torch.from_numpy(hr_np).float()

        # ----------------------------------------------------
        # Ensure channel dimension: (C, H, W)
        # ----------------------------------------------------
        if lr_image.dim() == 2:
            lr_image = lr_image.unsqueeze(0)
        if hr_image is not None and hr_image.dim() == 2:
            hr_image = hr_image.unsqueeze(0)

        # ---- Center crop BEFORE normalization ----
        if self.lr_crop_size is not None:
            lr_image = _center_crop_tensor(lr_image, self.lr_crop_size)
        if hr_image is not None and self.hr_crop_size is not None:
            hr_image = _center_crop_tensor(hr_image, self.hr_crop_size)

        # ---- Asinh normalization (on cropped patches) ----
        if self.lr_norm is not None and self.lr_norm.sigma_sky is not None:
            lr_np_2d = lr_image.squeeze(0).cpu().numpy()
            lr_norm_np, _ = self.lr_norm.normalize(lr_np_2d)
            lr_image = torch.from_numpy(lr_norm_np).unsqueeze(0)

        if hr_image is not None and self.hr_norm is not None and self.hr_norm.sigma_sky is not None:
            hr_np_2d = hr_image.squeeze(0).cpu().numpy()
            hr_norm_np, _ = self.hr_norm.normalize(hr_np_2d)
            hr_image = torch.from_numpy(hr_norm_np).unsqueeze(0)

        # ----------------------------------------------------
        # Optional extra transform
        # ----------------------------------------------------
        if self.transform is not None:
            lr_image = self.transform(lr_image)
            if hr_image is not None and not self.inference_mode and self.has_hr:
                hr_image = self.transform(hr_image)

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------
        if self.inference_mode or not self.has_hr:
            return lr_image
        else:
            return lr_image, hr_image
