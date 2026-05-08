"""
evaluate_models.py

Generates image comparison grids and ellipticity/shear plots
for the ResNet and Diffusion super-resolution models, using the
same validation set for both.
"""
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import galsim
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Import model classes and dataset from your existing files
from claude_model_NIR_2 import EuclidToJWSTDataset, EuclidToJWSTSuperResolution
from diffusion.model_sr3 import SR3UNet, SR3SuperResolution
from diffusion.dataset import AsinhNormalizer   # adjust import path as needed

# ============================================================
# CONFIGURATION — edit these
# ============================================================
RRDB_MODEL_PATH      = "best_model_rrdb.pth"
DIFFUSION_MODEL_PATH = "best_model_diffusion.pth"
EUCLID_PATH          = "/global/cfs/cdirs/m2218/eramey16/SR_data/euclid_NIR_cosmos_41px_Y.npy"
JWST_PATH            = "/global/cfs/cdirs/m2218/eramey16/SR_data/jwst_cosmos_205px_F115W.npy"

RRDB_RESULTS_PATH = "/global/cfs/cdirs/m2218/eramey16/SR_data/results/rrdb_results.npz"
DIFF_RESULTS_PATH = "/global/cfs/cdirs/m2218/eramey16/SR_data/results/diff_results.npz"

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT      = 0.2
SEED           = 42
N_EVAL         = 2000
N_DISPLAY      = 200

RRDB_NUM_RRDB  = 8
RRDB_FEATURES  = 64

DIFF_HIDDEN_DIM  = 32        # base_channels used during training
DIFF_TIMESTEPS   = 1000
LR_CROP_SIZE     = 21
HR_CROP_SIZE     = 105

PIXEL_SCALE_LR = 0.10        # arcsec/pixel — Euclid NISP
PIXEL_SCALE_HR = 0.06        # arcsec/pixel — JWST NIRCam

DIFF_INFERENCE_STEPS = 100
DIFF_INIT_SIGMA      = 1.0

os.makedirs("/global/cfs/cdirs/m2218/eramey16/SR_data/results", exist_ok=True)

FONT_SIZE = 16
plt.rcParams.update({
    'font.size':        FONT_SIZE,
    'axes.titlesize':   FONT_SIZE + 2,
    'axes.labelsize':   FONT_SIZE,
    'xtick.labelsize':  FONT_SIZE - 2,
    'ytick.labelsize':  FONT_SIZE - 2,
    'legend.fontsize':  FONT_SIZE - 6,
    'figure.titlesize': FONT_SIZE + 4,
})

# ============================================================
# DIFFUSION VALIDATION WRAPPER
# Applies center crop + asinh normalization to the shared
# ResNet validation split, so both models see the same galaxies
# ============================================================
class DiffusionValWrapper(Dataset):
    def __init__(self, val_dataset, lr_crop_size=21, hr_crop_size=105):
        self.val_dataset  = val_dataset
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = hr_crop_size

        # Fit asinh normalizers on the full raw numpy arrays
        full_ds = val_dataset.dataset        # unwrap random_split Subset
        lr_all  = full_ds.euclid_data
        hr_all  = full_ds.jwst_data

        self.lr_norm = AsinhNormalizer(alpha=3.0)
        self.hr_norm = AsinhNormalizer(alpha=3.0)
        self.lr_norm.fit(lr_all)
        self.hr_norm.fit(hr_all)

    def __len__(self):
        return len(self.val_dataset)

    def __getitem__(self, idx):
        global_idx = self.val_dataset.indices[idx]
        full_ds    = self.val_dataset.dataset

        lr_np = full_ds.euclid_data[global_idx].astype(np.float32)
        hr_np = full_ds.jwst_data[global_idx].astype(np.float32)

        lr_tensor = torch.from_numpy(lr_np).unsqueeze(0)    # [1,41,41]
        hr_tensor = torch.from_numpy(hr_np).unsqueeze(0)    # [1,205,205]

        lr_tensor = center_crop_tensor(lr_tensor, self.lr_crop_size)
        hr_tensor = center_crop_tensor(hr_tensor, self.hr_crop_size)

        lr_norm_np, _ = self.lr_norm.normalize(lr_tensor.squeeze(0).numpy())
        hr_norm_np, _ = self.hr_norm.normalize(hr_tensor.squeeze(0).numpy())

        return (torch.from_numpy(lr_norm_np).unsqueeze(0),
                torch.from_numpy(hr_norm_np).unsqueeze(0))
    


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_results(results, path):
    """Save inference results dict to a .npz file."""
    np.savez(path, **{k: np.array(v) for k, v in results.items()})
    print(f"  Saved results to {path}")


def load_results(path):
    """Load inference results dict from a .npz file."""
    data = np.load(path, allow_pickle=True)
    results = {}
    for k in data.files:
        v = data[k]
        # Display images are 2D arrays stored in an object array — convert back to list
        if v.dtype == object or k.startswith('display_'):
            results[k] = list(v)
        else:
            results[k] = v.tolist()
    print(f"  Loaded results from {path}")
    return results


def center_crop_tensor(tensor, crop_size):
    """Center crop a CHW tensor to (C, crop_size, crop_size)."""
    _, h, w = tensor.shape
    sh = (h - crop_size) // 2
    sw = (w - crop_size) // 2
    return tensor[:, sh:sh + crop_size, sw:sw + crop_size]


def get_moments(image_np, pixel_scale, bkg_subtract=True):
    """
    Compute ellipticity (e1, e2) and reduced shear (g1, g2)
    via GalSim HSM adaptive moments.
    Returns [e1, e2, g1, g2], or [nan, nan, nan, nan] on failure.
    """
    try:
        arr = np.asarray(image_np, dtype=np.float32)
        if bkg_subtract:
            arr = arr - np.nanmedian(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        img = galsim.ImageF(arr.shape[1], arr.shape[0], scale=pixel_scale)
        img.array[:, :] = arr
        res = galsim.hsm.FindAdaptiveMom(img)
        return [res.observed_e1, res.observed_e2,
                res.observed_shape.g1, res.observed_shape.g2]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]


def compute_pixel_metrics(pred_np, target_np):
    """Compute L1, L2, PSNR, SSIM between pred and target 2D arrays."""
    data_range = target_np.max() - target_np.min()
    return {
        'l1':   np.mean(np.abs(pred_np - target_np)),
        'l2':   np.sqrt(np.mean((pred_np - target_np) ** 2)),
        'psnr': peak_signal_noise_ratio(target_np, pred_np, data_range=data_range),
        'ssim': structural_similarity(target_np, pred_np, data_range=data_range),
    }


def empty_results_dict():
    return {k: [] for k in [
        'display_lr', 'display_sr', 'display_hr', 'display_bilinear',
        'e1_lr', 'e2_lr', 'g1_lr', 'g2_lr',
        'e1_hr', 'e2_hr', 'g1_hr', 'g2_hr',
        'e1_sr', 'e2_sr', 'g1_sr', 'g2_sr',
        'e1_bl', 'e2_bl', 'g1_bl', 'g2_bl',
        'l1_sr', 'l2_sr', 'psnr_sr', 'ssim_sr',
        'l1_bl', 'l2_bl', 'psnr_bl', 'ssim_bl',
    ]}


# ============================================================
# SHARED VALIDATION DATASETS
# ============================================================
def get_val_datasets():
    """
    Returns val datasets for both models using the same underlying indices.
    The ResNet split (torch.manual_seed + random_split) is the source of truth.
    """
    full_dataset = EuclidToJWSTDataset(EUCLID_PATH, JWST_PATH, normalize_method='z_score')
    val_size     = int(VAL_SPLIT * len(full_dataset))
    train_size   = len(full_dataset) - val_size

    torch.manual_seed(SEED)
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    diff_val_dataset = DiffusionValWrapper(
        val_dataset,
        lr_crop_size=LR_CROP_SIZE,
        hr_crop_size=HR_CROP_SIZE,
    )

    print(f"Validation set: {val_size} samples (seed={SEED}) — shared by both models")
    return val_dataset, diff_val_dataset


# ============================================================
# MODEL LOADERS
# ============================================================
def load_rrdb_model():
    model = EuclidToJWSTSuperResolution(num_rrdb=RRDB_NUM_RRDB, features=RRDB_FEATURES)
    model.load_state_dict(torch.load(RRDB_MODEL_PATH, map_location=DEVICE))
    print(f"ResNet model loaded from {RRDB_MODEL_PATH}")
    return model.to(DEVICE).eval()


def load_diffusion_model():
    unet = SR3UNet(
        in_channels=1,
        cond_channels=1,
        base_channels=DIFF_HIDDEN_DIM,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
    )
    model = SR3SuperResolution(
        unet=unet,
        timesteps=DIFF_TIMESTEPS,
        upscale_factor=5,
    )
    model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
    print(f"Diffusion model loaded from {DIFFUSION_MODEL_PATH}")
    return model.to(DEVICE).eval()


# ============================================================
# INFERENCE
# ============================================================
def run_inference(model, val_dataset, model_type='rrdb'):
    """
    Run inference over the validation set and collect shape and pixel metrics.

    Args:
        model:       Loaded PyTorch model, or None for bilinear baseline
        val_dataset: Validation dataset — EuclidToJWSTDataset split for 'rrdb'
                     and 'bilinear', DiffusionValWrapper for 'diffusion'
        model_type:  'rrdb', 'diffusion', or 'bilinear'

    Returns:
        dict containing display images, ellipticity/shear values, and pixel
        metrics for both SR output and bilinear baseline
    """
    if model_type=='rrdb': res_file = RRDB_RESULTS_PATH
    elif model_type=='diffusion': res_file = DIFF_RESULTS_PATH
    else: res_file = None

    if res_file is not None and os.path.exists(res_file):
        return load_results(res_file)

    n       = min(N_EVAL, len(val_dataset))
    results = empty_results_dict()

    for i in range(n):

        # --- Load sample ---
        if model_type == 'diffusion':
            lr_img, hr_img = val_dataset[i]        # [1,21,21], [1,105,105]
        else:
            lr_img, hr_img, _ = val_dataset[i]     # [1,41,41], [1,205,205]

        lr_input = lr_img
        hr_ref   = hr_img

        # --- SR output ---
        if model_type == 'rrdb':
            with torch.no_grad():
                sr_img = model(lr_input.unsqueeze(0).to(DEVICE))
            sr_img = sr_img.detach().cpu().squeeze(0)          # [1,205,205]

        elif model_type == 'diffusion':
            with torch.no_grad():
                sr_img = model.sample(
                    lr_input.unsqueeze(0).to(DEVICE),
                    num_steps=DIFF_INFERENCE_STEPS,
                    deterministic=True,
                    init_sigma=DIFF_INIT_SIGMA,
                )[0].cpu()                                     # [1,105,105]

        elif model_type == 'bilinear':
            sr_img = F.interpolate(
                lr_input.unsqueeze(0),
                size=(hr_ref.shape[-2], hr_ref.shape[-1]),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}. "
                             "Expected 'rrdb', 'diffusion', or 'bilinear'.")

        # --- Bilinear baseline (always computed for comparison) ---
        bl_img = F.interpolate(
            lr_input.unsqueeze(0),
            size=(hr_ref.shape[-2], hr_ref.shape[-1]),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        lr_np = lr_input[0].cpu().numpy()
        hr_np = hr_ref[0].cpu().numpy()
        sr_np = sr_img[0].numpy() if torch.is_tensor(sr_img) else sr_img[0]
        bl_np = bl_img[0].numpy()

        # --- Display grid ---
        if i < N_DISPLAY:
            results['display_lr'].append(lr_np)
            results['display_sr'].append(sr_np)
            results['display_hr'].append(hr_np)
            results['display_bilinear'].append(bl_np)

        # --- Shape metrics ---
        for key_prefix, arr, scale in [
            ('lr', lr_np, PIXEL_SCALE_LR),
            ('hr', hr_np, PIXEL_SCALE_HR),
            ('sr', sr_np, PIXEL_SCALE_HR),
            ('bl', bl_np, PIXEL_SCALE_HR),
        ]:
            e1, e2, g1, g2 = get_moments(arr, pixel_scale=scale)
            results[f'e1_{key_prefix}'].append(e1)
            results[f'e2_{key_prefix}'].append(e2)
            results[f'g1_{key_prefix}'].append(g1)
            results[f'g2_{key_prefix}'].append(g2)

        # --- Pixel metrics vs HR ---
        for key_prefix, arr in [('sr', sr_np), ('bl', bl_np)]:
            m = compute_pixel_metrics(arr, hr_np)
            for metric_key, val in m.items():
                results[f'{metric_key}_{key_prefix}'].append(val)

        if (i + 1) % 200 == 0:
            print(f"  [{model_type}] {i+1}/{n} done")
    
    if res_file is not None: save_results(results, res_file)
    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_image_grid(results, title, ncols=3, nrows=4):
    """
    Plot a compact grid of (NISP, Super-Res, JWST) triplets.
    
    Args:
        ncols: number of triplet columns (default 3)
        nrows: number of rows (default 4)
    """
    n_display = min(len(results['display_lr']), ncols * nrows)
    
    # Each triplet is 3 image panels + 1 spacer column, except after the last
    # Total axes columns: ncols * 3 + (ncols - 1) spacers
    n_ax_cols = ncols * 3 + (ncols - 1)
    
    # Width ratios: image columns get 1, spacer columns get 0.15
    width_ratios = []
    for col in range(ncols):
        width_ratios += [1, 1, 1]
        if col < ncols - 1:
            width_ratios.append(0.15)   # spacer

    fig, axs = plt.subplots(
        nrows, n_ax_cols,
        figsize=(n_ax_cols * 1.2, nrows * 1.3),
        gridspec_kw={
            'width_ratios': width_ratios,
            'wspace': 0.05,
            'hspace': 0.1,
        }
    )

    if nrows == 1:
        axs = axs.reshape(1, -1)

    labels       = ['NISP Y', 'SR', 'NIRCam']
    display_keys = ['display_lr', 'display_sr', 'display_hr']

    for i in range(n_display):
        row      = i // ncols
        triplet  = i  % ncols
        # Offset in axes columns: each triplet takes 3 + 1 spacer (except last)
        base     = triplet * 4

        for j, (key, label) in enumerate(zip(display_keys, labels)):
            ax = axs[row, base + j]
            ax.imshow(results[key][i], origin='lower', cmap='gray',
                      interpolation='nearest')
            ax.axis('off')
            # Labels only on the first row
            if row == 0:
                ax.set_title(label, pad=2)

    # Hide spacer axes and any unused image axes
    for row in range(nrows):
        for col in range(n_ax_cols):
            ax = axs[row, col]
            # Spacer columns
            if col % 4 == 3:
                ax.set_visible(False)
                continue
            # Unused image slots
            triplet = col // 4
            j       = col  % 4
            i       = row * ncols + triplet
            if i >= n_display:
                ax.set_visible(False)

    fig.suptitle(title, y=1.01)
    return fig


def plot_ellipticity_comparison(rrdb_results, diff_results):
    """
    2x4 grid: rows = ResNet / Diffusion, columns = e1, e2, g1, g2.
    Each panel shows LR (blue), bilinear (green), and SR (red) vs HR,
    with lines of best fit for SR and bilinear.
    """
    metrics = [
        ('e1', r'$e_1$'),
        ('e2', r'$e_2$'),
        ('g1', r'$g_1$'),
        ('g2', r'$g_2$'),
    ]
    model_results = [
        (rrdb_results, 'ResNet'),
        (diff_results, 'Diffusion'),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharex='col', sharey='col')

    for row, (results, model_name) in enumerate(model_results):
        for col, (key, label) in enumerate(metrics):
            ax = axs[row, col]

            hr = np.array(results[f'{key}_hr'], dtype=float)
            lr = np.array(results[f'{key}_lr'], dtype=float)
            sr = np.array(results[f'{key}_sr'], dtype=float)
            bl = np.array(results[f'{key}_bl'], dtype=float)

            mask_lr = np.isfinite(lr) & np.isfinite(hr)
            mask_sr = np.isfinite(sr) & np.isfinite(hr)
            mask_bl = np.isfinite(bl) & np.isfinite(hr)

            labels = ['NISP (LR)', 'Bilinear', f'{model_name} SR']
            if col!=0: labels = ["_"+el for el in labels]

            ax.scatter(lr[mask_lr], hr[mask_lr], s=2, alpha=0.3,
                       color='steelblue', label=labels[0],       rasterized=True)
            ax.scatter(bl[mask_bl], hr[mask_bl], s=2, alpha=0.3,
                       color='seagreen',  label=labels[1],         rasterized=True)
            ax.scatter(sr[mask_sr], hr[mask_sr], s=2, alpha=0.3,
                       color='tomato',    label=labels[2], rasterized=True)

            # 1:1 reference line
            ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8)

            # Lines of best fit for SR and bilinear
            x_line = np.linspace(-1, 1, 200)
            for vals, mask, color in [
                (bl, mask_bl, 'seagreen'),
                (sr, mask_sr, 'tomato'),
            ]:
                if mask.sum() > 10:
                    m, c = np.polyfit(vals[mask], hr[mask], 1)
                    ax.plot(x_line, m * x_line + c, color=color, lw=1.5,
                            label=f'm={m:.2f}, c={c:.2f}')

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xlabel(f'{label} (input / SR)')
            if col == 0:
                ax.set_ylabel(f'{label} (NIRCam)')
            ax.set_title(f'{model_name} — {label}')
            ax.legend(markerscale=2)

    fig.suptitle('Ellipticity & Shear Comparison')
    plt.tight_layout()
    return fig


def plot_shear_residuals(rrdb_results, diff_results):
    """
    Residual histograms (SR − HR) for ResNet, Diffusion, and bilinear baseline.
    """
    metrics = [('e1', r'$e_1$'), ('e2', r'$e_2$'), ('g1', r'$g_1$'), ('g2', r'$g_2$')]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    for col, (key, label) in enumerate(metrics):
        ax = axs[col]
        for results, model_name, color in [
            (rrdb_results, 'ResNet',      'tomato'),
            (diff_results, 'Diffusion', 'seagreen'),
        ]:
            sr  = np.array(results[f'{key}_sr'], dtype=float)
            hr  = np.array(results[f'{key}_hr'], dtype=float)
            res = sr - hr
            mask = np.isfinite(res)
            ax.hist(res[mask], bins=60, alpha=0.5, color=color, density=True,
                    label=f'{model_name} (μ={np.nanmean(res):.3f})')

        # Bilinear residual — use rrdb_results since it's the same baseline
        bl  = np.array(rrdb_results[f'{key}_bl'], dtype=float)
        hr  = np.array(rrdb_results[f'{key}_hr'], dtype=float)
        res_bl = bl - hr
        mask_bl = np.isfinite(res_bl)
        ax.hist(res_bl[mask_bl], bins=60, alpha=0.5, color='steelblue', density=True,
                label=f'Bilinear (μ={np.nanmean(res_bl):.3f})')

        ax.axvline(0, color='k', lw=1, ls='--')
        ax.set_xlabel(f'{label} residual (SR − HR)')
        ax.set_ylabel('Density' if col == 0 else '')
        ax.set_title(label)
        ax.legend()

    fig.suptitle('Moment Residuals vs. NIRCam')
    plt.tight_layout()
    return fig


def plot_pixel_metrics(rrdb_results, diff_results, bins=60):
    """
    2x4 grid of histograms: rows = ResNet / Diffusion,
    columns = L1, L2, PSNR, SSIM. Each panel shows SR vs bilinear.
    """
    metrics = [
        ('l1',   'L1 (MAE)',  False),
        ('l2',   'L2 (RMSE)', False),
        ('psnr', 'PSNR (dB)', True),
        ('ssim', 'SSIM',      True),
    ]
    model_pairs = [
        (rrdb_results, 'ResNet'),
        (diff_results, 'Diffusion'),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(18, 8))

    for col, (metric_key, metric_label, higher_is_better) in enumerate(metrics):
        for row, (results, model_name) in enumerate(model_pairs):
            ax = axs[row, col]

            sr_vals = np.array(results[f'{metric_key}_sr'], dtype=float)
            bl_vals = np.array(results[f'{metric_key}_bl'], dtype=float)
            sr_vals = sr_vals[np.isfinite(sr_vals)]
            bl_vals = bl_vals[np.isfinite(bl_vals)]

            ax.hist(bl_vals, bins=bins, alpha=0.5, color='steelblue', density=True,
                    label=f'Bilinear (μ={np.mean(bl_vals):.3f})')
            ax.hist(sr_vals, bins=bins, alpha=0.5, color='tomato',    density=True,
                    label=f'{model_name} SR (μ={np.mean(sr_vals):.3f})')
            ax.axvline(np.mean(bl_vals), color='steelblue', lw=1.5, ls='--')
            ax.axvline(np.mean(sr_vals), color='tomato',    lw=1.5, ls='--')

            better = '← better' if not higher_is_better else 'better →'
            ax.set_xlabel(f'{metric_label}  ({better})')
            ax.set_ylabel('Density' if col == 0 else '')
            ax.set_title(f'{model_name} — {metric_label}')
            ax.legend()

    fig.suptitle('Pixel-level Metrics vs NIRCam')
    plt.tight_layout()
    return fig

def plot_shear_bias(rrdb_results, diff_results):
    """
    Plot SR - HR residuals vs HR value for each shape metric.
    A perfectly unbiased model would scatter horizontally around y=0.
    A slope indicates multiplicative bias (m); a vertical offset indicates
    additive bias (c), following the weak lensing convention e_SR = (1+m)*e_HR + c.

    Layout: 2 rows (ResNet, Diffusion) x 4 cols (e1, e2, g1, g2)
    """
    metrics = [
        ('e1', r'$e_1$'),
        ('e2', r'$e_2$'),
        ('g1', r'$g_1$'),
        ('g2', r'$g_2$'),
    ]
    model_results = [
        (rrdb_results, 'ResNet',      'tomato'),
        (diff_results, 'Diffusion', 'seagreen'),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(18, 8),
                            sharex='col', sharey='col')

    for row, (results, model_name, color) in enumerate(model_results):
        for col, (key, label) in enumerate(metrics):
            ax = axs[row, col]

            hr  = np.array(results[f'{key}_hr'], dtype=float)
            sr  = np.array(results[f'{key}_sr'], dtype=float)
            bl  = np.array(results[f'{key}_bl'], dtype=float)

            res_sr = sr - hr
            res_bl = bl - hr

            mask_sr = np.isfinite(hr) & np.isfinite(res_sr)
            mask_bl = np.isfinite(hr) & np.isfinite(res_bl)

            # --- Scatter points ---
            ax.scatter(hr[mask_bl], res_bl[mask_bl], s=4, alpha=0.2,
                       color='steelblue', label='Bilinear', rasterized=True)
            ax.scatter(hr[mask_sr], res_sr[mask_sr], s=4, alpha=0.2,
                       color=color, label=f'{model_name} SR', rasterized=True)

            # --- Contours for SR ---
            try:
                from scipy.stats import gaussian_kde
                for res_vals, mask, c, lw in [
                    (res_bl, mask_bl, 'steelblue', 1.0),
                    (res_sr, mask_sr, color,        1.5),
                ]:
                    x_c = hr[mask]
                    y_c = res_vals[mask]
                    xy  = np.vstack([x_c, y_c])
                    kde = gaussian_kde(xy, bw_method=0.15)
                    xg  = np.linspace(x_c.min(), x_c.max(), 80)
                    yg  = np.linspace(y_c.min(), y_c.max(), 80)
                    Xg, Yg = np.meshgrid(xg, yg)
                    Zg  = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    ax.contour(Xg, Yg, Zg, levels=4, colors=c, linewidths=lw, alpha=0.7)
            except Exception:
                pass  # skip contours if too few points

            # --- Fit and plot linear bias: residual = m*e_hr + c ---
            for res_vals, mask, c, ls in [
                (res_bl, mask_bl, 'steelblue', '--'),
                (res_sr, mask_sr, color,        '-'),
            ]:
                x_fit = hr[mask]
                y_fit = res_vals[mask]
                if len(x_fit) > 10:
                    coeffs = np.polyfit(x_fit, y_fit, 1)
                    m, c_bias = coeffs
                    x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
                    ax.plot(x_line, np.polyval(coeffs, x_line),
                            color=c, lw=2, ls=ls,
                            label=f'm={m:.3f}, c={c_bias:.3f}')

            # --- Reference lines ---
            ax.axhline(0, color='k', lw=1, ls='--')
            ax.axvline(0, color='k', lw=0.5, ls=':')

            ax.set_xlabel(f'{label} (NIRCam)')
            if col == 0:
                ax.set_ylabel(f'{label} residual (SR − NIRCam)')
            ax.set_title(f'{model_name} — {label}')
            ax.legend(markerscale=2)

    fig.suptitle('Shape Measurement Bias: Residual vs NIRCam Truth\n'
                 r'Slope $m$ = multiplicative bias, intercept $c$ = additive bias')
    plt.tight_layout()
    return fig

def plot_pixel_metrics_summary(rrdb_results, diff_results):
    """
    Summary bar chart comparing mean metrics for bilinear, ResNet, and Diffusion.
    """
    metrics = [
        ('l1',   'L1 (MAE)',  False),
        ('l2',   'L2 (RMSE)', False),
        ('psnr', 'PSNR (dB)', True),
        ('ssim', 'SSIM',      True),
    ]
    labels = ['Bilinear', 'ResNet SR', 'Diffusion SR']
    colors = ['steelblue', 'tomato', 'seagreen']
    x      = np.arange(len(labels))
    width  = 0.6

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    for col, (metric_key, metric_label, higher_is_better) in enumerate(metrics):
        ax = axs[col]
        means = [
            np.nanmean(rrdb_results[f'{metric_key}_bl']),
            np.nanmean(rrdb_results[f'{metric_key}_sr']),
            np.nanmean(diff_results[f'{metric_key}_sr']),
        ]
        stds = [
            np.nanstd(rrdb_results[f'{metric_key}_bl']),
            np.nanstd(rrdb_results[f'{metric_key}_sr']),
            np.nanstd(diff_results[f'{metric_key}_sr']),
        ]
        bars = ax.bar(x, means, width, capsize=4, #yerr=stds,
                      color=colors, alpha=0.8, ecolor='black', error_kw={'lw': 1})
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height()*1.05,
                    f'{mean:.3f}', ha='center', va='bottom', 
                    fontsize=10)
        ax.set_xlabel('↓ better' if not higher_is_better else '↑ better')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], ylim[1]*1.2])

    fig.suptitle('Mean Pixel Metrics vs NIRCam')
    plt.tight_layout()
    return fig


def plot_ellipticity_plane(rrdb_results, diff_results):
    """
    Plot e1 vs. e2 and g1 vs. g2 in the shape plane using hexbin density plots.
    Shows HR, Bilinear, and SR side-by-side for each model and shape metric.
    4 rows x 6 cols: (e1-e2 ResNet, e1-e2 Diffusion, g1-g2 ResNet, g1-g2 Diffusion) x (HR, Bilinear, SR)
    """
    model_results = [
        (rrdb_results, 'ResNet'),
        (diff_results, 'Diffusion'),
    ]
    shape_pairs = [
        ('e1', 'e2', r'$e_1$ vs. $e_2$'),
        ('g1', 'g2', r'$g_1$ vs. $g_2$'),
    ]
    
    # 4 rows (2 shape pairs x 2 models), 3 columns (HR, Bilinear, SR)
    fig, axs = plt.subplots(4, 3, figsize=(15, 18))
    
    row_idx = 0
    for shape_idx, (key1, key2, shape_label) in enumerate(shape_pairs):
        for model_idx, (results, model_name) in enumerate(model_results):
            # Extract data
            hr_x = np.array(results[f'{key1}_hr'], dtype=float)
            hr_y = np.array(results[f'{key2}_hr'], dtype=float)
            sr_x = np.array(results[f'{key1}_sr'], dtype=float)
            sr_y = np.array(results[f'{key2}_sr'], dtype=float)
            bl_x = np.array(results[f'{key1}_bl'], dtype=float)
            bl_y = np.array(results[f'{key2}_bl'], dtype=float)
            
            # Create masks for finite values
            mask_hr = np.isfinite(hr_x) & np.isfinite(hr_y)
            mask_sr = np.isfinite(sr_x) & np.isfinite(sr_y)
            mask_bl = np.isfinite(bl_x) & np.isfinite(bl_y)
            
            # Plot HR, Bilinear, SR in separate columns
            for col_idx, (x_data, y_data, mask, label, cmap) in enumerate([
                (hr_x, hr_y, mask_hr, 'NIRCam (HR)', 'Greys'),
                (bl_x, bl_y, mask_bl, 'Bilinear', 'Blues'),
                (sr_x, sr_y, mask_sr, f'{model_name} SR', 'Reds'),
            ]):
                ax = axs[row_idx, col_idx]
                
                # Hexbin density plot
                hb = ax.hexbin(x_data[mask], y_data[mask], gridsize=30,
                              cmap=cmap, mincnt=1, extent=[-1, 1, -1, 1])
                
                # Add colorbar
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Count', rotation=270, labelpad=15)
                
                # Add reference lines at 0
                ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
                ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
                
                # Add unit circle for reference
                circle = plt.Circle((0, 0), 1.0, fill=False, color='gray',
                                   ls=':', lw=1, alpha=0.5)
                ax.add_patch(circle)
                
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect('equal')
                ax.set_xlabel(f'${key1}$')
                if col_idx == 0:
                    ax.set_ylabel(f'${key2}$')
                ax.set_title(f'{model_name} {shape_label}\n{label}')
                ax.grid(True, alpha=0.3)
            
            row_idx += 1
    
    fig.suptitle('Shape Measurements in the Ellipticity/Shear Plane', y=0.995)
    plt.tight_layout()
    return fig


def plot_ellipticity_plane_residuals(rrdb_results, diff_results):
    """
    Plot residuals (SR - HR) in the e1-e2 and g1-g2 planes.
    Shows where the SR models systematically over/under-predict shapes.
    2x2 grid: rows = e1 vs. e2 / g1 vs. g2, columns = ResNet / Diffusion
    """
    model_results = [
        (rrdb_results, 'ResNet'),
        (diff_results, 'Diffusion'),
    ]
    shape_pairs = [
        ('e1', 'e2', r'$\Delta e_1$ vs. $\Delta e_2$'),
        ('g1', 'g2', r'$\Delta g_1$ vs. $\Delta g_2$'),
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    for row, (key1, key2, title_suffix) in enumerate(shape_pairs):
        for col, (results, model_name) in enumerate(model_results):
            ax = axs[row, col]
            
            # Extract data
            hr_x = np.array(results[f'{key1}_hr'], dtype=float)
            hr_y = np.array(results[f'{key2}_hr'], dtype=float)
            sr_x = np.array(results[f'{key1}_sr'], dtype=float)
            sr_y = np.array(results[f'{key2}_sr'], dtype=float)
            
            # Compute residuals
            res_x = sr_x - hr_x
            res_y = sr_y - hr_y
            
            # Create mask for finite values
            mask = np.isfinite(res_x) & np.isfinite(res_y)
            
            # Hexbin plot for residuals
            hb = ax.hexbin(res_x[mask], res_y[mask], gridsize=30,
                          cmap='RdBu_r', alpha=0.8, mincnt=1,
                          extent=[-0.5, 0.5, -0.5, 0.5], reduce_C_function=np.mean)
            
            # Add colorbar
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Count', rotation=270, labelpad=15)
            
            # Add reference lines at 0
            ax.axhline(0, color='black', lw=1, ls='--', alpha=0.7)
            ax.axvline(0, color='black', lw=1, ls='--', alpha=0.7)
            
            # Calculate and display statistics
            mean_x = np.nanmean(res_x[mask])
            mean_y = np.nanmean(res_y[mask])
            std_x = np.nanstd(res_x[mask])
            std_y = np.nanstd(res_y[mask])
            
            ax.text(0.05, 0.95,
                   f'μ_x={mean_x:.4f}, σ_x={std_x:.4f}\nμ_y={mean_y:.4f}, σ_y={std_y:.4f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_aspect('equal')
            ax.set_xlabel(f'$\Delta {key1}$ (SR − HR)')
            ax.set_ylabel(f'$\Delta {key2}$ (SR − HR)')
            ax.set_title(f'{model_name} — {title_suffix}')
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('Shape Residuals in the Ellipticity/Shear Plane', y=0.995)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Loading validation datasets ===")
    val_dataset, diff_val_dataset = get_val_datasets()

    print("\n=== Loading models ===")
    rrdb_model      = load_rrdb_model()
    diffusion_model = load_diffusion_model()

    print(f"\n=== ResNet inference on {N_EVAL} samples ===")
    rrdb_results = run_inference(rrdb_model, val_dataset, model_type='rrdb')

    print(f"\n=== Diffusion inference on {N_EVAL} samples ===")
    diff_results = run_inference(diffusion_model, diff_val_dataset, model_type='diffusion')

    print("\n=== Generating plots ===")
    fig1 = plot_image_grid(rrdb_results, title="ResNet Super-Resolution", ncols=2, nrows=6)
    fig1.savefig("../figs/rrdb_image_grid.png", bbox_inches='tight', dpi=150)

    fig2 = plot_image_grid(diff_results, title="Diffusion Super-Resolution", ncols=2, nrows=6)
    fig2.savefig("../figs/diffusion_image_grid.png", bbox_inches='tight', dpi=150)

    fig3 = plot_ellipticity_comparison(rrdb_results, diff_results)
    fig3.savefig("../figs/ellipticity_comparison.png", bbox_inches='tight', dpi=150)

    fig4 = plot_shear_residuals(rrdb_results, diff_results)
    fig4.savefig("../figs/shear_residuals.png", bbox_inches='tight', dpi=150)

    fig5 = plot_shear_bias(rrdb_results, diff_results)
    fig5.savefig("../figs/shear_bias.png", bbox_inches='tight', dpi=150)

    fig6 = plot_pixel_metrics(rrdb_results, diff_results)
    fig6.savefig("../figs/pixel_metrics_histograms.png", bbox_inches='tight', dpi=150)

    fig7 = plot_pixel_metrics_summary(rrdb_results, diff_results)
    fig7.savefig("../figs/pixel_metrics_summary.png", bbox_inches='tight', dpi=150)

    fig8 = plot_ellipticity_plane(rrdb_results, diff_results)
    fig8.savefig("../figs/ellipticity_plane.png", bbox_inches='tight', dpi=150)

    fig9 = plot_ellipticity_plane_residuals(rrdb_results, diff_results)
    fig9.savefig("../figs/ellipticity_plane_residuals.png", bbox_inches='tight', dpi=150)

    # ============================================================
    # PERFORMANCE SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY: SR Models vs. Bilinear Baseline")
    print("="*70)
    
    # Compute mean metrics
    bl_ssim_rrdb = np.nanmean(rrdb_results['ssim_bl'])
    bl_ssim_diff = np.nanmean(diff_results['ssim_bl'])
    rrdb_ssim = np.nanmean(rrdb_results['ssim_sr'])
    diff_ssim = np.nanmean(diff_results['ssim_sr'])
    
    bl_psnr_rrdb = np.nanmean(rrdb_results['psnr_bl'])
    bl_psnr_diff = np.nanmean(diff_results['psnr_bl'])
    rrdb_psnr = np.nanmean(rrdb_results['psnr_sr'])
    diff_psnr = np.nanmean(diff_results['psnr_sr'])
    
    bl_l2_rrdb = np.nanmean(rrdb_results['l2_bl'])
    bl_l2_diff = np.nanmean(diff_results['l2_bl'])
    rrdb_l2 = np.nanmean(rrdb_results['l2_sr'])
    diff_l2 = np.nanmean(diff_results['l2_sr'])
    
    # Compute percentage improvements (for SSIM and PSNR, higher is better)
    rrdb_ssim_improvement = ((rrdb_ssim - bl_ssim_rrdb) / bl_ssim_rrdb) * 100
    diff_ssim_improvement = ((diff_ssim - bl_ssim_diff) / bl_ssim_diff) * 100
    
    rrdb_psnr_improvement = ((rrdb_psnr - bl_psnr_rrdb) / bl_psnr_rrdb) * 100
    diff_psnr_improvement = ((diff_psnr - bl_psnr_diff) / bl_psnr_diff) * 100
    
    # For L2, lower is better, so improvement is negative of percentage change
    rrdb_l2_improvement = ((bl_l2_rrdb - rrdb_l2) / bl_l2_rrdb) * 100
    diff_l2_improvement = ((bl_l2_diff - diff_l2) / bl_l2_diff) * 100
    
    print(f"\nBilinear Baseline:")
    print(f"  SSIM: {bl_ssim_rrdb:.4f}  |  PSNR: {bl_psnr_rrdb:.2f} dB  |  L2 (RMSE): {bl_l2_rrdb:.4f}")
    
    print(f"\nResNet SR:")
    print(f"  SSIM: {rrdb_ssim:.4f} ({rrdb_ssim_improvement:+.1f}% vs bilinear)")
    print(f"  PSNR: {rrdb_psnr:.2f} dB ({rrdb_psnr_improvement:+.1f}% vs bilinear)")
    print(f"  L2 (RMSE): {rrdb_l2:.4f} ({rrdb_l2_improvement:+.1f}% vs bilinear)")
    
    print(f"\nDiffusion SR:")
    print(f"  SSIM: {diff_ssim:.4f} ({diff_ssim_improvement:+.1f}% vs bilinear)")
    print(f"  PSNR: {diff_psnr:.2f} dB ({diff_psnr_improvement:+.1f}% vs bilinear)")
    print(f"  L2 (RMSE): {diff_l2:.4f} ({diff_l2_improvement:+.1f}% vs bilinear)")
    
    print("\n" + "="*70)
    print(f"Summary: ResNet achieves {rrdb_ssim_improvement:.1f}% better SSIM than bilinear")
    print(f"         Diffusion achieves {diff_ssim_improvement:.1f}% better SSIM than bilinear")
    print("="*70)

    print("\nDone! Saved:")
    print("  rrdb_image_grid.png, diffusion_image_grid.png")
    print("  ellipticity_comparison.png, shear_residuals.png, shear_bias.png")
    print("  pixel_metrics_histograms.png, pixel_metrics_summary.png")
    print("  ellipticity_plane_resnet.png, ellipticity_plane_diffusion.png")
    print("  ellipticity_plane_residuals.png")
    plt.show()
