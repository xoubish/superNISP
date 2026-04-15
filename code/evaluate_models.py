"""
evaluate_models.py

Generates image comparison grids and ellipticity/shear plots
for the RRDB and Diffusion super-resolution models, using the
same validation set for both.
"""

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
RRDB_MODEL_PATH      = "best_sweep_model_tbra2akh.pth"
DIFFUSION_MODEL_PATH = "final_model.pth"
EUCLID_PATH          = "/global/cfs/cdirs/m2218/eramey16/SR_data/euclid_NIR_cosmos_41px_Y.npy"
JWST_PATH            = "/global/cfs/cdirs/m2218/eramey16/SR_data/jwst_cosmos_205px_F115W.npy"

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT      = 0.2
SEED           = 42
N_EVAL         = 2000
N_DISPLAY      = 8

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


# ============================================================
# DIFFUSION VALIDATION WRAPPER
# Applies center crop + asinh normalization to the shared
# RRDB validation split, so both models see the same galaxies
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
    The RRDB split (torch.manual_seed + random_split) is the source of truth.
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
    print(f"RRDB model loaded from {RRDB_MODEL_PATH}")
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

    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_image_grid(results, title):
    """Plot a grid of (LR, Bilinear, SR, HR) quadruplets."""
    n     = len(results['display_lr'])
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols * 4, figsize=(ncols * 6, nrows * 6))
    if nrows == 1:
        axs = axs.reshape(1, -1)

    labels       = ['NISP Y', 'Bilinear', 'Super-Res', 'NIRCam F115W']
    display_keys = ['display_lr', 'display_bilinear', 'display_sr', 'display_hr']

    for i in range(n):
        row  = i // ncols
        base = (i % ncols) * 4
        for j, (key, label) in enumerate(zip(display_keys, labels)):
            ax = axs[row, base + j]
            ax.imshow(results[key][i], origin='lower', cmap='gray')
            ax.axis('off')
            ax.set_title(label, fontsize=8)

    for i in range(n, nrows * ncols):
        row  = i // ncols
        base = (i % ncols) * 4
        for j in range(4):
            axs[row, base + j].axis('off')

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_ellipticity_comparison(rrdb_results, diff_results):
    """
    2x4 grid: rows = RRDB / Diffusion, columns = e1, e2, g1, g2.
    Each panel shows LR (blue), bilinear (green), and SR (red) vs HR.
    """
    metrics = [
        ('e1', r'$e_1$'),
        ('e2', r'$e_2$'),
        ('g1', r'$g_1$'),
        ('g2', r'$g_2$'),
    ]
    model_results = [
        (rrdb_results, 'RRDB'),
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

            ax.scatter(lr[mask_lr], hr[mask_lr], s=6, alpha=0.3,
                       color='steelblue', label='NISP (LR)',       rasterized=True)
            ax.scatter(bl[mask_bl], hr[mask_bl], s=6, alpha=0.3,
                       color='seagreen',  label='Bilinear',         rasterized=True)
            ax.scatter(sr[mask_sr], hr[mask_sr], s=6, alpha=0.3,
                       color='tomato',    label=f'{model_name} SR', rasterized=True)
            ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8)

            ax.set_xlabel(f'{label} (input / SR)', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{label} (NIRCam)', fontsize=10)
            ax.set_title(f'{model_name} — {label}', fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, markerscale=2)

    fig.suptitle('Ellipticity & Shear: RRDB vs Diffusion', fontsize=14)
    plt.tight_layout()
    return fig


def plot_shear_residuals(rrdb_results, diff_results):
    """
    Residual histograms (SR − HR) for RRDB, Diffusion, and bilinear baseline.
    """
    metrics = [('e1', r'$e_1$'), ('e2', r'$e_2$'), ('g1', r'$g_1$'), ('g2', r'$g_2$')]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    for col, (key, label) in enumerate(metrics):
        ax = axs[col]
        for results, model_name, color in [
            (rrdb_results, 'RRDB',      'tomato'),
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
        ax.set_xlabel(f'{label} residual (SR − HR)', fontsize=10)
        ax.set_ylabel('Density' if col == 0 else '')
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle('Shear Residuals: SR − NIRCam', fontsize=14)
    plt.tight_layout()
    return fig


def plot_pixel_metrics(rrdb_results, diff_results, bins=60):
    """
    2x4 grid of histograms: rows = RRDB / Diffusion,
    columns = L1, L2, PSNR, SSIM. Each panel shows SR vs bilinear.
    """
    metrics = [
        ('l1',   'L1 (MAE)',  False),
        ('l2',   'L2 (RMSE)', False),
        ('psnr', 'PSNR (dB)', True),
        ('ssim', 'SSIM',      True),
    ]
    model_pairs = [
        (rrdb_results, 'RRDB'),
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
            ax.set_xlabel(f'{metric_label}  ({better})', fontsize=9)
            ax.set_ylabel('Density' if col == 0 else '', fontsize=9)
            ax.set_title(f'{model_name} — {metric_label}', fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle('Pixel-level Metrics vs NIRCam: SR vs Bilinear Baseline', fontsize=14)
    plt.tight_layout()
    return fig


def plot_pixel_metrics_summary(rrdb_results, diff_results):
    """
    Summary bar chart comparing mean metrics for bilinear, RRDB, and Diffusion.
    """
    metrics = [
        ('l1',   'L1 (MAE)',  False),
        ('l2',   'L2 (RMSE)', False),
        ('psnr', 'PSNR (dB)', True),
        ('ssim', 'SSIM',      True),
    ]
    labels = ['Bilinear', 'RRDB SR', 'Diffusion SR']
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
        bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                      color=colors, alpha=0.8, ecolor='black', error_kw={'lw': 1})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('↓ better' if not higher_is_better else '↑ better', fontsize=9)

    fig.suptitle('Mean Pixel Metrics vs NIRCam Ground Truth', fontsize=14)
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

    print(f"\n=== RRDB inference on {N_EVAL} samples ===")
    rrdb_results = run_inference(rrdb_model, val_dataset, model_type='rrdb')

    print(f"\n=== Diffusion inference on {N_EVAL} samples ===")
    diff_results = run_inference(diffusion_model, diff_val_dataset, model_type='diffusion')

    print("\n=== Generating plots ===")
    fig1 = plot_image_grid(rrdb_results, title="RRDB Super-Resolution")
    fig1.savefig("../figs/rrdb_image_grid.pdf", bbox_inches='tight', dpi=150)

    fig2 = plot_image_grid(diff_results, title="Diffusion Super-Resolution")
    fig2.savefig("../figs/diffusion_image_grid.pdf", bbox_inches='tight', dpi=150)

    fig3 = plot_ellipticity_comparison(rrdb_results, diff_results)
    fig3.savefig("../figs/ellipticity_comparison.pdf", bbox_inches='tight', dpi=150)

    fig4 = plot_shear_residuals(rrdb_results, diff_results)
    fig4.savefig("../figs/shear_residuals.pdf", bbox_inches='tight', dpi=150)

    fig5 = plot_pixel_metrics(rrdb_results, diff_results)
    fig5.savefig("../figs/pixel_metrics_histograms.pdf", bbox_inches='tight', dpi=150)

    fig6 = plot_pixel_metrics_summary(rrdb_results, diff_results)
    fig6.savefig("../figs/pixel_metrics_summary.pdf", bbox_inches='tight', dpi=150)

    print("\nDone! Saved:")
    print("  rrdb_image_grid.pdf, diffusion_image_grid.pdf")
    print("  ellipticity_comparison.pdf, shear_residuals.pdf")
    print("  pixel_metrics_histograms.pdf, pixel_metrics_summary.pdf")
    plt.show()