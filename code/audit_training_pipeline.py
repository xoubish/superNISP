"""Reproducible, read-only diagnostics of the current training inputs and code.

Run from any directory: python code/audit_training_pipeline.py
Writes JSON and a contact sheet under paper/diagnostics; never trains or logs
to W&B. Array statistics use a seeded subset, not a final evaluation sample.
Selected definitions are executed from their AST to avoid training-module
import side effects. This does not test those modules' full import paths.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative_path, names, extra=None):
    """Execute only the named definitions, preserving their source code."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [n for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f"Missing requested definitions in {path}")
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math)
    namespace.update(extra or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(zip(['p05', 'p50', 'p95'], np.quantile(values, [.05, .5, .95]).tolist())) if values.size else None


def array_summary(images):
    finite = np.isfinite(images)
    complete = finite.all(axis=(1, 2))
    valid = images[complete]
    return {
        'sample_n': len(images),
        'nonfinite_pixels': int((~finite).sum()),
        'images_with_nonfinite': int((~complete).sum()),
        'constant_images': int((np.std(valid, axis=(1, 2)) == 0).sum()),
        'zero_pixel_fraction': float((images == 0).mean()),
        'images_with_any_zero': int((images == 0).any(axis=(1, 2)).sum()),
        'per_image_zero_fraction': quantiles((images == 0).mean(axis=(1, 2))),
        'per_image_std_finite_only': quantiles(np.std(valid, axis=(1, 2))),
    }


def audit_arrays(n, seed, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    normalizer_cls = load_module('code/diffusion/dataset.py', 'audit_diff_dataset').AsinhNormalizer
    results = {}
    fig, axes = plt.subplots(6, 4, figsize=(11, 15))
    for col, suffix in enumerate(['', '_20251124']):
        paths = [ROOT / f'data/euclid_NIR_cosmos_41px_Y{suffix}.npy',
                 ROOT / f'data/jwst_cosmos_205px_F115W{suffix}.npy']
        arrays = [np.load(p, mmap_mode='r', allow_pickle=False) for p in paths]
        if len(arrays[0]) != len(arrays[1]):
            raise ValueError('Unpaired array lengths')
        ids = np.sort(np.random.default_rng(seed).choice(len(arrays[0]), min(n, len(arrays[0])), replace=False))
        lr, hr = [np.asarray(a[ids]) for a in arrays]
        lr_zero = lr == 0
        # Block-all-zero is only a diagnostic proxy for a resampled mask.
        hr_zero = (hr == 0).reshape(len(hr), 41, 5, 41, 5).all(axis=(2, 4))
        item = {
            'paths': [str(p.relative_to(ROOT)) for p in paths],
            'shapes': [list(a.shape) for a in arrays],
            'file_bytes': [p.stat().st_size for p in paths],
            'sample_indices': ids.tolist(),
            'nisp': array_summary(lr), 'jwst': array_summary(hr),
            'mask_proxy_disagreement_fraction_same_orientation': float((lr_zero != hr_zero).mean()),
            'mask_proxy_disagreement_fraction_flipped_jwst': float((lr_zero != hr_zero[:, :, ::-1]).mean()),
        }
        # Fit on a small diagnostic subset. NOT the original run's parameters.
        finite_ids = np.flatnonzero(np.isfinite(hr).all(axis=(1, 2)))
        norm = normalizer_cls(alpha=3.0)
        np.random.seed(seed)
        norm.fit(hr[finite_ids[:64]], sample_frac=1.0)
        normalized = np.stack([norm.normalize(x[50:155, 50:155])[0] for x in hr[finite_ids]])
        item['diagnostic_asinh_not_original_training_normalizer'] = {
            'fit_images': min(64, len(finite_ids)),
            'sigma_sky': norm.sigma_sky, 's99': norm.s99,
            'evaluated_images': len(finite_ids),
            'fraction_pixels_outside_minus1_plus1': float((np.abs(normalized) > 1).mean()),
            'fraction_images_exceeding_plus1': float((normalized > 1).any(axis=(1, 2)).mean()),
            'peak_per_image': quantiles(normalized.max(axis=(1, 2))),
        }
        results[suffix or 'undated'] = item
        for row in range(6):
            # Fixed seeded examples; individual stretches are for inspection only.
            j = min(row * max(1, len(ids) // 6), len(ids) - 1)
            for k, img in enumerate([lr[j], hr[j]]):
                ax = axes[row, 2 * col + k]
                vmin, vmax = np.nanpercentile(img, [5, 99.7])
                ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f'{suffix or "undated"} #{ids[j]} {"NISP" if k == 0 else "JWST"}', fontsize=9)
                ax.axis('off')
    fig.suptitle('Seeded input pairs; separate display stretches; not model predictions')
    fig.tight_layout()
    fig.savefig(out / 'training_input_examples.png', dpi=140)
    plt.close(fig)
    return results


def audit_models():
    names = ['ResidualDenseBlock', 'DetailEnhancementModule', 'EuclidToJWSTSuperResolution']
    results = {}
    for path, blocks, features, checkpoint in [
        ('code/claude_model_NIR_2.py', 6, 64, 'code/best_sweep_model_tbra2akh.pth'),
        ('code/claudemodel.py', 6, 48, 'code/final_best.pth'),
    ]:
        cls = definitions(path, names)['EuclidToJWSTSuperResolution']
        state = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True)
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        blocks = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('rrdb_blocks.'))
        features = state['conv_first.weight'].shape[0]
        model = cls(num_rrdb=blocks, features=features)
        model.load_state_dict(state, strict=True)
        shapes = {}
        for size in [21, 25, 41]:
            with torch.no_grad():
                prediction = model(torch.randn(1, 1, size, size))
            shapes[str(size)] = {'output': list(prediction.shape), 'finite': bool(torch.isfinite(prediction).all())}
        model.zero_grad()
        loss = model(torch.randn(1, 1, 21, 21)).square().mean()
        loss.backward()
        results[path] = {
            'checkpoint': checkpoint, 'strict_load': True,
            'blocks': blocks, 'features': features,
            'parameters': sum(p.numel() for p in model.parameters()),
            'shape_checks': shapes,
            'finite_present_gradients': all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
        }
    sr3 = load_module('code/diffusion/model_sr3.py', 'audit_sr3')
    unet = sr3.SR3UNet(base_channels=32)
    model = sr3.SR3SuperResolution(unet, timesteps=1000)
    pred, noise, x0 = model(torch.randn(1, 1, 21, 21), torch.randn(1, 1, 105, 105), torch.tensor([500]))
    ((pred - noise).square().mean() + x0.square().mean()).backward()
    results['code/diffusion/model_sr3.py'] = {
        'parameters': sum(p.numel() for p in model.parameters()),
        'output_shape': list(x0.shape), 'finite_output': bool(torch.isfinite(x0).all()),
        'finite_present_gradients': all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
        'note': 'Random initialization; execution check only, not performance.',
    }
    return results


def audit_versions(n, seed):
    """Match every LR stamp by exact bytes, then verify sampled HR partners.

    This establishes overlap of image products, not astronomical object IDs.
    """
    arrays = [np.load(ROOT / f'data/euclid_NIR_cosmos_41px_Y{s}.npy', mmap_mode='r')
              for s in ['', '_20251124']]
    lookups = []
    for data in arrays:
        lookup = {}
        for i, img in enumerate(data):
            key = hashlib.sha256(img.tobytes()).hexdigest()
            lookup.setdefault(key, []).append(i)
        lookups.append(lookup)
    shared = lookups[0].keys() & lookups[1].keys()
    pairs = [(lookups[0][key][0], lookups[1][key][0]) for key in sorted(shared)
             if len(lookups[0][key]) == len(lookups[1][key]) == 1]
    chosen = np.random.default_rng(seed).choice(len(pairs), min(n, len(pairs)), replace=False)
    hrs = [np.load(ROOT / f'data/jwst_cosmos_205px_F115W{s}.npy', mmap_mode='r')
           for s in ['', '_20251124']]
    matches = sum(np.array_equal(hrs[0][pairs[j][0]], hrs[1][pairs[j][1]], equal_nan=True) for j in chosen)
    return {
        'undated_lr_total': len(arrays[0]), 'dated_lr_total': len(arrays[1]),
        'undated_unique_lr_hashes': len(lookups[0]), 'dated_unique_lr_hashes': len(lookups[1]),
        'dated_lr_images_matching_undated': sum(len(lookups[1][key]) for key in shared),
        'unambiguous_shared_lr_images': len(pairs),
        'sampled_hr_partners_checked': len(chosen), 'sampled_hr_partners_exactly_equal': int(matches),
        'note': 'Do not treat the dated array as an independent test field or split.',
    }


def audit_geometry():
    files = {p.name: p for p in (ROOT / 'data').rglob('*')
             if p.is_file() and (p.name.endswith('.fits') or p.name.endswith('.fits.gz'))}
    records = []
    for cat_path in sorted((ROOT / 'catalog').glob('matched*.csv')):
        cat = pd.read_csv(cat_path)
        for name in cat.nisp_image.unique()[:4]:
            path = files.get(Path(name).name)
            if path is None:
                records.append({'catalog': cat_path.name, 'missing_basename': Path(name).name})
                continue
            hdu = 1 if 'W-STK' in path.name else 0
            header = fits.getheader(path, hdu)
            wcs = WCS(header)
            matrix = wcs.pixel_scale_matrix
            records.append({
                'catalog': cat_path.name, 'file': str(path.relative_to(ROOT)), 'hdu': hdu,
                'pixel_scale_arcsec': (3600 * proj_plane_pixel_scales(wcs)).tolist(),
                'orientation_deg': float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
                'determinant_sign': float(np.sign(np.linalg.det(matrix))),
            })
    return records


def audit_clipping():
    import os
    ns = definitions('code/clipping.py', ['clip_images', 'rotate_jwst'], dict(
        os=os, fits=fits, WCS=WCS, SkyCoord=SkyCoord, Cutout2D=Cutout2D,
        ndimage=ndimage, tqdm=tqdm, FRAC_ZERO=.25))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [51, 51]
        wcs.wcs.crval = [150, 2]
        wcs.wcs.cdelt = [-.3 / 3600, .3 / 3600]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        hdr = wcs.to_header()
        img = np.ones((101, 101), dtype=np.float32)
        mask = np.zeros_like(img, dtype=np.int16)
        mask[50, 49] = 1  # deliberately asymmetric
        fits.writeto(tmp / 'jwst.fits', img, hdr)
        fits.HDUList([fits.PrimaryHDU(img, hdr), fits.ImageHDU(mask, hdr)]).writeto(tmp / 'nisp.fits')
        cat = pd.DataFrame([dict(ra=150., dec=2., jwst_image=str(tmp / 'jwst.fits'), nisp_image=str(tmp / 'nisp.fits'))])
        common = dict(size_jwst=5, size_nisp=5, pad=0, rot_jwst=0)
        clips = ns['clip_images'](cat, mask=True, mirror_nisp=True, **common)
        result = {
            'same_grid_fixture': True,
            'masked_nisp_locations': np.argwhere(clips[0][2].data == 0).tolist(),
            'masked_jwst_locations': np.argwhere(clips[0][1].data == 0).tolist(),
            'mask_locations_agree': bool(np.array_equal(clips[0][2].data == 0, clips[0][1].data == 0)),
        }
        try:
            ns['clip_images'](cat, mask=False, mirror_nisp=False, **common)
            result['mask_false_error'] = None
        except Exception as error:
            result['mask_false_error'] = type(error).__name__ + ': ' + str(error)
        try:
            clips = ns['clip_images'](cat, mask=True, mirror_nisp=False, size_jwst=5, size_nisp=5, pad=4, rot_jwst=0)
            result['zero_rotation_requested_size5_pad4_actual_shape'] = list(clips[0][1].data.shape)
        except Exception as error:
            result['zero_rotation_padded_error'] = type(error).__name__ + ': ' + str(error)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/diagnostics')
    args = parser.parse_args()
    if args.samples < 6:
        parser.error('--samples must be at least 6')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    sources = ['code/clipping.py', 'code/claudemodel.py', 'code/claude_model_NIR_2.py',
               'code/diffusion/model_sr3.py', 'code/diffusion/dataset.py', 'code/diffusion/train_sr3.py',
               'code/audit_training_pipeline.py']
    report = {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
        'seed': args.seed, 'requested_sample_n': args.samples,
    }
    report['catalogs'] = {}
    for path in sorted((ROOT / 'catalog').glob('matched*.csv')):
        cat = pd.read_csv(path)
        report['catalogs'][path.name] = {
            'rows': len(cat), 'unique_ids': int(cat.id_classic.nunique()),
            'jwst_tiles': int(cat.jwst_image.nunique()), 'nisp_tiles': int(cat.nisp_image.nunique()),
        }
    print('Checking sample arrays...', flush=True)
    report['arrays'] = audit_arrays(args.samples, args.seed, args.output)
    print('Checking exact overlap between array versions...', flush=True)
    report['array_version_overlap'] = audit_versions(args.samples, args.seed)
    print('Checking source FITS pixel geometry...', flush=True)
    report['source_geometry_examples'] = audit_geometry()
    print('Checking model loading, forward and backward passes...', flush=True)
    report['models'] = audit_models()
    print('Checking extraction with synthetic FITS images...', flush=True)
    report['synthetic_clipping'] = audit_clipping()
    target = args.output / 'training_pipeline_audit.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {target}', flush=True)


if __name__ == '__main__':
    main()
