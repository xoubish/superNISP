"""Local PSNR/SSIM comparison using frozen Figure 6 predictions; no inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from spatial_spectra import block_summary

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['bilinear', 'bicubic', 'diffusion', 'diffusion_mean3', 'legacy', 'direct5']
REGIONS = ['all', 'source', 'outside']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ssim_map(target, prediction, valid, data_range):
    # Filled invalid pixels never enter retained windows. Source/outside pixels
    # are not zeroed: region-specific aggregation uses fully contained windows.
    target = np.where(valid, target, 0.).astype(np.float64)
    prediction = np.where(valid, prediction, 0.).astype(np.float64)
    return structural_similarity(target, prediction, data_range=data_range,
        win_size=7, gaussian_weights=False, use_sample_covariance=True,
        K1=.01, K2=.03, full=True)[1]


def main():
    output = ROOT/'runs/psnr-ssim-v01'
    bundle = ROOT/'paper_figures/inputs/psnr_ssim_v01.json'
    arrays_path = bundle.with_suffix('.npz')
    if output.exists() or bundle.exists() or arrays_path.exists():
        raise FileExistsError('Preserve existing results; use a fresh version.')
    output.mkdir(parents=True)
    benchmark_path = ROOT/'paper_figures/inputs/model_benchmark_v03.json'
    benchmark = json.loads(benchmark_path.read_text())
    assert benchmark['complete'] and benchmark['split'] == 'validation'
    cache = ROOT/benchmark['prediction_cache']['path']
    mask_path = ROOT/benchmark['source_mask_path']
    assert sha(cache) == benchmark['prediction_cache']['sha256']
    assert sha(mask_path) == benchmark['source_mask_sha256']
    contract = dict(version='v01', local_only=True, split='validation',
        methods=METHODS, regions=REGIONS,
        population='All 2118 cached validation objects, shared methods and masks; no metric-based exclusions.',
        intensity='Original signed, sky-subtracted linear MJy/sr in central 105x105 pixels; no rescaling, clipping, offset, flux fitting or noise subtraction.',
        data_range='For each cutout, max(target[common_valid])-min(target[common_valid]); identical for every method and all three regions. No model-dependent ranges.',
        psnr='20 log10(data_range / per-object regional RMSE); arithmetic mean of per-object dB, not PSNR of mean RMSE. All regional common-valid pixels included.',
        ssim='scikit-image local SSIM: uniform 7x7 window, sample covariance, K1=0.01, K2=0.03. Average only centres whose entire window lies inside the selected region and common-valid support, including at image borders. Then average objects equally. Source/outside pixels are not zeroed. Invalid pixels are filled only for computation and cannot enter retained windows.',
        masks='Frozen JWST 2-sigma source mask including neighbours; outside is its complement within common validity.',
        bootstrap='2000 paired spatial-block bootstrap resamples, seed 20261017, over 18 blocks; equal object weighting; 95% percentile intervals. Paired score differences relative to bilinear saved.',
        limitations=['Range depends on observed reference extrema and is sensitive to bright pixels and noise.',
                    'SSIM depends on window, range and stabilizing constants; signed near-zero sky means affect its luminance term. This is a diagnostic of observed-image similarity, not noise-free morphology.',
                    'SSIM uses region interiors while PSNR uses every regional pixel; those supports differ. Outside-source pixels may contain faint light.',
                    'Reference-derived masks, reference noise, band/PSF differences, validation-selected fixed checkpoints and one training seed remain.'])
    (output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    for name in ['analyze_psnr_ssim.py', 'spatial_spectra.py']:
        shutil.copy2(ROOT/'code'/name, output/name)
    with np.load(mask_path) as saved:
        assert saved['object_ids'].tolist() == benchmark['object_ids']
        common = saved['common']
        source = saved['source'][:, saved['thresholds'].tolist().index(2.)] & common
    masks = [common, source, common & ~source]
    centres = [np.stack([binary_erosion(m, structure=np.ones((7, 7), bool),
                                      border_value=0) for m in region]) for region in masks]
    counts = np.stack([c.sum((1, 2)) for c in centres], axis=1)
    assert (counts > 0).all()
    n = len(benchmark['object_ids'])
    ranges = np.empty(n)
    psnr = np.empty((n, 3, len(METHODS)))
    ssim = np.empty_like(psnr)
    max_rmse_difference = 0.
    primary = next(r for r in benchmark['source_regions'] if r['threshold_sigma'] == 2.)
    old_regions = [benchmark, primary['source'], primary['background']]
    with h5py.File(cache) as saved:
        assert saved.attrs['complete'] and saved['object_ids'][:].tolist() == benchmark['object_ids']
        for i in range(n):
            target = saved['target'][i].astype(np.float64)
            assert np.isfinite(target[common[i]]).all()
            ranges[i] = np.ptp(target[common[i]])
            assert ranges[i] > 0
            for j, method in enumerate(METHODS):
                prediction = saved[method][i].astype(np.float64)
                assert np.isfinite(prediction[common[i]]).all()
                similarity = ssim_map(target, prediction, common[i], ranges[i])
                for r, region in enumerate(masks):
                    valid = region[i]
                    rmse = np.sqrt(np.mean((prediction[valid]-target[valid])**2))
                    old_rmse = old_regions[r]['methods'][method]['rmse_mjy_sr'][i]
                    max_rmse_difference = max(max_rmse_difference, abs(rmse-old_rmse))
                    np.testing.assert_allclose(rmse, old_rmse, rtol=1e-12, atol=1e-14)
                    psnr[i, r, j] = 20*np.log10(ranges[i]/rmse)
                    ssim[i, r, j] = similarity[centres[r][i]].mean()
                    if i in [0, n//2, n-1]:
                        np.testing.assert_allclose(psnr[i, r, j], peak_signal_noise_ratio(
                            target[valid], prediction[valid], data_range=ranges[i]), atol=1e-12)
                if i == 0 and j == 0:
                    np.testing.assert_allclose(ssim_map(target, target, common[i], ranges[i])[centres[0][i]], 1.)
                    # Independent window calculation agrees with selected map centres.
                    for r in range(3):
                        y, x = np.argwhere(centres[r][i])[0]
                        expected = structural_similarity(target[y-3:y+4, x-3:x+4],
                            prediction[y-3:y+4, x-3:x+4], data_range=ranges[i], win_size=7,
                            gaussian_weights=False, use_sample_covariance=True, K1=.01, K2=.03)
                        np.testing.assert_allclose(similarity[y, x], expected, atol=1e-10)
                    changed = prediction.copy()
                    changed[~source[i]] = 3*ranges[i]
                    np.testing.assert_allclose(ssim_map(target, changed, common[i], ranges[i])[centres[1][i]],
                                               similarity[centres[1][i]], atol=1e-9)
            if (i+1) % 300 == 0 or i+1 == n:
                print(f'Measured {i+1}/{n}', flush=True)
    assert np.isfinite(psnr).all() and np.isfinite(ssim).all()
    assert (np.abs(ssim) <= 1+1e-10).all()
    blocks = np.asarray(benchmark['blocks'])
    summary = {}
    for r, region in enumerate(REGIONS):
        summary[region] = dict(objects=n, ssim_centres_min=int(counts[:, r].min()),
            ssim_centres_median=float(np.median(counts[:, r])), methods={})
        for j, method in enumerate(METHODS):
            summary[region]['methods'][method] = {
                metric:dict(**block_summary(values[:, r, j], blocks),
                    difference_vs_bilinear=block_summary(values[:, r, j]-values[:, r, 0], blocks))
                for metric, values in [('psnr_db', psnr), ('ssim', ssim)]}
    np.savez_compressed(arrays_path, psnr_db=psnr, ssim=ssim, data_range=ranges,
                        ssim_centres=counts, object_ids=np.asarray(benchmark['object_ids']), blocks=blocks)
    verification = dict(all_cached_rmse_reproduced=True, max_rmse_difference=max_rmse_difference,
        psnr_checked_against_skimage=True, identity_ssim=True, independent_local_windows=True,
        source_ssim_invariant_to_outside_pixels=True, all_metrics_finite=True)
    result = dict(complete=True, **contract, object_ids=benchmark['object_ids'], blocks=benchmark['blocks'],
        summary=summary, arrays_path=str(arrays_path.relative_to(ROOT)), arrays_sha256=sha(arrays_path),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [benchmark_path, cache, mask_path]},
        source_hashes={p.name:sha(p) for p in output.glob('*.py')}, verification=verification,
        versions={name:importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-image', 'h5py']})
    bundle.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(complete=True, bundle_sha256=sha(bundle)))+'\n')
    print('Complete:', bundle, flush=True)


if __name__ == '__main__':
    main()
