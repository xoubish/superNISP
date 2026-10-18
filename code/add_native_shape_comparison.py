"""Extend frozen validation shapes with native NISP; CPU only, no inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.stats import gaussian_kde
import torch

from analyze_spatial_spectra import ROOT, sha
from cached_shape_comparison import summarize
from robust_shape_measurements import adaptive_moments, HSM_REASONS
from shape_diagnostics import diagnose_moments, REASONS


def density_thresholds(density, probabilities=(.68, .95)):
    """Highest-density thresholds enclosing probability on a uniform grid."""
    ordered = np.sort(np.asarray(density).ravel())[::-1]
    if not np.isfinite(ordered).all() or ordered[-1] < 0 or ordered.sum() <= 0:
        raise ValueError('Density must be finite, nonnegative, and nonempty')
    cumulative = np.cumsum(ordered) / ordered.sum()
    return np.array([ordered[np.searchsorted(cumulative, p)] for p in probabilities])


def native_gains(measurements, common, blocks, methods, repeats=2000):
    """Paired block-bootstrap RMSE reduction; positive means better than native."""
    _, inverse = np.unique(blocks[common], return_inverse=True)
    count = np.bincount(inverse)
    draws = np.random.default_rng(20261017).integers(len(count), size=(repeats, len(count)))
    denominator = count[draws].sum(1)[:, None]
    ref = measurements['target']['e'][common]
    rmses, boots = {}, {}
    for method in methods:
        squared = (measurements[method]['e'][common] - ref)**2
        values = np.column_stack([squared.mean(1), squared])
        sums = np.stack([values[inverse == i].sum(0) for i in range(len(count))])
        rmses[method] = np.sqrt(values.mean(0))
        boots[method] = np.sqrt(sums[draws].sum(1) / denominator)
    return {m: dict(order=['combined', 'e1', 'e2'], rmse=rmses[m].tolist(),
                   reduction_percent=(100 * (1 - rmses[m] / rmses['native'])).tolist(),
                   ci95_percent=np.percentile(100 * (1 - boots[m] / boots['native']),
                                             [2.5, 97.5], axis=0).T.tolist())
            for m in methods}


def main(args):
    torch.set_num_threads(2)
    output = args.output
    bundle = ROOT / 'paper_figures/inputs/cached_shapes_v02.json'
    arrays_path = bundle.with_suffix('.npz')
    if bundle.exists() or arrays_path.exists():
        raise FileExistsError('Use a new version; existing results are immutable')
    old_path = ROOT / 'paper_figures/inputs/cached_shapes_v01.json'
    old = json.loads(old_path.read_text())
    old_arrays_path = ROOT / old['arrays_path']
    assert sha(old_arrays_path) == old['arrays_sha256']
    bench_path = ROOT / 'paper_figures/inputs/model_benchmark_v03.json'
    bench = json.loads(bench_path.read_text())
    assert sha(bench_path) == old['inputs'][str(bench_path.relative_to(ROOT))]
    for name, digest in bench['data_hashes'].items():
        assert sha(args.data / name) == digest, name
    for name in ['shape_diagnostics.py', 'robust_shape_measurements.py', 'cached_shape_comparison.py']:
        assert sha(ROOT / 'code' / name) == old['source_hashes'][name], name
    with np.load(old_arrays_path) as f:
        arrays = {k: f[k] for k in f.files}
    ids, blocks = arrays['object_ids'], arrays['blocks']
    assert ids.tolist() == old['object_ids'] == bench['object_ids']
    rows = np.asarray(bench['rows'])
    methods = ['native', *old['methods']]
    contract = dict(local_only=True, split='validation', version='v02',
        population='All 2118 frozen validation objects attempted. Per-estimator intersection of successful JWST, native NISP, and all six cached methods; same objects in every model panel. Also report the intersection across both estimators and full-denominator failure counts.',
        fixed=old['fixed'], adaptive=old['adaptive'], summary=old['summary'], bootstrap=old['bootstrap'],
        native='Signed background-subtracted MJy/sr from the original NISP data, central 21x21 native pixels. Pixel step 5 target pixels: fixed Gaussian sigma 2 native pixels and aperture radius 5 native pixels; adaptive fit uses the same angular crop and scaled initial guess/centroid bounds. Native pixel-center quadrature is not identical to resampled quadrature.',
        masks='Native uses its own validity mask; cached JWST and reconstructions retain their original common-valid mask. Each measurement requires its full central aperture valid on its own grid; compare their common successful objects. No source segmentation, PSF matching, smoothing of images, or model inference.',
        contours='Native-versus-JWST joint density, separately for e1 and e2 on the fixed-estimator common sample. Gaussian KDE with Scott bandwidth; 401x401 uniform grid over [-2,2] in both axes. Highest-density contours enclose 68% and 95% of integrated grid probability; same density and levels reused in every column. Display limits [-1,1].',
        gain='100*(1-RMSE_method/RMSE_native), on identical objects; positive means improvement, negative means degradation. Paired spatial-block 95% intervals with 2000 draws, seed 20261017.',
        limitations=old['limitations'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    sources = ['add_native_shape_comparison.py', 'cached_shape_comparison.py',
               'shape_diagnostics.py', 'robust_shape_measurements.py']
    for name in sources:
        shutil.copy2(ROOT / 'code' / name, output / name)
    parts = {case: [] for case in ['fixed', 'adaptive']}
    native_digest = hashlib.sha256()
    with h5py.File(args.data / 'sample.h5', 'r') as f:
        assert np.array_equal(f['validation_indices'][:], rows)
        for start in range(0, len(ids), 32):
            rr = rows[start:start+32]
            assert np.array_equal(f['object_id'][rr], ids[start:start+32])
            image = ((f['nisp'][rr].astype(np.float64) - f['nisp_background'][rr][:, None, None])
                     * (1.e-12 / f['nisp_pixel_sr'][rr][:, None, None]))[:, 10:31, 10:31]
            valid = f['nisp_valid'][rr].astype(bool)[:, 10:31, 10:31]
            native_digest.update(np.ascontiguousarray(image).tobytes())
            native_digest.update(np.ascontiguousarray(valid).tobytes())
            tensor, mask = torch.from_numpy(image[:, None]), torch.from_numpy(valid[:, None])
            parts['fixed'].append({k: v.numpy() for k, v in diagnose_moments(tensor, mask, pixel_step=5.).items()})
            parts['adaptive'].append(adaptive_moments(tensor, mask, pixel_step=5.))
            if start % 256 == 0:
                print(f'Native shapes {min(start+32,len(ids))}/{len(ids)}', flush=True)
    data = {}
    for case in parts:
        for key in parts[case][0]:
            arrays[f'{case}__native__{key}'] = np.concatenate([p[key] for p in parts[case]])
        data[case] = {m: {k.split('__')[2]: v for k, v in arrays.items() if k.startswith(f'{case}__{m}__')}
                      for m in ['target', *methods]}
    common = np.logical_and.reduce([v['good'] for d in data.values() for v in d.values()])
    arrays['global_common'] = common
    summaries, matched, gains = {}, {}, {}
    for case in data:
        reasons = REASONS if case == 'fixed' else HSM_REASONS
        summaries[case] = summarize(data[case], blocks, ids, methods, reasons)
        matched[case] = summarize(data[case], blocks, ids, methods, reasons, selection=common)
        keep = np.isin(ids, summaries[case]['object_ids'])
        gains[case] = native_gains(data[case], keep, blocks, methods)
    keep = np.isin(ids, summaries['fixed']['object_ids'])
    axis = np.linspace(-2, 2, 401)
    xx, yy = np.meshgrid(axis, axis)
    arrays['contour_axis'] = axis
    masses = []
    for component in range(2):
        sample = np.stack([data['fixed'][m]['e'][keep, component] for m in ['target', 'native']])
        kde = gaussian_kde(sample, bw_method='scott')
        density = kde(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        arrays[f'contour_density_e{component+1}'] = density
        arrays[f'contour_thresholds_e{component+1}'] = density_thresholds(density)
        masses.append(float(density.sum() * (axis[1]-axis[0])**2))
        assert abs(masses[-1]-1) < .001
    np.savez_compressed(arrays_path, **arrays)
    report = dict(complete=True, **contract, methods=methods, object_ids=ids.tolist(), blocks=blocks.tolist(),
        summaries=summaries, cross_estimator_common=matched, cross_estimator_objects=int(common.sum()),
        native_comparison=gains, contour_grid_integrals=masses,
        arrays_path=str(arrays_path.relative_to(ROOT)), arrays_sha256=sha(arrays_path),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in [old_path, old_arrays_path, bench_path]},
        data_directory=str(args.data.resolve()), data_metadata_hashes=bench['data_hashes'],
        native_input_sha256=native_digest.hexdigest(), native_input_hash_recipe='Per batch of 32 in benchmark row order: C-contiguous float64 physical 21x21 images then boolean 21x21 validity masks.',
        source_hashes={name: sha(output/name) for name in sources},
        versions={m: importlib.metadata.version(m) for m in ['numpy', 'scipy', 'torch', 'galsim', 'h5py']})
    bundle.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(bundle, output / 'report.json')
    (output/'status.json').write_text(json.dumps(dict(complete=True, local_only=True, bundle_sha256=sha(bundle)), indent=2)+'\n')
    print('Complete:', bundle, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent/'supernisp_nir_y_f115w_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/cached-shapes-native-v02')
    main(parser.parse_args())
