"""Current Figure 7 shape measurements binned by input-only NISP S/N; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from analyze_spatial_spectra import ROOT, sha
from add_native_shape_comparison import native_gains
from cached_shape_comparison import summarize
from robust_shape_measurements import HSM_REASONS
from shape_diagnostics import REASONS


EDGES = np.array([-np.inf, 5., 10., 20., 50., np.inf])
LABELS = ['<5', '5–10', '10–20', '20–50', '≥50']


def bin_indices(values):
    """Left-closed, right-open historical bins; nonfinite values are excluded."""
    values = np.asarray(values)
    return np.where(np.isfinite(values), np.searchsorted(EDGES[1:-1], values, side='right'), -1)


def input_snr(images, rms, valid):
    """Signed Gaussian-weighted native flux / quadrature weighted pixel RMS."""
    axis = (np.arange(41) - 20) * 5
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    weight = np.exp(-(xx**2 + yy**2)/200) * (xx**2 + yy**2 <= 625)
    selected = valid & (weight > 0)
    finite = np.all(~selected | (np.isfinite(images) & np.isfinite(rms) & (rms >= 0)), axis=(1,2))
    signal = (np.where(selected, images, 0)*weight).sum((1,2))
    variance = (np.where(selected, rms, 0)*weight)**2
    denominator = np.sqrt(variance.sum((1,2)))
    return np.divide(signal, denominator, out=np.full(len(images), np.nan), where=finite & (denominator > 0))


def main(args):
    path = ROOT/'paper_figures/inputs/shape_snr_v02.json'
    arrays_path = path.with_suffix('.npz')
    if path.exists() or arrays_path.exists():
        raise FileExistsError('Use a fresh version; existing results are immutable')
    shapes_path = ROOT/'paper_figures/inputs/cached_shapes_v02.json'
    shapes = json.loads(shapes_path.read_text())
    assert shapes['complete'] and shapes['split'] == 'validation'
    shape_arrays_path = ROOT/shapes['arrays_path']
    assert sha(shape_arrays_path) == shapes['arrays_sha256']
    bench_path = ROOT/'paper_figures/inputs/model_benchmark_v03.json'
    assert sha(bench_path) == shapes['inputs'][str(bench_path.relative_to(ROOT))]
    bench = json.loads(bench_path.read_text())
    data_path = Path(shapes['data_directory'])
    for name, digest in shapes['data_metadata_hashes'].items():
        assert sha(data_path/name) == digest, name
    for name in ['add_native_shape_comparison.py', 'cached_shape_comparison.py']:
        assert sha(ROOT/'code'/name) == shapes['source_hashes'][name], name
    with np.load(shape_arrays_path) as f:
        arrays = {k: f[k] for k in f.files}
    ids, blocks = arrays['object_ids'], arrays['blocks']
    rows = np.asarray(bench['rows'])
    assert ids.tolist() == shapes['object_ids'] == bench['object_ids']
    old_path = ROOT/'runs/validation-shapes-full50-best/measurements.npz'
    old_provenance = ROOT/'runs/validation-shapes-full50-best/provenance.json'
    old_meta = json.loads(old_provenance.read_text())
    assert old_meta['training_signature']['data_hashes'] == shapes['data_metadata_hashes']
    with np.load(old_path) as f:
        assert np.array_equal(ids, f['object_ids']) and np.array_equal(rows, f['rows'])
        assert np.array_equal(blocks, f['blocks'])
        old_snr = f['snr_proxy']
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(version='v02', local_only=True, split='validation',
        measurements='Frozen cached_shapes_v02 from current Figures 6/7; original epoch-49 ResNet and ResNet-D5, epoch-35 diffusion. No model inference, retraining or checkpoint changes.',
        population='All 2118 validation fields attempted. Within each S/N bin and estimator, the identical Figure 7 intersection of successful JWST, native NISP and all six cached methods; full-denominator success counts retained. Also report the intersection shared by both estimators.',
        snr='Input-only signed Gaussian-weighted native NISP flux divided by quadrature weighted RMS; sigma 2 native pixels, radius 5, native validity mask. Ignores correlated noise and is not a calibrated detection significance. No reconstructed image or JWST quantity enters the bins.',
        bins=dict(edges=['-inf',5,10,20,50,'inf'], closure='left closed, right open', labels=LABELS,
                  rule='Existing predeclared Figure 8 edges, unchanged before looking at current results; negative finite S/N enters the first bin. Nonfinite values excluded and counted.'),
        metric='100*(1-RMSE_model/RMSE_native), combining both e components; positive means smaller error. Native and model errors use identical objects.',
        uncertainty='2000 paired spatial-block draws, seed 20261017, equal object weights; conditional on the successful sample and this field. Marginal intervals for each bin/method, not a simultaneous trend test or training-seed uncertainty.',
        figures='Compact main figure: fixed and adaptive gain versus S/N, plus common measurement success over all attempted objects in each bin. Same bins, learned methods and gain scale across estimators.',
        limitations=['Observed cross-band shapes include PSF and sampling differences; no shear calibration.',
                     'S/N and the native-baseline shape share input pixels; trends are descriptive, not causal.',
                     'Higher-S/N bins can differ in size, brightness, crowding and redshift; this does not isolate an S/N effect.',
                     'Estimator samples differ; joint-estimator matched results are retained separately.',
                     'Single cached diffusion draw and one selected checkpoint per architecture; no new training-run claim.'])
    (output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    sources = ['analyze_shape_snr.py', 'test_shape_snr.py', 'add_native_shape_comparison.py',
               'cached_shape_comparison.py', 'shape_diagnostics.py', 'robust_shape_measurements.py']
    for name in sources:
        shutil.copy2(ROOT/'code'/name, output/name)
    parts = []
    digest = hashlib.sha256()
    with h5py.File(data_path/'sample.h5', 'r') as f:
        assert np.array_equal(rows, f['validation_indices'][:])
        for start in range(0,len(ids),64):
            rr = rows[start:start+64]
            assert np.array_equal(ids[start:start+64], f['object_id'][rr])
            conversion = (1e-12/f['nisp_pixel_sr'][rr])[:,None,None]
            image = (f['nisp'][rr].astype(float)-f['nisp_background'][rr][:,None,None])*conversion
            rms = f['nisp_rms'][rr].astype(float)*conversion
            valid = f['nisp_valid'][rr].astype(bool)
            for v in [image,rms,valid]:
                digest.update(np.ascontiguousarray(v).tobytes())
            parts.append(input_snr(image,rms,valid))
    snr = np.concatenate(parts)
    bins = bin_indices(snr)
    # The old pipeline used float32 on GPU. Bin membership must be unchanged;
    # old per-galaxy shapes and old aggregate gains are deliberately not reused.
    np.testing.assert_allclose(snr,old_snr,rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(bins,bin_indices(old_snr))
    methods = shapes['methods']
    results, joint_results = {}, {}
    common_masks = {}
    for case in ['fixed','adaptive']:
        measurements = {m:{k.split('__')[2]:v for k,v in arrays.items() if k.startswith(f'{case}__{m}__')}
                        for m in ['target',*methods]}
        good = np.logical_and.reduce([v['good'] for v in measurements.values()])
        assert ids[good].tolist() == shapes['summaries'][case]['object_ids']
        common_masks[case] = good
        results[case],joint_results[case] = [],[]
        for j,label in enumerate(LABELS):
            selected = bins == j
            record = summarize(measurements,blocks,ids,methods,REASONS if case=='fixed' else HSM_REASONS,selection=selected)
            common = good & selected
            assert common.sum() > 0 and len(np.unique(blocks[common])) >= 2
            record.update(label=label, bin_index=j,
                          native_comparison=native_gains(measurements,common,blocks,methods))
            results[case].append(record)
            joint = selected & arrays['global_common']
            joint_results[case].append(dict(label=label,objects=int(joint.sum()),object_ids=ids[joint].tolist(),
                native_comparison=native_gains(measurements,joint,blocks,methods)))
        assert sum(r['objects'] for r in results[case]) == int(np.sum(bins>=0))
        # Recombine squared bin RMSEs: they must reproduce Figure 7 exactly.
        for m in methods:
            recombined=np.sqrt(sum(r['common_successes']*r['native_comparison'][m]['rmse'][0]**2 for r in results[case]) / good.sum())
            np.testing.assert_allclose(recombined,shapes['native_comparison'][case][m]['rmse'][0],rtol=1e-12)
        print(case, 'common by bin:', [r['common_successes'] for r in results[case]], flush=True)
    np.savez_compressed(arrays_path,object_ids=ids,blocks=blocks,rows=rows,snr_proxy=snr,bin_index=bins,
                        **{f'{case}_common':mask for case,mask in common_masks.items()})
    report = dict(complete=True,**contract,methods=methods,objects=len(ids),excluded_snr=int(np.sum(bins<0)),
        snr_range=[float(np.nanmin(snr)),float(np.nanmax(snr))],
        object_ids=ids.tolist(),summaries=results,cross_estimator_common=joint_results,
        arrays_path=str(arrays_path.relative_to(ROOT)),arrays_sha256=sha(arrays_path),
        shapes_path=str(shapes_path.relative_to(ROOT)),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [shapes_path,shape_arrays_path,bench_path,old_path,old_provenance]},
        source_hashes={name:sha(output/name) for name in sources},
        native_input_sha256=digest.hexdigest(),
        native_input_hash_recipe='Batches of 64 validation rows: contiguous float64 physical NISP, float64 physical RMS, bool mask, all 41x41.',
        checks=dict(snr_matches_original_bins=True,maximum_snr_difference=float(np.max(np.abs(snr-old_snr))),
                    same_figure7_common_ids=True,recombined_rmse_matches_figure7=True))
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shutil.copy2(path,output/'report.json')
    (output/'status.json').write_text(json.dumps(dict(complete=True,local_only=True,bundle_sha256=sha(path)),indent=2)+'\n')
    print('Complete:',path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/shape-snr-v02')
    main(parser.parse_args())
