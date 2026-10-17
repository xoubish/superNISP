"""Local CPU-only shape comparison using immutable cached reconstructions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
import torch
from analyze_spatial_spectra import sha, METHODS, ROOT
from shape_diagnostics import diagnose_moments, REASONS
from robust_shape_measurements import adaptive_moments, HSM_REASONS
from cached_shape_comparison import summarize


def main(args):
    torch.set_num_threads(2)
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    bundle=ROOT/'paper_figures/inputs'/f'cached_shapes_{args.version}.json'
    arrays_path=bundle.with_suffix('.npz')
    if bundle.exists() or arrays_path.exists():raise FileExistsError('Use a fresh version')
    bench_path=ROOT/'paper_figures/inputs/model_benchmark_v03.json'
    bench=json.loads(bench_path.read_text());cache=ROOT/bench['prediction_cache']['path'];mask_path=ROOT/bench['source_mask_path']
    assert sha(cache)==bench['prediction_cache']['sha256']
    assert sha(mask_path)==bench['source_mask_sha256']
    with np.load(mask_path) as f:valid=f['common'];ids=f['object_ids']
    assert ids.tolist()==bench['object_ids']
    blocks=np.asarray(bench['blocks']);n=len(ids)
    contract=dict(local_only=True,split='validation',version=args.version,
        population='All 2118 cached validation fields attempted. Each estimator compares the identical intersection of successful JWST and all six model/baseline measurements. Full-denominator success counts and overlapping failure flags retained. Also report intersection across both estimators.',
        fixed='Existing signed Gaussian-weighted moments: sigma 10 target pixels, central aperture radius 25; intensity-weighted centroid with fixed window. e1=(Qxx-Qyy)/(Qxx+Qyy), e2=2Qxy/(Qxx+Qyy). Require positive flux and positive-definite quadrupoles and full valid aperture.',
        adaptive='Existing GalSim HSM adaptive moments on 105x105 target pixels; guess sigma 10; max centroid displacement 10 target pixels; bad pixels masked, full radius-25 validity required; retain convergence/failure details. Observed distortion e1/e2; no PSF correction.',
        masks='Same cached common-valid mask for all methods and reference; NO segmentation/source clipping of shapes, which would imprint a reference outline. Fixed central weight/aperture targets the central object; neighbours may contaminate it. HSM centroid guard reduces neighbour capture but is not deblending.',
        images='Signed physical MJy/sr, no clipping, additional background subtraction, smoothing or model inference. Frozen Figure5/6 checkpoints and diffusion seeds from benchmark v03. Compare single cached diffusion draw and shape measured on mean-of-3 image; the latter is not the mean of three shape estimates.',
        summary='Component RMSE sqrt(mean_over_objects((delta_e1**2+delta_e2**2)/2)); per-component bias, scatter, fraction with vector error>0.2, diagnostic OLS slopes; HSM log-size ratios. Round-control e1=e2=0 is diagnostic only.',
        bootstrap='2000 paired spatial-block draws, seed20261017, equal object weighting, conditional on the common-success sample. Differences are paired RMSE differences; no independent bootstrap comparison.',
        limitations=['Observed cross-band shapes, not calibrated gravitational shear or PSF-corrected morphology.',
                     'Failure-dependent selection can favour easier galaxies; report success fractions beside errors.',
                     'Fixed and adaptive moments use different weights and measure different shapes; compare model rankings within each estimator.',
                     'Single training seed and one cached individual diffusion realization; no claim about realization-to-realization shape uncertainty.',
                     'One field, validation-selected checkpoints; not independent held-out test performance.'])
    (output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    sources=['analyze_cached_shapes.py','cached_shape_comparison.py','test_cached_shape_comparison.py',
             'shape_diagnostics.py','robust_shape_measurements.py','analyze_spatial_spectra.py',
             'spatial_spectra.py','test_shape_robustness.py']
    for name in sources:shutil.copy2(ROOT/'code'/name,output/name)
    collected={case:{m:[] for m in ['target',*METHODS]} for case in ['fixed','adaptive']}
    with h5py.File(cache) as f:
        assert f.attrs['complete'] and np.array_equal(f['object_ids'][:],ids)
        for start in range(0,n,32):
            end=min(start+32,n);mask=torch.from_numpy(valid[start:end,None])
            for m in ['target',*METHODS]:
                image=torch.from_numpy(f[m][start:end][:,None].astype(np.float64))
                fixed=diagnose_moments(image,mask)
                collected['fixed'][m].append({k:v.numpy() for k,v in fixed.items()})
                collected['adaptive'][m].append(adaptive_moments(image,mask))
            if end%256==0 or end==n:
                print(f'Shapes {end}/{n}',flush=True)
                (output/'progress.json').write_text(json.dumps(dict(objects=end,total=n)))
    data={case:{m:{k:np.concatenate([p[k] for p in parts]) for k in parts[0]} for m,parts in methods.items()} for case,methods in collected.items()}
    global_common=np.logical_and.reduce([v['good'] for methods in data.values() for v in methods.values()])
    summaries={case:summarize(methods,blocks,ids,METHODS,REASONS if case=='fixed' else HSM_REASONS) for case,methods in data.items()}
    matched={case:summarize(methods,blocks,ids,METHODS,REASONS if case=='fixed' else HSM_REASONS,selection=global_common) for case,methods in data.items()}
    np.savez_compressed(arrays_path,object_ids=ids,blocks=blocks,global_common=global_common,
        **{f'{case}__{m}__{k}':v for case,methods in data.items() for m,values in methods.items() for k,v in values.items()})
    report=dict(complete=True,**contract,methods=METHODS,object_ids=ids.tolist(),blocks=blocks.tolist(),
        summaries=summaries,cross_estimator_common=matched,cross_estimator_objects=int(global_common.sum()),
        arrays_path=str(arrays_path.relative_to(ROOT)),arrays_sha256=sha(arrays_path),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [bench_path,cache,mask_path]},
        source_hashes={name:sha(output/name) for name in sources},
        versions={m:importlib.metadata.version(m) for m in ['numpy','torch','galsim','h5py']})
    bundle.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shutil.copy2(bundle,output/'report.json')
    (output/'status.json').write_text(json.dumps(dict(complete=True,bundle_sha256=sha(bundle),local_only=True),indent=2)+'\n')
    print('Complete',bundle,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--version',default='v01')
    main(p.parse_args())
