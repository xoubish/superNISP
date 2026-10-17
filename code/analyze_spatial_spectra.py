"""Local CPU-only Fourier diagnostic from the frozen Figure 6 prediction cache."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from spatial_spectra import EDGES, BANDS, windows, transform, frequency_masks, paired_spectrum, block_summary

ROOT=Path(__file__).resolve().parents[1]
METHODS=['bilinear','bicubic','diffusion','diffusion_mean3','legacy','direct5']
CASES=['tukey50','tukey25','hann','soft_source','background']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    bundle=ROOT/'paper_figures/inputs'/f'spatial_spectra_{args.version}.json'
    arrays_path=bundle.with_suffix('.npz')
    if bundle.exists() or arrays_path.exists():raise FileExistsError('Use a fresh version')
    benchmark_path=ROOT/'paper_figures/inputs/model_benchmark_v03.json'
    benchmark=json.loads(benchmark_path.read_text())
    cache=ROOT/benchmark['prediction_cache']['path'];mask_path=ROOT/benchmark['source_mask_path']
    assert sha(cache)==benchmark['prediction_cache']['sha256']
    assert sha(mask_path)==benchmark['source_mask_sha256']
    with np.load(mask_path) as saved:masks={k:saved[k] for k in saved.files}
    assert masks['object_ids'].tolist()==benchmark['object_ids']
    source_index=masks['thresholds'].tolist().index(2.)
    n=len(benchmark['object_ids']); frequency=frequency_masks(); nb=len(frequency)
    contract=dict(local_only=True,split='validation',version=args.version,
        main_population='All cutouts with complete common-valid coverage in the cached central 105x105 field; no model-success or error cuts.',
        main_window='Separable Tukey alpha=0.5; subtract each image own weighted mean before FFT; FFT divided by sqrt(sum(window**2)). Same window for every model and JWST.',
        frequencies='Cycles per target pixel. Radial annular means, excluding DC and radii above 0.5. Mean frequencies measured from actual included Fourier modes.',
        power='Geometric mean over objects of P_model/P_observed_JWST (mean log10 ratio); ideal 1. No image flux fitting, PSF correction or noise subtraction.',
        correlation='Mean over objects of real cross-spectrum / sqrt(P_model*P_observed_JWST), each annularly averaged first. Signed, phase-sensitive r; ideal 1. Not magnitude-squared coherence.',
        bootstrap='2000 paired spatial-block draws, seed 20261017. Objects have equal weight within each statistic; same block draws for all models.',
        background='Outside the fixed 2-sigma source mask, with a 3-pixel cosine taper to source/invalid boundaries and the outer Tukey window. Window-energy normalized. Usable if sum(w**2)>16 and effective area sum(w)**2/sum(w**2)>=64. An empirical background proxy, not an unbiased noise PSD; contains faint sources and mask coupling.',
        checks='Tukey alpha=0.25, Hann, a soft source weight (2-sigma source mask Gaussian-smoothed with sigma=3 pixels), and all 2118 cutouts with validity-boundary taper width=5.',
        contrast_subset='For each predeclared scale band, retain primary objects with usable background and observed reference/background power ratio >3. Reference-only sensitivity diagnostic, not calibrated S/N or a noise correction.',
        bands_cycles_per_pixel=BANDS,radial_edges_cycles_per_pixel=EDGES.tolist(),
        limitations=['No independent JWST exposures or known reference noise PSD: no claim of noise-debiased resolution.',
                     'Power agreement may reflect generated noise or plausible but misplaced structure; interpret jointly with r.',
                     'Windowing mixes Fourier modes; matched windows and alternative tapers are diagnostics, not a window deconvolution.',
                     'Different NISP/JWST bands and instrumental responses; observed reference is not a noiseless latent truth.',
                     'Fixed validation-selected checkpoints and one training seed per model. No inference or training in this job.'])
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for name in ['spatial_spectra.py','analyze_spatial_spectra.py','test_spatial_spectra.py']:
        shutil.copy2(ROOT/'code'/name,args.output/name)
    shape=(n,len(CASES),len(METHODS),nb)
    arrays={key:np.full(shape,np.nan) for key in ['power','cross','log_power_ratio','correlation']}
    reference=np.full((n,len(CASES),nb),np.nan)
    areas=np.zeros((n,len(CASES),2))
    fully_valid=masks['common'].all((1,2));background_good=np.zeros(n,bool)
    with h5py.File(cache) as saved:
        assert saved.attrs['complete'] and saved['object_ids'][:].tolist()==benchmark['object_ids']
        for i in range(n):
            images={name:saved[name][i] for name in METHODS+['target']}
            weights=windows(masks['common'][i],masks['source'][i,source_index])
            for j,case in enumerate(CASES):
                w=weights[case];energy=np.sum(w*w);effective=w.sum()**2/max(energy,1e-30)
                areas[i,j]=[energy,effective]
                if case=='background':
                    background_good[i]=energy>16 and effective>=64
                    if not background_good[i]:continue
                ref=transform(images['target'],w)
                for k,name in enumerate(METHODS):
                    result=paired_spectrum(transform(images[name],w),ref,frequency)
                    for key in arrays:arrays[key][i,j,k]=result[key]
                    reference[i,j]=result['reference']
            if (i+1)%200==0 or i+1==n:
                print(f'Spectra {i+1}/{n}',flush=True)
                (args.output/'progress.json').write_text(json.dumps(dict(objects=i+1,total=n)))
    blocks=np.asarray(benchmark['blocks']);ids=np.asarray(benchmark['object_ids'])
    summaries={}
    selections={case:fully_valid & (background_good if case=='background' else True) for case in CASES}
    selections['all_validity_taper']=np.ones(n,bool)
    for case,keep in selections.items():
        j=CASES.index(case) if case in CASES else 0
        summaries[case]=dict(object_ids=ids[keep].tolist(),objects=int(keep.sum()),methods={})
        for k,name in enumerate(METHODS):
            summaries[case]['methods'][name]={key:block_summary(arrays[key][keep,j,k],blocks[keep])
                                              for key in ['log_power_ratio','correlation']}
    noise_keep=fully_valid & background_good
    noise_log=np.log10(reference[noise_keep,0]/reference[noise_keep,4])
    noise_summary=block_summary(noise_log,blocks[noise_keep])
    band_summaries={}
    for bi,band in enumerate(BANDS):
        index=len(EDGES)-1+bi
        contrast=np.full(n,np.nan)
        contrast[background_good]=reference[background_good,0,index]/reference[background_good,4,index]
        keep=fully_valid & background_good & (contrast>3)
        band_summaries[band]=dict(objects=int(keep.sum()),object_ids=ids[keep].tolist(),methods={},paired={})
        if len(np.unique(blocks[keep]))<2:continue
        for k,name in enumerate(METHODS):
            band_summaries[band]['methods'][name]={key:block_summary(arrays[key][keep,0,k,index],blocks[keep])
                for key in ['log_power_ratio','correlation']}
        for baseline in ['legacy','direct5']:
            a=METHODS.index('diffusion');z=METHODS.index(baseline)
            band_summaries[band]['paired'][baseline]=dict(
                correlation_difference=block_summary(arrays['correlation'][keep,0,a,index]-arrays['correlation'][keep,0,z,index],blocks[keep]),
                absolute_log_power_error_difference=block_summary(np.abs(arrays['log_power_ratio'][keep,0,a,index])-np.abs(arrays['log_power_ratio'][keep,0,z,index]),blocks[keep]))
    # Power-distance paired comparisons on the complete main population as well.
    paired={}
    for baseline in ['legacy','direct5']:
        a=METHODS.index('diffusion');z=METHODS.index(baseline)
        paired[baseline]={}
        for case in ['tukey50','tukey25','hann','soft_source']:
            j=CASES.index(case)
            paired[baseline][case]=dict(
                correlation_difference=block_summary(arrays['correlation'][fully_valid,j,a]-arrays['correlation'][fully_valid,j,z],blocks[fully_valid]),
                absolute_log_power_error_difference=block_summary(np.abs(arrays['log_power_ratio'][fully_valid,j,a])-np.abs(arrays['log_power_ratio'][fully_valid,j,z]),blocks[fully_valid]))
    np.savez_compressed(arrays_path,**arrays,reference=reference,window_areas=areas,
        object_ids=ids,fully_valid=fully_valid,background_good=background_good)
    freq=np.fft.fftfreq(105);radius=np.hypot(freq[:,None],freq[None,:])
    report=dict(complete=True,**contract,methods=METHODS,cases=CASES,object_ids=ids.tolist(),blocks=blocks.tolist(),
        primary_objects=int(fully_valid.sum()),excluded_incomplete_object_ids=ids[~fully_valid].tolist(),
        frequencies_cycles_per_pixel=[float(radius[m].mean()) for m in frequency],
        modes_per_bin=[int(m.sum()) for m in frequency],summaries=summaries,
        reference_background_log_ratio=noise_summary,reference_background_ids=ids[noise_keep].tolist(),
        contrast_band_summaries=band_summaries,paired_diffusion_vs_resnet=paired,
        arrays_path=str(arrays_path.relative_to(ROOT)),arrays_sha256=sha(arrays_path),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [benchmark_path,cache,mask_path]},
        source_hashes={p.name:sha(p) for p in args.output.glob('*.py')},
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','h5py']},python=sys.version)
    bundle.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shutil.copy2(bundle,args.output/'spectra.json')
    (args.output/'status.json').write_text(json.dumps(dict(complete=True,bundle_sha256=sha(bundle),local_only=True),indent=2)+'\n')
    print('Complete',bundle,'primary objects',int(fully_valid.sum()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--version',default='v01')
    main(p.parse_args())
