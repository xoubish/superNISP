"""Frozen-model shear pilot using the supplied NISP effective PSF profiles.

No extra native-pixel convolution. Original Gaussian pilot files are immutable.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import galsim
import numpy as np
from astropy.io import fits

from pilot_shear_response import (ROOT, DATA, CHECKPOINT, EXPECTED_SHA, SHEARS,
    METHODS, sha, observed, corrected, response, draw_sb)

RUN = ROOT/'runs/shear-response-empirical-v01'
PSF_RECORDS = ROOT/'runs/psf-features-v01/report.json'
RADII = [.18, .35, .65]
SNRS = [None, 20, 50, 100]
OBSERVED = [0, 2, 3]


def effective_profile(array, basis, recenter=True):
    """basis maps PSF-grid offsets to native-axis angular offsets (arcsec)."""
    array = np.asarray(array, dtype=float)
    basis = np.asarray(basis, dtype=float)
    assert np.isfinite(array).all() and array.sum() > 0
    wcs = galsim.JacobianWCS(*basis.ravel())
    profile = galsim.InterpolatedImage(galsim.ImageD(array, wcs=wcs), flux=1.,
        x_interpolant=galsim.Lanczos(11), pad_factor=4.,
        gsparams=galsim.GSParams(folding_threshold=1.e-4))
    if recenter:
        fit = galsim.hsm.FindAdaptiveMom(galsim.ImageD(array), strict=True)
        displacement = basis @ np.array([fit.moments_centroid.x-(array.shape[1]+1)/2,
                                         fit.moments_centroid.y-(array.shape[0]+1)/2])
        profile = profile.shift(-displacement[0], -displacement[1])
    return profile


def draw_effective(profile, scale, size=41):
    # Effective PSF already includes detector/stack response. Sampling only.
    return profile.drawImage(nx=size, ny=size, scale=scale, method='no_pixel').array.astype(float)/scale**2


def load_psfs():
    association = json.loads(PSF_RECORDS.read_text())
    manifest = json.loads((DATA/'manifest.json').read_text())
    profiles, records = [], []
    for rec in association['records']:
        path = DATA/'nisp_psf'/rec['filename']
        assert sha(path) == rec['sha256']
        assert any(Path(s['path']).name == path.name and s['sha256'] == sha(path)
                   for s in manifest['source_files'])
        header_path = DATA/'source_headers'/(rec['image']+'.json')
        assert sha(header_path) == rec['source_header_sha256']
        hdr = fits.Header.fromstring(json.loads(header_path.read_text())['SCI'])
        native_cd = np.array([[hdr[f'CD{i}_{j}'] for j in [1,2]] for i in [1,2]])
        with fits.open(path) as hdus:
            array = hdus[1].data.astype(float)
            h = hdus[1].header
            assert hdus[0].header['PSFTYPE'] == 'Inflight'
            assert hdus[0].header['OVERSAMP'] == 6 and h['POLNAXIS'] == 0
            psf_cd = np.array([[h[f'CD{i}_{j}'] for j in [1,2]] for i in [1,2]])
            observation = int(hdus[0].header['OBS_ID'])
            headers = {k:float(h[k]) for k in ['PSF_FWHM','FWHMXPSF','FWHMYPSF','REE50','REE80']}
        scale = float(np.sqrt(abs(np.linalg.det(native_cd)))*3600)
        basis = scale*np.linalg.solve(native_cd, psf_cd)
        profile = effective_profile(array, basis)
        profiles.append((profile, scale))
        # Reconstruct supplied samples with their original WCS, before recentering.
        raw = effective_profile(array, basis, recenter=False)
        recovered = raw.drawImage(nx=61, ny=61, wcs=galsim.JacobianWCS(*basis.ravel()),
                                  method='no_pixel').array
        reconstruction_error = float(np.max(abs(recovered-array/array.sum())))
        assert reconstruction_error < 2.e-6
        records.append(dict(filename=path.name, sha256=sha(path), observation=observation,
            source_header_sha256=sha(header_path), native_cd=native_cd.tolist(),
            psf_cd=psf_cd.tolist(), native_scale_arcsec=scale, basis_arcsec=basis.tolist(),
            signed_sum=float(array.sum()), negative_pixels=int((array<0).sum()),
            header_widths=headers, supplied_sample_max_absolute_error=reconstruction_error,
            psf_native_stamp_flux=float(draw_effective(profile, scale).sum()*scale**2),
            native_psf_adaptive_status=observed(draw_effective(profile, scale),scale)[2]))
    assert len(records) == 4
    return profiles, records


def summary(e, good, truth, common):
    """Return all method statuses; observed-common sample excludes REGAUSS."""
    out = dict(attempted=len(e), common_success=int(common.sum()), methods={})
    assert common.any()
    ref = response(truth[common].mean(axis=0))
    for m, name in enumerate(METHODS):
        ok = bool(good[common,:,m].all())
        rec = dict(success_by_branch=good[:,:,m].sum(axis=0).tolist(), same_sample_available=ok)
        if ok:
            rr = response(e[common,:,m].mean(axis=0))
            rec.update(response=rr.tolist(), truth_response=ref.tolist(),
                relative_trace_response=float(np.trace(rr)/np.trace(ref)),
                zero_shear_mean_e=e[common,0,m].mean(axis=0).tolist())
        out['methods'][name] = rec
    return out


def aggregate(arrays):
    cells = []
    for ri, radius in enumerate(RADII):
        for si, snr in enumerate(SNRS):
            keys = [f'psf{pi}_radius{ri}_snr{si}' for pi in range(4)]
            common = np.ones(len(arrays[keys[0]+'__e']), bool)
            for key in keys:
                common &= arrays[key+'__good'][:,:,OBSERVED].all(axis=(1,2))
            by_psf = [summary(arrays[k+'__e'],arrays[k+'__good'],arrays[k+'__truth'],common)
                      for k in keys]
            values = {}
            for name in METHODS:
                if all(p['methods'][name]['same_sample_available'] for p in by_psf):
                    matrices = np.array([p['methods'][name]['response'] for p in by_psf])
                    ratios = [p['methods'][name]['relative_trace_response'] for p in by_psf]
                    values[name] = dict(response=matrices.mean(axis=0).tolist(),
                        relative_trace_response=float(np.mean(ratios)),
                        per_psf_relative_trace_response=ratios,
                        psf_range=[float(min(ratios)),float(max(ratios))])
            cells.append(dict(half_light_radius=radius, snr=snr, attempted=len(common),
                common_success=int(common.sum()), common_indices=np.flatnonzero(common).tolist(),
                methods=values, by_psf=by_psf))
    return cells


def run(args):
    import torch
    import wandb
    from residual_baseline import EuclidToJWSTSuperResolution
    assert os.environ.get('TMUX'), 'tmux required'
    assert sha(CHECKPOINT) == EXPECTED_SHA
    controls = json.loads((ROOT/'paper/diagnostics/empirical_psf_controls_v01.json').read_text())
    assert controls['passed'] and controls['source_sha256'] == sha(__file__)
    profiles, psfs = load_psfs()
    cfg = dict(version='v01', seed=20261017, half_light_radii_arcsec=RADII, snr=SNRS,
        displayed_radii=[.18,.35], displayed_snr=[20,50,100], psfs=psfs,
        psf_convention='Supplied signed effective PSF, flux-normalized, adaptive-centered; header CD mapping into native image axes; Lanczos11; render no_pixel, no extra detector convolution.',
        psf_aggregation='Equal weight across all four supplied stack PSFs, not field-abundance weighting. Common observed-method success across all PSFs and five shear branches. Mean per-PSF response ratios.',
        psf_geometry_limit='Uses linear supplied PSF/native header bases as the local orientation convention; no spatially varying atlas or independent stellar reconstruction validation.',
        radii_selection='0.18 and 0.35 chosen in earlier Gaussian pilot for larger gains; 0.65 still evaluated and reported.',
        shears=SHEARS, intrinsic_distortion=.35, rotations=8, noisy_repeats=4,
        noise_sigma_mjy_sr=.028, native_stamp=41, output_stamp=205,
        noise='Same Gaussian noise for five shear branches and all four PSFs at fixed radius/SNR/orientation/repeat. Noiseless uses SNR-50 flux. Independent noise rings; no uncertainty claim.',
        flux='Same original Gaussian weight (sigma 2 px, radius 5); circular unsheared profile sets nominal SNR separately for each PSF; flux fixed across shear and rotation.',
        target='Unchanged noiseless circular Gaussian FWHM 0.08 arcsec; same exponential profile; pixel integration at native scale/5. Not calibrated JWST PSF.',
        measurements='Adaptive observed shapes on full stamps; native REGAUSS auxiliary only and never selects observed sample.',
        model='Original frozen epoch-49 ResNet; no updates or selection.',
        checkpoint_sha256=sha(CHECKPOINT), normalization_sha256=sha(DATA/'normalization.json'),
        controls_sha256=sha(ROOT/'paper/diagnostics/empirical_psf_controls_v01.json'),
        psf_associations_sha256=sha(PSF_RECORDS), protocol_sha256=sha(ROOT/'paper/empirical_shear_protocol.md'),
        source_hashes={n:sha(ROOT/'code'/n) for n in ['empirical_shear_response.py',
            'test_empirical_shear.py','pilot_shear_response.py','residual_baseline.py']},
        versions=dict(galsim=galsim.__version__,torch=torch.__version__,numpy=np.__version__))
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/'contract.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for n in cfg['source_hashes']: shutil.copy2(ROOT/'code'/n,out/n)
    shutil.copy2(ROOT/'paper/empirical_shear_protocol.md',out/'protocol.md')
    norm = json.loads((DATA/'normalization.json').read_text())
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(.12,device)
    started=time.time()
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name=out.name,
        job_type='empirical-psf-shear-response',mode='online',dir=str(out),
        config=cfg,settings=wandb.Settings(init_timeout=60)) as tracking:
        assert tracking.settings.mode=='online' and tracking.url
        (out/'wandb_url.txt').write_text(tracking.url+'\n')
        parent=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        for name,digest in parent['signature']['data_hashes'].items(): assert sha(DATA/name)==digest
        assert cfg['source_hashes']['residual_baseline.py']==parent['signature']['source_hashes']['residual_baseline.py']
        model=EuclidToJWSTSuperResolution(head='legacy').to(device).eval()
        model.load_state_dict(parent['model']);del parent
        for p in model.parameters():p.requires_grad_(False)
        y,x=np.mgrid[-20:21,-20:21]
        weight=np.exp(-(x*x+y*y)/8)*(x*x+y*y<=25)
        arrays={};records=[]
        for pi,(psf,scale) in enumerate(profiles):
            psf_image=draw_effective(psf,scale)
            for ri,radius in enumerate(RADII):
                base=galsim.Exponential(half_light_radius=radius)
                circular=draw_effective(galsim.Convolve(base,psf),scale)
                unit_snr=np.sum(circular*weight)/(.028*np.sqrt(np.sum(weight**2)))
                for si,snr in enumerate(SNRS):
                    flux=(50 if snr is None else snr)/unit_snr
                    repeats=1 if snr is None else 4
                    count=8*repeats
                    native=np.empty((count,5,41,41))
                    e=np.full((count,5,4,2),np.nan);good=np.zeros((count,5,4),bool)
                    status=np.full((count,5,4),-999,int);truth=np.empty((count,5,2))
                    for rep in range(repeats):
                        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],ri,si,rep]))
                        for rot in range(8):
                            obj=rep*8+rot
                            intrinsic=galsim.Shear(e=.35,beta=rot*np.pi/8*galsim.radians)
                            galaxy=base.withFlux(flux).shear(intrinsic)
                            noise=np.zeros((41,41)) if snr is None else rng.normal(0,.028,(41,41))
                            for s,(g1,g2) in enumerate(SHEARS):
                                lens=galsim.Shear(g1=g1,g2=g2);shape=lens+intrinsic
                                truth[obj,s]=[shape.e1,shape.e2]
                                sheared=galaxy.shear(lens)
                                native[obj,s]=draw_effective(galsim.Convolve(sheared,psf),scale)+noise
                                target=draw_sb(galsim.Convolve(sheared,galsim.Gaussian(fwhm=.08)),scale/5,205)
                                for m,measurement in [(0,observed(native[obj,s],scale)),
                                    (1,corrected(native[obj,s],psf_image,scale)),(3,observed(target,scale/5))]:
                                    e[obj,s,m],good[obj,s,m],status[obj,s,m]=measurement
                    normalized=(native/norm['nisp_scale']).reshape(-1,1,41,41).astype('float32')
                    with torch.inference_mode():
                        for start in range(0,len(normalized),4):
                            pred=model(torch.from_numpy(normalized[start:start+4]).to(device))
                            pred=pred[:,0].cpu().numpy()*norm['jwst_scale']
                            for offset,image in enumerate(pred):
                                obj,s=divmod(start+offset,5)
                                e[obj,s,2],good[obj,s,2],status[obj,s,2]=observed(image,scale/5)
                    key=f'psf{pi}_radius{ri}_snr{si}'
                    arrays.update({key+'__'+k:v for k,v in dict(e=e,good=good,status=status,truth=truth).items()})
                    common=good[:,:,OBSERVED].all(axis=(1,2))
                    rec=dict(psf_index=pi,observation=psfs[pi]['observation'],half_light_radius=radius,
                        snr=snr,flux=flux,**summary(e,good,truth,common))
                    records.append(rec)
                    np.savez_compressed(out/'measurements.npz',**arrays)
                    (out/'progress.json').write_text(json.dumps(dict(complete=False,cells_done=len(records),
                        cells_total=48,elapsed_seconds=time.time()-started,results=records),indent=2)+'\n')
                    tracking.log(dict(cells_done=len(records),common_success=rec['common_success'],
                        resnet_relative_response=rec['methods']['resnet_observed']['relative_trace_response']))
                    print(key,'common',rec['common_success'],'/',count,flush=True)
        assert sha(CHECKPOINT)==EXPECTED_SHA
        report=dict(complete=True,contract=cfg,cells=records,aggregated=aggregate(arrays),
            measurements_sha256=sha(out/'measurements.npz'),checkpoint_unchanged=True,
            elapsed_seconds=time.time()-started,wandb_url=tracking.url)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        tracking.summary['complete']=True
        tracking.summary['elapsed_seconds']=report['elapsed_seconds']
        (out/'status.json').write_text(json.dumps(dict(complete=True,cells=48,
            report_sha256=sha(out/'report.json')),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=RUN)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
