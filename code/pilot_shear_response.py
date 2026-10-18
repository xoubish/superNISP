"""Synthetic shear-response pilot. Not a calibration of the observed galaxy sample.

ResNet execution requires tmux and online W&B, following HANDOFF_GPU.md.
No real galaxy images or reserved test objects are read. Never PSF-correct ResNet.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import galsim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/'supernisp_nir_y_f115w_v1'
CHECKPOINT = ROOT/'runs/residual-full50-s20261016/best.pt'
EXPECTED_SHA = 'be747f0cd0b02bb81ff0ab42a9ce8ea81a0e473c28c58d8406a428f758ea6b56'
SCALE = 0.29672
DG = .01
SHEARS = [(0.,0.),(DG,0.),(-DG,0.),(0.,DG),(0.,-DG)]
METHODS = ['native_observed','native_regauss','resnet_observed','target_observed']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_sb(profile, scale, size):
    """Flux-integrated pixels divided by pixel area: surface-brightness units."""
    return profile.drawImage(nx=size,ny=size,scale=scale).array.astype(float)/scale**2


def observed(image, scale):
    im = galsim.ImageD(np.asarray(image,dtype=float), scale=scale)
    result = galsim.hsm.FindAdaptiveMom(im,guess_sig=max(1.,.4/scale),strict=False)
    e = np.array([result.observed_shape.e1,result.observed_shape.e2])
    good = result.moments_status == 0 and not result.error_message and np.isfinite(e).all()
    return (e if good else np.full(2,np.nan)), bool(good), int(result.moments_status)


def corrected(image, psf_image, scale):
    result = galsim.hsm.EstimateShear(galsim.ImageD(image,scale=scale),
        galsim.ImageD(psf_image,scale=scale),guess_sig_gal=max(1.,.4/scale),
        guess_sig_PSF=1., shear_est='REGAUSS',strict=False)
    e = np.array([result.corrected_e1,result.corrected_e2])
    good = result.correction_status == 0 and not result.error_message and np.isfinite(e).all()
    return (e if good else np.full(2,np.nan)), bool(good), int(result.correction_status)


def response(e):
    """Mean e[g branch, component] -> R[component, applied shear component]."""
    return np.stack([(e[1]-e[2])/(2*DG),(e[3]-e[4])/(2*DG)],axis=-1)


def summarize(e, good, truth):
    # e: object, shear branch, method, component. All methods share objects.
    common = np.all(good,axis=(1,2))
    result = dict(attempted=len(e), common_success=int(common.sum()), methods={})
    for m,name in enumerate(METHODS):
        branch_good = good[:,:,m]
        selected = np.stack([np.nanmean(e[branch_good[:,s],s,m],axis=0)
                             if branch_good[:,s].any() else [np.nan,np.nan]
                             for s in range(5)])
        rec = dict(success_by_branch=branch_good.sum(axis=0).tolist(),
                   branch_selected_response=response(selected).tolist())
        if common.any():
            mean = e[common,:,m].mean(axis=0)
            ref = response(truth[common].mean(axis=0))
            rr = response(mean)
            rec.update(response=rr.tolist(), truth_response=ref.tolist(),
                relative_trace_response=float(np.trace(rr)/np.trace(ref)),
                zero_shear_mean_e=mean[0].tolist())
        result['methods'][name] = rec
    return result


def configuration():
    return dict(version='v01', purpose='Controlled feasibility diagnostic; not survey shear calibration',
        seed=20261017, shears=SHEARS, half_light_radii_arcsec=[.18,.35,.65],
        nisp_psf_fwhm_arcsec=[.45,.60], snr=[None,20,50,100],
        intrinsic_distortion=.35, rotations=8, noisy_repeats=4, noiseless_repeats=1,
        native_pixel_scale_arcsec=SCALE, output_pixel_scale_arcsec=SCALE/5,
        native_stamp=41, output_stamp=205, noise_sigma_mjy_sr=.028,
        noise='Independent Gaussian pixels; identical realization across all five shear branches. Four independent noise rings per noisy cell; no uncertainty/precision claim.',
        flux='Set once per size/PSF/SNR from a circular unsheared profile and fixed Gaussian-weight SNR (sigma 2 native pixels, radius 5). Do not renormalize after shear/rotation. Noiseless cell uses SNR-50 flux.',
        profiles='Exponential, fixed intrinsic |e|=.35; 8 angles spaced pi/8, fixed central position. Applied shear occurs before PSF convolution and pixel integration.',
        psfs='Analytic Gaussian NISP-like FWHM .45/.60 arcsec, e1=.03 e2=.02. Illustrative high-resolution target Gaussian FWHM .08 arcsec, circular. Neither is a validated instrument PSF.',
        target='Noiseless same-band high-resolution control; same intrinsic profile and surface-brightness flux. No bandpass/color gradients or target noise.',
        measurements='Adaptive observed distortion on full stamps; REGAUSS only for native image with exact known pixel-integrated PSF. Raw errors/status retained; no clipping successful corrected distortions.',
        comparison='Primary response on intersection of all five branches and all methods within each cell. Also retain per-branch success and separately selected means. No detection rerun or full selection correction.',
        response='Finite difference d<e_i>/dg_j; relative trace divided by analytic pre-PSF intrinsic response for the identical common objects. This ratio is not multiplicative shear bias.',
        model='Frozen original legacy ResNet epoch 49, evaluation mode; training normalization; no optimizer, updates, or checkpoint writes.',
        limitations=['Small idealized morphology/noise population and fixed pixel phase.',
            'No blends, correlated noise, real PSF variation, chromatic effects, or detection selection.',
            'Common-success response is conditional; branch-selected response is diagnostic, not a selection correction.',
            'No effective output PSF assumed; no PSF correction of reconstructed images.',
            'No response correction fitted and tested on independent simulations; no calibrated shear catalog.'])


def run_pilot(args):
    import torch
    import wandb
    from residual_baseline import EuclidToJWSTSuperResolution

    if not (os.environ.get('TMUX') or os.environ.get('STY')):
        raise RuntimeError('Persistent tmux/screen session required')
    if sha(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError('Unexpected checkpoint')
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    cfg = configuration()
    cfg.update(checkpoint_sha256=sha(CHECKPOINT),
        normalization_sha256=sha(DATA/'normalization.json'),
        source_hashes={p:sha(ROOT/'code'/p) for p in
            ['pilot_shear_response.py','residual_baseline.py','audit_shape_psfs.py','test_shear_response.py']},
        psf_audit_sha256=sha(ROOT/'paper/diagnostics/psf_audit_v01.json'),
        versions=dict(galsim=galsim.__version__,numpy=np.__version__,torch=torch.__version__))
    (out/'contract.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for name in cfg['source_hashes']:
        shutil.copy2(ROOT/'code'/name,out/name)
    norm = json.loads((DATA/'normalization.json').read_text())
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.12,device)
    started = time.time()
    # Required existing online-monitoring policy: no model execution before init.
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name=out.name,
        job_type='synthetic-shear-response-pilot',mode='online',dir=str(out),
        config=cfg,settings=wandb.Settings(init_timeout=60)) as tracking:
        if tracking.settings.mode != 'online' or not tracking.url:
            raise RuntimeError('Online monitoring unavailable')
        (out/'wandb_url.txt').write_text(tracking.url+'\n')
        parent = torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        for name,digest in parent['signature']['data_hashes'].items():
            assert sha(DATA/name) == digest, name
        assert cfg['source_hashes']['residual_baseline.py'] == parent['signature']['source_hashes']['residual_baseline.py']
        model = EuclidToJWSTSuperResolution(head='legacy').to(device).eval()
        model.load_state_dict(parent['model'])
        del parent
        for p in model.parameters():
            p.requires_grad_(False)
        y,x = np.mgrid[-20:21,-20:21]
        weight = np.exp(-(x*x+y*y)/8)*(x*x+y*y<=25)
        records, arrays = [], {}
        high_psf = galsim.Gaussian(fwhm=.08)
        for pi,fwhm in enumerate(cfg['nisp_psf_fwhm_arcsec']):
            psf = galsim.Gaussian(fwhm=fwhm).shear(e1=.03,e2=.02)
            psf_image = draw_sb(psf,SCALE,41)
            for ri,radius in enumerate(cfg['half_light_radii_arcsec']):
                base = galsim.Exponential(half_light_radius=radius)
                circular = draw_sb(galsim.Convolve(base,psf),SCALE,41)
                unit_snr = np.sum(circular*weight)/(.028*np.sqrt(np.sum(weight**2)))
                for si,snr in enumerate(cfg['snr']):
                    flux = (50 if snr is None else snr)/unit_snr
                    repeats = 1 if snr is None else cfg['noisy_repeats']
                    count = repeats*8
                    native = np.empty((count,5,41,41))
                    e = np.full((count,5,len(METHODS),2),np.nan)
                    good = np.zeros((count,5,len(METHODS)),bool)
                    status = np.full((count,5,len(METHODS)),-999,int)
                    truth = np.empty((count,5,2))
                    for rep in range(repeats):
                        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'],pi,ri,si,rep]))
                        for rot in range(8):
                            obj = rep*8+rot
                            intrinsic = galsim.Shear(e=.35,beta=rot*np.pi/8*galsim.radians)
                            galaxy = base.withFlux(flux).shear(intrinsic)
                            noise = np.zeros((41,41)) if snr is None else rng.normal(0,.028,(41,41))
                            for s,(g1,g2) in enumerate(SHEARS):
                                lens = galsim.Shear(g1=g1,g2=g2)
                                shape = lens+intrinsic
                                truth[obj,s] = [shape.e1,shape.e2]
                                sheared = galaxy.shear(lens)
                                native[obj,s] = draw_sb(galsim.Convolve(sheared,psf),SCALE,41)+noise
                                target = draw_sb(galsim.Convolve(sheared,high_psf),SCALE/5,205)
                                for m,measurement in [(0,observed(native[obj,s],SCALE)),
                                        (1,corrected(native[obj,s],psf_image,SCALE)),
                                        (3,observed(target,SCALE/5))]:
                                    e[obj,s,m],good[obj,s,m],status[obj,s,m] = measurement
                    normalized = (native/norm['nisp_scale']).reshape(-1,1,41,41).astype('float32')
                    with torch.inference_mode():
                        for start in range(0,len(normalized),4):
                            pred = model(torch.from_numpy(normalized[start:start+4]).to(device))
                            pred = pred[:,0].cpu().numpy()*norm['jwst_scale']
                            for offset,image in enumerate(pred):
                                obj,s = divmod(start+offset,5)
                                e[obj,s,2],good[obj,s,2],status[obj,s,2] = observed(image,SCALE/5)
                    rec = dict(psf_fwhm=fwhm,half_light_radius=radius,snr=snr,flux=flux,
                               **summarize(e,good,truth))
                    records.append(rec)
                    key = f'psf{pi}_radius{ri}_snr{si}'
                    arrays.update({key+'__'+k:v for k,v in dict(e=e,good=good,status=status,truth=truth).items()})
                    np.savez_compressed(out/'measurements.npz',**arrays)
                    (out/'progress.json').write_text(json.dumps(dict(complete=False,cells_done=len(records),cells_total=24,
                        seconds=time.time()-started,results=records),indent=2)+'\n')
                    tracking.log(dict(cells_done=len(records),common_success=rec['common_success'],attempted=count,
                        resnet_relative_response=rec['methods']['resnet_observed'].get('relative_trace_response',float('nan'))))
                    print(key, 'common',rec['common_success'],'/',count,flush=True)
        assert sha(CHECKPOINT) == EXPECTED_SHA
        report = dict(complete=True,contract=cfg,wandb_url=tracking.url,cells=records,
            elapsed_seconds=time.time()-started,checkpoint_unchanged=True,
            measurements_sha256=sha(out/'measurements.npz'))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        plot(report,out)
        tracking.log({'response_plot':wandb.Image(str(out/'response.png'))})
        (out/'status.json').write_text(json.dumps(dict(complete=True,cells=24,
            report_sha256=sha(out/'report.json'),plot_sha256=sha(out/'response.png')),indent=2)+'\n')


def plot(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = [('native_observed','Native observed','0.4','o'),
              ('native_regauss','Native + known PSF correction','#228833','^'),
              ('resnet_observed','ResNet observed','#4477aa','s'),
              ('target_observed','High-resolution control','#cc8833','D')]
    fig,axes = plt.subplots(2,3,figsize=(11,6),sharex=True,sharey=True)
    for i,psf in enumerate([.45,.60]):
        for j,snr in enumerate([20,50,100]):
            ax=axes[i,j]
            rows=[r for r in report['cells'] if r['psf_fwhm']==psf and r['snr']==snr]
            for name,label,color,marker in styles:
                ax.plot([r['half_light_radius'] for r in rows],
                        [r['methods'][name].get('relative_trace_response',np.nan) for r in rows],
                        color=color,marker=marker,label=label)
            ax.axhline(1,color='black',lw=.8,ls=':')
            ax.set_title(f'S/N {snr}; PSF FWHM {psf:.2f} arcsec',fontsize=10)
            ax.text(.98,.04,'Common: '+', '.join(f"{r['common_success']}/{r['attempted']}" for r in rows),
                    transform=ax.transAxes,ha='right',fontsize=8)
            ax.set_xticks([.18,.35,.65])
            ax.grid(alpha=.15)
            if i==1: ax.set_xlabel('Intrinsic half-light radius [arcsec]')
            if j==0: ax.set_ylabel('Response / intrinsic response')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.015,'Synthetic pilot • conditional on common measurement success • no calibrated shear or uncertainty claim',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.90))
    for ext in ['png','pdf']:
        fig.savefig(out/f'response.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/shear-response-pilot-v01')
    parser.add_argument('--device',default='cuda:0')
    run_pilot(parser.parse_args())
