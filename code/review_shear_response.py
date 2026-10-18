"""Independent checks and an explicitly secondary comparison of frozen pilot shapes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/shear-response-pilot-v01'
OUT=ROOT/'paper/diagnostics'
METHODS=['native_observed','native_regauss','resnet_observed','target_observed']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matrix(e):
    return np.array([(e[1]-e[2])/.02,(e[3]-e[4])/.02]).T


def main():
    original=json.loads((RUN/'report.json').read_text())
    assert original['complete'] and original['checkpoint_unchanged']
    assert (ROOT/'runs/shear-response-pilot-v01-console/exit_status').read_text().strip()=='0'
    assert sha(RUN/'measurements.npz')==original['measurements_sha256']
    for name,digest in original['contract']['source_hashes'].items():
        assert sha(RUN/name)==digest
    assert sha(ROOT/'runs/residual-full50-s20261016/best.pt')==original['contract']['checkpoint_sha256']
    assert sha(OUT/'psf_audit_v01.json')==original['contract']['psf_audit_sha256']
    cells=[]
    with np.load(RUN/'measurements.npz') as a:
        for pi,psf in enumerate([.45,.60]):
            for ri,radius in enumerate([.18,.35,.65]):
                for si,snr in enumerate([None,20,50,100]):
                    key=f'psf{pi}_radius{ri}_snr{si}'
                    e,good,truth=[a[key+'__'+s] for s in ['e','good','truth']]
                    status=a[key+'__status']
                    assert good.shape==e.shape[:-1]
                    assert np.isfinite(e[good]).all() and np.isnan(e[~good]).all()
                    old=next(c for c in original['cells'] if (c['psf_fwhm'],c['half_light_radius'],c['snr'])==(psf,radius,snr))
                    common=np.all(good,axis=(1,2))
                    assert common.sum()==old['common_success']
                    for m,name in enumerate(METHODS):
                        assert good[:,:,m].sum(axis=0).tolist()==old['methods'][name]['success_by_branch']
                        if common.any():
                            expected=matrix(np.sum(e[common,:,m],axis=0)/common.sum())
                            np.testing.assert_allclose(expected,old['methods'][name]['response'],rtol=1e-12,atol=1e-12)
                    # Explicit secondary sample: native/ResNet/high-resolution observed shapes.
                    observed_common=np.all(good[:,:,[0,2,3]],axis=(1,2))
                    ref=matrix(truth[observed_common].mean(axis=0))
                    values={}
                    for m,name in enumerate(METHODS):
                        same_sample_ok=bool(good[observed_common,:,m].all())
                        values[name]=dict(same_sample_available=same_sample_ok,
                            success_by_branch=good[:,:,m].sum(axis=0).tolist(),
                            status_counts={str(int(s)):int(n) for s,n in zip(*np.unique(status[:,:,m],return_counts=True))})
                        if same_sample_ok:
                            mean=e[observed_common,:,m].mean(axis=0)
                            rr=matrix(mean)
                            values[name].update(response=rr.tolist(),relative_trace_response=float(np.trace(rr)/np.trace(ref)),
                                zero_shear_mean_e=mean[0].tolist())
                    cells.append(dict(psf_fwhm=psf,half_light_radius=radius,snr=snr,
                        attempted=len(e),all_method_common=int(common.sum()),
                        observed_common=int(observed_common.sum()),truth_response=ref.tolist(),methods=values))
    report=dict(complete=True,original_report_sha256=sha(RUN/'report.json'),
        measurements_sha256=sha(RUN/'measurements.npz'),source_sha256=sha(__file__),
        original_primary_comparison_reproduced=True,checkpoint_unchanged=True,
        secondary_comparison='Added after the primary run: compare native, ResNet, and target observed moments on their joint five-branch success sample. Show native REGAUSS only where it succeeds on exactly that entire same sample; no substitution or clipping.',
        reason='All native REGAUSS measurements fail for FWHM 0.45 arcsec; the original four-method intersection is empty there. Preserve that primary failure and display the separately valid observed-shape comparison.',
        attempted_base_objects=sum(c['attempted'] for c in cells),attempted_shear_images=5*sum(c['attempted'] for c in cells),
        original_all_method_common=sum(c['all_method_common'] for c in cells),
        observed_common=sum(c['observed_common'] for c in cells),cells=cells)
    output=OUT/'shear_response_review_v01.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(report)
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles=[('native_observed','Native observed','0.4','o'),
            ('native_regauss','Native + known PSF correction','#228833','^'),
            ('resnet_observed','ResNet observed','#4477aa','s'),
            ('target_observed','High-resolution control','#cc8833','D')]
    fig,axes=plt.subplots(2,4,figsize=(13.8,6.3),sharex=True,sharey=True)
    for i,psf in enumerate([.45,.60]):
        for j,snr in enumerate([None,20,50,100]):
            ax=axes[i,j]
            rows=[r for r in report['cells'] if r['psf_fwhm']==psf and r['snr']==snr]
            for name,label,color,marker in styles:
                ax.plot([r['half_light_radius'] for r in rows],
                    [r['methods'][name].get('relative_trace_response',np.nan) for r in rows],
                    color=color,marker=marker,label=label,lw=1.5,ms=5)
            ax.axhline(1,color='black',lw=.8,ls=':')
            ax.set_title(('Noiseless' if snr is None else f'Nominal S/N {snr}')+f'\nPSF FWHM {psf:.2f} arcsec',fontsize=10)
            ax.text(.98,.04,'Common: '+', '.join(f"{r['observed_common']}/{r['attempted']}" for r in rows),
                    transform=ax.transAxes,ha='right',fontsize=7)
            if i==0:
                ax.text(.04,.94,'PSF correction failed',transform=ax.transAxes,va='top',fontsize=8,color='#228833')
            ax.set_xticks([.18,.35,.65])
            ax.set_ylim(.2,1.18)
            ax.grid(alpha=.15)
            if i==1: ax.set_xlabel('Half-light radius [arcsec]')
            if j==0: ax.set_ylabel('Response / intrinsic response')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,fontsize=10)
    fig.text(.5,.012,'Synthetic pilot • common observed-shape sample • no calibrated shear or uncertainty claim',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.93))
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'shear_response_review_v01.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
