"""Validation-only fixed moment diagnostics and paired spatial-block summaries.

Coordinates are in target pixels for every sampling. These signed, weighted
moments are diagnostic ellipticities, not calibrated gravitational shear.
"""
import numpy as np
import torch
from torch.nn import functional as F

METHODS = ('native','bilinear','bicubic','sr')
REASONS = ('nonfinite_input','incomplete_aperture','nonpositive_flux',
           'nonpositive_qxx','nonpositive_qyy','nonpositive_determinant','nonfinite_shape')


def diagnose_moments(images, valid, pixel_step=1.):
    side = images.shape[-1]
    axis = (torch.arange(side,device=images.device,dtype=images.dtype)-(side-1)/2)*pixel_step
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    aperture = xx.square()+yy.square()<=25**2
    weight = torch.exp(-(xx.square()+yy.square())/(2*10**2))*aperture
    finite = torch.isfinite(images[:,0])
    missing = ~(valid[:,0]|~aperture).all((-2,-1))
    nonfinite = (~finite & aperture & valid[:,0]).any((-2,-1))
    clean = torch.where(finite & valid[:,0],images[:,0],0.)
    flux = clean*weight
    total = flux.sum((-2,-1))
    safe = torch.where(total.abs()>1e-12,total,torch.ones_like(total))
    cx,cy = (flux*xx).sum((-2,-1))/safe,(flux*yy).sum((-2,-1))/safe
    dx,dy = xx-cx[:,None,None],yy-cy[:,None,None]
    qxx = (flux*dx.square()).sum((-2,-1))/safe
    qyy = (flux*dy.square()).sum((-2,-1))/safe
    qxy = (flux*dx*dy).sum((-2,-1))/safe
    trace = qxx+qyy
    denominator = torch.where(trace.abs()>1e-12,trace,torch.ones_like(trace))
    e = torch.stack(((qxx-qyy)/denominator,2*qxy/denominator),-1)
    flags = torch.stack((nonfinite,missing,~(total>1e-12),~(qxx>0),~(qyy>0),
                         ~(qxx*qyy>qxy.square()),~torch.isfinite(e).all(-1)),-1)
    return dict(e=e,good=~flags.any(-1),flags=flags,flux=total*pixel_step**2,
                centroid=torch.stack((cx,cy),-1),qxx=qxx,qyy=qyy,qxy=qxy,
                determinant=qxx*qyy-qxy.square(),trace=trace)


def interpolation(images, valid, mode):
    output = F.interpolate(images,size=(205,205),mode=mode,align_corners=False)
    if mode=='bilinear':
        mask=F.interpolate(valid.float(),size=(205,205),mode=mode,align_corners=False)>.99999
    elif mode=='bicubic':
        # Signed cubic weights make interpolating a mask unsafe. Require the
        # entire clamped 4x4 input stencil, including zero-weight neighbors.
        base = torch.floor((torch.arange(205,device=images.device)+.5)/5-.5).long()
        mask = torch.ones_like(output,dtype=torch.bool)
        for dy in (-1,0,1,2):
            for dx in (-1,0,1,2):
                y=(base+dy).clamp(0,40)
                x=(base+dx).clamp(0,40)
                mask &= valid[:,:,y[:,None],x[None,:]]
    else:
        raise ValueError(mode)
    return output,mask


def snr_proxy(images,rms,valid):
    axis=(torch.arange(41,device=images.device,dtype=images.dtype)-20)*5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    weight=torch.exp(-(xx.square()+yy.square())/200)*(xx.square()+yy.square()<=625)
    signal=(images[:,0]*weight*valid[:,0]).sum((-2,-1))
    variance=((rms[:,0]*weight*valid[:,0]).square()).sum((-2,-1))
    return signal/variance.sqrt().clamp_min(1e-12)


def paired_summary(measurements,blocks,selection=None,bootstrap=1000,seed=20261016):
    n=len(blocks)
    selection=np.ones(n,dtype=bool) if selection is None else np.asarray(selection,dtype=bool)
    eligible=selection & measurements['jwst']['good']
    for method in METHODS[:-1]:
        eligible &= measurements[method]['good']
    common=eligible & measurements['sr']['good']
    report=dict(objects=int(selection.sum()),reference_baselines_eligible=int(eligible.sum()),
                common_successes=int(common.sum()),sr_failures_on_eligible=int((eligible & ~measurements['sr']['good']).sum()),
                success_counts={m:int((selection & values['good']).sum()) for m,values in measurements.items()},
                failure_reasons={m:{reason:int((selection & values['flags'][:,j]).sum()) for j,reason in enumerate(REASONS)} for m,values in measurements.items()},
                methods={},bootstrap_replicates=bootstrap)
    if not common.any():
        return report
    reference=measurements['jwst']['e'][common]
    errors={m:measurements[m]['e'][common]-reference for m in METHODS}
    errors['round_control']=-reference
    squared={m:(error**2).mean(-1) for m,error in errors.items()}
    for method,error in errors.items():
        report['methods'][method]=dict(component_rmse=float(np.sqrt(squared[method].mean())),
            bias=error.mean(0).tolist(),component_scatter=error.std(0).tolist(),
            vector_error_above_0_2_fraction=float((np.linalg.norm(error,axis=1)>.2).mean()))
    labels,inverse=np.unique(np.asarray(blocks)[common],return_inverse=True)
    report['spatial_blocks']=len(labels)
    counts=np.bincount(inverse,minlength=len(labels))
    sums={m:np.bincount(inverse,weights=s,minlength=len(labels)) for m,s in squared.items()}
    # Resample blocks with replacement, keeping all methods/objects paired.
    # A single spatial field and just 18 validation blocks limit interpretation.
    draws=np.random.default_rng(seed).integers(len(labels),size=(bootstrap,len(labels))) if bootstrap else None
    for baseline in ('native','bilinear','bicubic','round_control'):
        baseline_rmse=report['methods'][baseline]['component_rmse']
        improvement=100*(1-report['methods']['sr']['component_rmse']/baseline_rmse) if baseline_rmse else None
        comparison=dict(improvement_percent=improvement,ci95_percent=None)
        if bootstrap and len(labels)>=2:
            baseline_sums=sums[baseline][draws].sum(1)
            sr_sums=sums['sr'][draws].sum(1)
            # Counts cancel in the paired RMSE ratio.
            usable=baseline_sums>0
            values=100*(1-np.sqrt(sr_sums[usable]/baseline_sums[usable]))
            if len(values):
                comparison['ci95_percent']=np.percentile(values,[2.5,97.5]).tolist()
            comparison['usable_bootstrap_replicates']=int(usable.sum())
        report['methods'][baseline]['sr_comparison']=comparison
    return report


def binned_summaries(measurements,blocks,properties,bootstrap=1000):
    specs={
        'nisp_weighted_snr_proxy':[-np.inf,5,10,20,50,np.inf],
        'ACS_F814W_MAG':[-np.inf,22,23,24,25,np.inf],
        'ez_z_phot':[0,.5,1,2,3],
        'FLUX_RADIUS':[4,8,16,32,500],
    }
    result={}
    for name,edges in specs.items():
        values=np.asarray(properties[name],dtype=float)
        entries=[]
        for lower,upper in zip(edges[:-1],edges[1:]):
            subset=np.isfinite(values)&(values>=lower)&(values<upper)
            if not subset.any():
                continue
            entry=paired_summary(measurements,blocks,subset,bootstrap=bootstrap)
            entry['range']=f'[{lower}, {upper})'
            entries.append(entry)
        result[name]=entries
    return result
