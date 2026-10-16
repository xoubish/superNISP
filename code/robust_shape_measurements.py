"""Independent observed-shape checks; frozen choices, no PSF/shear correction."""
import numpy as np
import torch
from torch.nn import functional as F
import galsim
from shape_diagnostics import REASONS, paired_summary

HSM_REASONS=('nonfinite_input','incomplete_central_aperture','hsm_failure',
             'nonfinite_output','nonpositive_amplitude','nonpositive_size','centroid_shift_gt_10_target_pixels')


def fixed_moments(images,valid,pixel_step=1.,sigma=10.,radius=25.,offset=(0.,0.)):
    side=images.shape[-1]
    axis=(torch.arange(side,device=images.device,dtype=images.dtype)-(side-1)/2)*pixel_step
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    distance=(xx-offset[0]).square()+(yy-offset[1]).square()
    aperture=distance<=radius**2
    weight=torch.exp(-distance/(2*sigma**2))*aperture
    finite=torch.isfinite(images[:,0])
    missing=~(valid[:,0]|~aperture).all((-2,-1))
    nonfinite=(~finite & aperture & valid[:,0]).any((-2,-1))
    clean=torch.where(finite & valid[:,0],images[:,0],0.)
    flux=clean*weight
    total=flux.sum((-2,-1))
    safe=torch.where(total.abs()>1e-12,total,torch.ones_like(total))
    cx,cy=(flux*xx).sum((-2,-1))/safe,(flux*yy).sum((-2,-1))/safe
    dx,dy=xx-cx[:,None,None],yy-cy[:,None,None]
    qxx=(flux*dx.square()).sum((-2,-1))/safe
    qyy=(flux*dy.square()).sum((-2,-1))/safe
    qxy=(flux*dx*dy).sum((-2,-1))/safe
    trace=qxx+qyy
    denominator=torch.where(trace.abs()>1e-12,trace,torch.ones_like(trace))
    e=torch.stack(((qxx-qyy)/denominator,2*qxy/denominator),-1)
    flags=torch.stack((nonfinite,missing,~(total>1e-12),~(qxx>0),~(qyy>0),
                       ~(qxx*qyy>qxy.square()),~torch.isfinite(e).all(-1)),-1)
    return dict(e=e,good=~flags.any(-1),flags=flags)


def adaptive_moments(images,valid,pixel_step=1.,crop_target=105,guess_target=10.):
    """HSM in a matched angular crop; require valid central radius 25.

    Fit is masked, not zero-filled. Shape, centroid and size are adaptive.
    We retain native pixel units inside HSM and convert settings consistently.
    A fixed 10-target-pixel centroid displacement guard rejects neighbor capture.
    """
    array=images.detach().cpu().numpy()[:,0]
    masks=valid.detach().cpu().numpy()[:,0]
    width=int(crop_target/pixel_step)
    assert width%2==1
    edge=(array.shape[-1]-width)//2
    array=array[:,edge:edge+width,edge:edge+width]
    masks=masks[:,edge:edge+width,edge:edge+width]
    axis=(np.arange(width)-(width-1)/2)*pixel_step
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    aperture=xx**2+yy**2<=25**2
    result=dict(e=np.full((len(array),2),np.nan),good=np.zeros(len(array),bool),
                flags=np.zeros((len(array),7),bool),centroid=np.full((len(array),2),np.nan),
                sigma=np.full(len(array),np.nan),status=np.full(len(array),-999,dtype=int),
                error=np.full(len(array),'',dtype='U256'))
    params=galsim.hsm.HSMParams(max_ashift=15/pixel_step,max_amoment=8000/pixel_step**2)
    for i,(data,mask) in enumerate(zip(array,masks)):
        finite=np.isfinite(data)
        result['flags'][i,0]=bool((~finite & mask).any())
        result['flags'][i,1]=bool((~mask & aperture).any())
        image=galsim.ImageD(np.where(finite & mask,data,0.),scale=pixel_step)
        badpix=galsim.ImageI((~mask | ~finite).astype(np.int32))
        try:
            fit=galsim.hsm.FindAdaptiveMom(image,badpix=badpix,guess_sig=guess_target/pixel_step,
                                          guess_centroid=image.true_center,strict=False,hsmparams=params)
            result['status'][i]=fit.moments_status
            result['error'][i]=fit.error_message
            result['flags'][i,2]=fit.moments_status!=0 or bool(fit.error_message)
            if not result['flags'][i,2]:
                e=np.array([fit.observed_e1,fit.observed_e2])
                center=np.array([fit.moments_centroid.x-image.true_center.x,
                                 fit.moments_centroid.y-image.true_center.y])*pixel_step
                result['e'][i]=e
                result['centroid'][i]=center
                result['sigma'][i]=fit.moments_sigma*pixel_step
                result['flags'][i,3]=not np.isfinite(np.r_[e,center,fit.moments_sigma,fit.moments_amp]).all()
                result['flags'][i,4]=fit.moments_amp<=0
                result['flags'][i,5]=fit.moments_sigma<=0
                result['flags'][i,6]=np.linalg.norm(center)>10
        except (galsim.GalSimError,RuntimeError,ValueError) as exc:
            result['flags'][i,2]=True
            result['error'][i]=str(exc)[:256]
        result['good'][i]=not result['flags'][i].any()
    return result


def common_mask(measurements):
    return np.logical_and.reduce([m['good'] for m in measurements.values()])


def summary(measurements,blocks,reasons=REASONS,selection=None,bootstrap=1000):
    result=paired_summary(measurements,blocks,selection,bootstrap)
    subset=np.ones(len(blocks),bool) if selection is None else selection
    result['failure_reasons']={method:{reason:int((subset & values['flags'][:,j]).sum())
        for j,reason in enumerate(reasons)} for method,values in measurements.items()}
    return result


def nearby_peak_proxy(native,rms,valid):
    """Input-only crowding indicator, not a deblender or complete neighbor catalog."""
    axis=torch.arange(41,device=native.device)-20
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    annulus=(xx.square()+yy.square()>=3**2)&(xx.square()+yy.square()<=9**2)
    pooled=F.max_pool2d(native,3,stride=1,padding=1)
    peak=(native==pooled)&(native>5*rms)&valid&annulus
    return peak.flatten(1).any(1)


def scenarios():
    # Commit the complete suite before inspecting results; do not pick whichever
    # measurement settings produce the largest SR advantage.
    return {
        'fixed_nominal':{},'weight_sigma8':{'sigma':8.},'weight_sigma12':{'sigma':12.},
        'aperture20':{'radius':20.},'aperture30':{'radius':30.},
        'center_xplus2':{'offset':(2.,0.)},'center_xminus2':{'offset':(-2.,0.)},
        'center_yplus2':{'offset':(0.,2.)},'center_yminus2':{'offset':(0.,-2.)},
        'background_sr_plus':{'background':('sr',.1)},'background_sr_minus':{'background':('sr',-.1)},
        'background_jwst_plus':{'background':('jwst',.1)},'background_jwst_minus':{'background':('jwst',-.1)},
        'background_nisp_plus':{'background':('nisp',.1)},'background_nisp_minus':{'background':('nisp',-.1)},
        'hsm105':{'adaptive':True},'hsm155':{'adaptive':True,'crop_target':155},
        'hsm105_guess15':{'adaptive':True,'guess_target':15.},
        'input_noise_seed0':{'noise_seed':0},'input_noise_seed1':{'noise_seed':1},
        'input_noise_seed2':{'noise_seed':2},
    }
