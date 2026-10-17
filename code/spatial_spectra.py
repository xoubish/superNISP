"""Paired, windowed radial spectra for observed-image reconstruction checks.

Power and signed cross-correlation are complementary: amplitude agreement
alone does not establish that structure is in the right place. No PSF
deconvolution or noise subtraction is performed.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal.windows import tukey


EDGES = np.array([1,2,3,4,5,6,8,10,13,17,22,28,35,43,52.5])/105
BANDS = {'large_15_30px':(1/30,1/15), 'intermediate_6_15px':(1/15,1/6),
         'fine_3_6px':(1/6,1/3), 'finest_2_3px':(1/3,.5)}


def taper(side=105, alpha=.5):
    one = tukey(side, alpha=alpha)
    return np.outer(one, one)


def validity_taper(valid, width=5):
    """Smoothly go to zero at invalid pixels; zero-pad for edge distances."""
    distance = distance_transform_edt(np.pad(valid,1))[1:-1,1:-1]
    return np.sin(.5*np.pi*np.minimum(distance/width,1))**2 * valid


def windows(valid, source):
    outer = taper()
    support = np.ones_like(valid,dtype=float) if valid.all() else validity_taper(valid)
    return {'tukey50':outer*support,
            'tukey25':taper(alpha=.25)*support,
            'hann':taper(alpha=1.)*support,
            'soft_source':outer*support*gaussian_filter(source.astype(float),3.),
            'background':outer*validity_taper(valid & ~source,width=3)}


def transform(image, window):
    image = np.asarray(image,dtype=float)
    window = np.asarray(window,dtype=float)
    if image.shape != window.shape or not np.isfinite(image[window>0]).all():
        raise ValueError('Invalid image/window')
    if window.sum() <= 0 or (window<0).any():
        raise ValueError('Invalid spectral window')
    clean = np.where(window>0,image,0.)
    mean = np.sum(clean*window)/window.sum()
    return np.fft.fft2((clean-mean)*window)/np.sqrt(np.sum(window**2))


def frequency_masks(side=105):
    freq = np.fft.fftfreq(side)
    radius = np.hypot(freq[:,None],freq[None,:])
    rings = [(radius>=a)&(radius<b) for a,b in zip(EDGES[:-1],EDGES[1:])]
    bands = [(radius>=a)&(radius<b) for a,b in BANDS.values()]
    masks = rings+bands
    if any(not mask.any() for mask in masks):
        raise ValueError('Empty Fourier bin')
    return masks


def paired_spectrum(prediction_fft, reference_fft, masks):
    pm = np.abs(prediction_fft)**2
    pj = np.abs(reference_fft)**2
    cross = np.real(prediction_fft*reference_fft.conj())
    power = np.array([pm[m].mean() for m in masks])
    reference = np.array([pj[m].mean() for m in masks])
    cross = np.array([cross[m].mean() for m in masks])
    if (power<=0).any() or (reference<=0).any():
        raise ValueError('Zero spectral power')
    correlation = cross/np.sqrt(power*reference)
    if (np.abs(correlation)>1+1e-12).any():
        raise ValueError('Cross-correlation outside Cauchy bound')
    return dict(power=power,reference=reference,cross=cross,
                log_power_ratio=np.log10(power/reference),
                correlation=np.clip(correlation,-1,1))


def block_summary(values, blocks, repeats=2000, seed=20261017):
    """Equal-object means; same spatial-block draws for every method/bin."""
    values = np.asarray(values,dtype=float)
    labels, inverse = np.unique(blocks,return_inverse=True)
    if len(labels)<2 or values.shape[0]!=len(blocks) or not np.isfinite(values).all():
        raise ValueError('Invalid paired summary population')
    counts = np.bincount(inverse)
    sums = np.stack([values[inverse==i].sum(0) for i in range(len(labels))])
    draws = np.random.default_rng(seed).integers(len(labels),size=(repeats,len(labels)))
    sampled = sums[draws].sum(1)/counts[draws].sum(1).reshape((-1,)+(1,)*(values.ndim-1))
    return dict(mean=values.mean(0).tolist(),ci95=np.percentile(sampled,[2.5,97.5],axis=0).T.tolist(),
                objects=len(blocks),spatial_blocks=len(labels))
