"""Reference-only segmentation for the local Figure 6 mask experiment.

Detection is performed on the full JWST cutout, before cropping. No model
prediction or reconstruction error enters detection or the mask parameters.
"""
import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes


PROTOCOL = dict(
    gaussian_sigma_pixels=1., gaussian_truncate=4.,
    sky_annulus_pixels=[70., 95.], sky_clip_sigma=3., sky_clip_iterations=5,
    threshold_sigma=[1.5, 2., 3.], primary_threshold_sigma=2.,
    minimum_connected_pixels=10, connectivity=8, dilation_radius_pixels=3,
    rule='Smooth the full signed JWST reference with a normalized Gaussian; '
         'estimate median and sigma-clipped standard deviation from valid '
         'smoothed pixels at radius 70–95. Detect all 8-connected regions with '
         'at least 10 pixels above median + threshold*sigma; fill holes and '
         'dilate by a radius-3 disk. Intersect with common validity and crop '
         'to the central 105x105 pixels only after detection. Background is '
         'the complement within common validity. No per-model masks.',
    reference='https://photutils.readthedocs.io/en/stable/user_guide/segmentation.html')


def reference_masks(target, valid, thresholds=(1.5, 2., 3.)):
    """Return source masks and measured detection noise, using JWST only."""
    target = np.asarray(target, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if target.shape != (205, 205) or valid.shape != target.shape:
        raise ValueError('Expected full 205x205 reference and validity')
    if not np.isfinite(target[valid]).all():
        raise ValueError('Nonfinite reference on valid support')
    weight = gaussian_filter(valid.astype(float), 1., truncate=4., mode='constant')
    smooth = gaussian_filter(np.where(valid, target, 0.), 1., truncate=4., mode='constant')
    smooth = np.divide(smooth, weight, out=np.zeros_like(smooth), where=weight > 0)
    reliable = weight > .99999
    yy, xx = np.indices(target.shape)
    radius = np.hypot(yy-102, xx-102)
    sky = reliable & (radius >= 70) & (radius <= 95)
    if sky.sum() < 100:
        raise ValueError('Insufficient valid reference sky for detection noise')
    _, median, sigma = sigma_clipped_stats(smooth[sky], sigma=3., maxiters=5)
    if not np.isfinite([median, sigma]).all() or sigma <= 0:
        raise ValueError('Invalid reference detection noise')
    yy, xx = np.mgrid[-3:4, -3:4]
    footprint = xx**2 + yy**2 <= 9
    masks, components = [], []
    for threshold in thresholds:
        # Empty detections are retained explicitly, never replaced by a
        # model-dependent fallback or zero error.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No sources were found.*')
            segments = detect_sources(smooth, median+threshold*sigma, npixels=10,
                                      connectivity=8, mask=~reliable)
        detected = np.zeros_like(valid) if segments is None else segments.data > 0
        masks.append(binary_dilation(binary_fill_holes(detected), structure=footprint) & valid)
        components.append(0 if segments is None else int(segments.nlabels))
    return np.asarray(masks), dict(sky_median=float(median), sky_sigma=float(sigma),
                                 sky_pixels=int(sky.sum()), components=components)
