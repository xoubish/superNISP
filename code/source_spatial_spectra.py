"""Strict source-supported Fourier windows for the Figure 6 diagnostic."""
import numpy as np
from spatial_spectra import taper, validity_taper, windows as full_windows


def windows(valid, source):
    """Identical model/reference windows, exactly zero outside detected sources.

    Tapers rise from the mask boundary inward. Unlike Gaussian smoothing of
    the mask, these weights cannot admit neighbouring background pixels.
    The 3-pixel taper is primary; 1 and 5 pixels test boundary sensitivity.
    """
    support = np.asarray(valid, dtype=bool) & np.asarray(source, dtype=bool)
    outer = taper(side=support.shape[0])
    result = {f'source{width}': outer * validity_taper(support, width=width)
              for width in (3, 1, 5)}
    previous = full_windows(valid, source)
    result.update(whole=previous['tukey50'], background=previous['background'])
    return result
