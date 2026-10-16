"""Catalog-only parent selection for Figure 2; no image or model loading."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


def in_mosaics(ra, dec, headers):
    """Union of image pixel rectangles, evaluated through their celestial WCS.

    This is geometric coverage, not a pixel-quality or exposure-mask selection.
    """
    inside = np.zeros(len(ra), dtype=bool)
    finite = np.isfinite(ra) & np.isfinite(dec)
    for header in headers:
        indices = np.flatnonzero(finite & ~inside)
        wcs = WCS(header).celestial
        for start in range(0, len(indices), 100000):
            rows = indices[start:start + 100000]
            x, y = wcs.all_world2pix(ra[rows], dec[rows], 0)
            inside[rows] |= ((x >= -.5) & (x < header['NAXIS1'] - .5)
                             & (y >= -.5) & (y < header['NAXIS2'] - .5))
    return inside


def load_parent(catalog_path, data_directory):
    """Galaxy-class objects within the NISP/JWST geometric footprint overlap.

    No sample magnitude, redshift, size, spatial-buffer or pixel-quality cuts.
    Missing-value filtering for each plotted quantity is handled separately.
    """
    catalog_path, data_directory = Path(catalog_path).expanduser(), Path(data_directory)
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f'Full COSMOS2020 catalog not found: {catalog_path}. '
            'Set COSMOS_CATALOG in the notebook on the machine holding the FITS file. '
            'The preselected candidates.csv is not a substitute for this parent sample.')
    manifest = json.loads((data_directory / 'manifest.json').read_text())
    groups = {'nisp': [], 'jwst': []}
    header_paths = []
    for name, relative in manifest['source_headers'].items():
        instrument = ('nisp' if name.startswith('EUC_NIR_') else
                      'jwst' if name.startswith('mosaic_nircam_f115w_') else None)
        if instrument is None:
            continue
        path = data_directory / relative
        saved = json.loads(path.read_text())
        groups[instrument].append(fits.Header.fromstring(saved['SCI']))
        header_paths.append(path)
    if not all(groups.values()):
        raise ValueError('Both NISP and JWST source headers are required for the parent footprint.')
    with fits.open(catalog_path, memmap=True) as hdul:
        table = hdul[1].data
        galaxy_rows = np.flatnonzero(np.asarray(table['lp_type']) == 0)
        ra = np.asarray(table['ALPHA_J2000'][galaxy_rows], dtype=float)
        dec = np.asarray(table['DELTA_J2000'][galaxy_rows], dtype=float)
        overlap = in_mosaics(ra, dec, groups['nisp']) & in_mosaics(ra, dec, groups['jwst'])
        rows = galaxy_rows[overlap]
        parent = pd.DataFrame({
            'id_classic': np.asarray(table['ID'][rows], dtype=np.int64),
            **{key: np.asarray(table[key][rows], dtype=float)
               for key in ['ACS_F814W_MAG', 'ez_z_phot']},
        })
        catalog_rows = len(table)
    if parent.empty or not parent.id_classic.is_unique:
        raise ValueError('Parent must contain nonempty, unique galaxy IDs.')
    metadata = dict(
        catalog_rows=catalog_rows, galaxy_class_rows=len(galaxy_rows), parent_rows=len(parent),
        selection='lp_type == 0; within union(NISP pixel rectangles) intersect union(JWST pixel rectangles)',
        footprint='Geometric WCS coverage from sample source headers; no exposure or pixel-quality masks.',
        source_mosaics={key: len(value) for key, value in groups.items()},
    )
    return parent, metadata, header_paths


def valid_values(frame, column, sentinels):
    """Remove explicit catalog missing-value codes, not the sample science cuts."""
    values = frame[column].to_numpy(dtype=float)
    valid = np.isfinite(values) & ~np.isin(values, sentinels)
    if column == 'ez_z_phot':
        valid &= values >= 0  # Negative photo-z values are invalid estimates.
    return values[valid]


def probability_density(values, bins):
    """Normalize by all valid values, preserving probability outside plot limits."""
    if not len(values):
        raise ValueError('No valid catalog measurements for a plotted distribution.')
    counts, _ = np.histogram(values, bins=bins)
    return counts / (len(values) * np.diff(bins))
