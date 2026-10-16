"""Catalog-only parent selection for Figure 2; no image or model loading."""
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def load_parent(catalog_path, magnitude_limit=25., sentinels=(-99, 99, -999, 999)):
    """Full COSMOS catalog with only a valid ACS F814W magnitude below the limit.

    No classification, footprint, redshift, size, buffer or quality selection.
    Invalid redshifts are excluded only from the redshift histogram, not here.
    """
    catalog_path = Path(catalog_path).expanduser()
    if not catalog_path.is_file():
        raise FileNotFoundError(f'Full COSMOS2020 catalog not found: {catalog_path}. '
                                'Set COSMOS_CATALOG to the full FITS catalog.')
    with fits.open(catalog_path, memmap=True) as hdul:
        table = hdul[1].data
        magnitude = np.asarray(table['ACS_F814W_MAG'], dtype=float)
        valid = np.isfinite(magnitude) & ~np.isin(magnitude, sentinels)
        rows = np.flatnonzero(valid & (magnitude < magnitude_limit))
        parent = pd.DataFrame({
            'id_classic': np.asarray(table['ID'][rows], dtype=np.int64),
            'ACS_F814W_MAG': magnitude[rows],
            'ez_z_phot': np.asarray(table['ez_z_phot'][rows], dtype=float),
        })
        class_counts = pd.Series(np.asarray(table['lp_type'][rows], dtype=np.int64)).value_counts().sort_index()
        metadata = dict(
            catalog_rows=len(table), parent_rows=len(parent),
            selection=f'ACS_F814W_MAG < {magnitude_limit:g}; valid magnitude required',
            magnitude_limit=float(magnitude_limit),
            invalid_catalog_magnitudes=int((~valid).sum()),
            valid_catalog_magnitudes_at_or_above_limit=int((valid & (magnitude >= magnitude_limit)).sum()),
            footprint='Full COSMOS2020 catalog; no mosaic-footprint restriction.',
            classification='All lp_type values included; no galaxy-class restriction.',
            parent_class_counts={str(key): int(value) for key, value in class_counts.items()},
        )
    if parent.empty or not parent.id_classic.is_unique:
        raise ValueError('Parent must contain nonempty, unique catalog IDs.')
    return parent, metadata


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
