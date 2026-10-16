"""Extract a real paired field for Figure 1; no model or GPU dependencies."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def locate(root, name, instrument, override=None):
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    root = Path(root).expanduser()
    folders = [root, root/'data'/instrument/'cosmos', root/instrument/'cosmos',
               root/instrument, root/'cosmos']
    names = [name, name.removesuffix('.gz')] if name.endswith('.gz') else [name]
    for folder in folders:
        for filename in names:
            if (folder/filename).is_file():
                return folder/filename
    raise FileNotFoundError(f'{name} not found under {root}. Set MOSAIC_ROOT or the explicit band path.')


def sky(image, valid):
    yy, xx = np.indices(image.shape)
    cy, cx = (np.asarray(image.shape)-1)/2
    border = (np.abs(xx-cx) > cx*.5) | (np.abs(yy-cy) > cy*.5)
    values = image[valid & border]
    if len(values) < 100:
        raise ValueError('Insufficient valid border pixels to estimate the sky.')
    _, median, sigma = sigma_clipped_stats(values, sigma=3, maxiters=5)
    if not np.isfinite(median+sigma):
        raise ValueError('Nonfinite background estimate.')
    return float(median)


def extract_region(data, mosaic_root, cache_stem, block='4:2', native_size=164,
                   nisp_path=None, jwst_path=None, force=False, center_object_id=None):
    """Native NISP field and a five-times-finer, identical-footprint JWST field.

    Cached arrays can be plotted later without the large mosaics. Cache identity
    includes geometry, input catalog/manifest hashes and this extractor's hash.
    An optional catalog object ID centres the field on that galaxy within the
    specified training or validation block. Set force=True after changing a
    source mosaic at the same path.
    """
    data, stem = Path(data), Path(cache_stem)
    signature = dict(block=block, native_size=int(native_size), factor=5,
                     center_rule=('catalog object position' if center_object_id is not None
                                  else 'centre of the selected spatial block'),
                     center_object_id=(int(center_object_id) if center_object_id is not None else None),
                     catalog_sha256=digest(data/'objects.csv'),
                     manifest_sha256=digest(data/'manifest.json'),
                     extractor_sha256=digest(__file__))
    npz, meta_path = stem.with_suffix('.npz'), stem.with_suffix('.json')
    if not force and npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['signature'] == signature and digest(npz) == meta['npz_sha256']:
            with np.load(npz, allow_pickle=False) as saved:
                return dict(saved), meta
    catalog = pd.read_csv(data/'objects.csv')
    selected = catalog[catalog.spatial_block == block]
    if selected.empty or selected.split.nunique() != 1 or selected.split.iloc[0] not in {'train', 'validation'}:
        raise ValueError('Choose a populated training or validation block.')
    region_split = selected.split.iloc[0]
    manifest = json.loads((data/'manifest.json').read_text())
    split = manifest['spatial_split']
    ra_origin, dec_origin = split['origin_deg']
    bx, by = map(int, block.split(':'))
    cosine = np.cos(np.deg2rad(dec_origin))
    width = split['block_arcsec']/3600
    ra_edges = ra_origin + np.array([bx, bx+1])*width/cosine
    dec_edges = dec_origin + np.array([by, by+1])*width
    center = np.array([ra_edges.mean(), dec_edges.mean()])
    distance = ((selected.ra-center[0])*cosine)**2 + (selected.dec-center[1])**2
    anchor = selected.loc[distance.idxmin()]
    if center_object_id is not None:
        matches = selected[selected.id_classic == int(center_object_id)]
        if len(matches) != 1:
            raise ValueError('The centre object must belong to the selected block.')
        anchor = matches.iloc[0]
        center = np.array([anchor.ra, anchor.dec])
    npath = locate(mosaic_root, anchor.nisp_image, 'NISP-Y', nisp_path)
    jpath = locate(mosaic_root, anchor.jwst_image, 'JWST', jwst_path)
    print('Extracting real mosaic pixels:', npath.name, jpath.name, flush=True)
    with fits.open(npath, memmap=True, do_not_scale_image_data=True) as hdus:
        science = hdus['SCI']
        if science.header.get('DATAUNIT', '').strip() != 'microJy':
            raise ValueError('Expected NISP SCI in microJy per pixel.')
        wcs = WCS(science.header)
        if wcs.has_distortion:
            raise ValueError('This extractor requires the supplied distortion-free NISP WCS.')
        xy = wcs.all_world2pix([center], 0)[0]
        cut = Cutout2D(science.data, xy, native_size, wcs=wcs, mode='strict', copy=True)
        slices = cut.slices_original
        nisp = (np.asarray(cut.data, dtype=np.float64)*science.header.get('BSCALE', 1)
                + science.header.get('BZERO', 0))
        rms_hdu, dq_hdu = hdus['RMS'], hdus['DQ']
        rms = (np.asarray(rms_hdu.data[slices], dtype=np.float64)*rms_hdu.header.get('BSCALE', 1)
               + rms_hdu.header.get('BZERO', 0))
        dq = (np.asarray(dq_hdu.data[slices], dtype=np.int64)*dq_hdu.header.get('BSCALE', 1)
              + dq_hdu.header.get('BZERO', 0))
        nvalid = (dq == 0) & np.isfinite(nisp) & np.isfinite(rms) & (rms > 0) & (rms < 1e7)
    cw = cut.wcs
    edge = native_size-.5
    corners = cw.all_pix2world([[-.5, -.5], [edge, -.5], [edge, edge], [-.5, edge]], 0)
    if not (np.all((corners[:, 0] > ra_edges[0]) & (corners[:, 0] < ra_edges[1]))
            and np.all((corners[:, 1] > dec_edges[0]) & (corners[:, 1] < dec_edges[1]))):
        raise ValueError('Requested field crosses the selected spatial-block boundary.')
    mid = (native_size-1)/2
    local = cw.all_pix2world([[mid, mid], [mid+1, mid], [mid, mid+1]], 0)
    delta = local[1:]-local[0]
    delta[:, 0] = ((delta[:, 0]+180) % 360-180)*np.cos(np.deg2rad(local[0, 1]))
    pixel_sr = abs(np.linalg.det(delta))*(np.pi/180)**2
    nisp *= 1e-12/pixel_sr
    # Pixel-edge mapping preserves the native stamp footprint exactly.
    size = native_size*5
    yy, xx = np.mgrid[:size, :size]
    ra, dec = cw.all_pix2world((xx+.5)/5-.5, (yy+.5)/5-.5, 0)
    with fits.open(jpath, memmap=True) as hdus:
        science = hdus[0]
        if science.header.get('BUNIT', '').strip() != 'MJy/sr':
            raise ValueError('Expected JWST science mosaic in MJy/sr.')
        sx, sy = WCS(science.header).all_world2pix(ra, dec, 0)
        x0, y0 = max(0, int(np.floor(sx.min()))-1), max(0, int(np.floor(sy.min()))-1)
        x1 = min(science.shape[1], int(np.ceil(sx.max()))+2)
        y1 = min(science.shape[0], int(np.ceil(sy.max()))+2)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Requested field is outside the selected JWST mosaic.')
        patch = np.asarray(science.data[y0:y1, x0:x1], dtype=np.float32)
        valid = np.isfinite(patch) & (patch != 0)
        coords = [sy-y0, sx-x0]
        support = map_coordinates(valid.astype(float), coords, order=1, mode='constant', cval=0, prefilter=False)
        jwst = map_coordinates(np.where(valid, patch, 0), coords, order=1, mode='constant', cval=0, prefilter=False)
        jvalid = support >= 1-1e-6
    if min(nvalid.mean(), jvalid.mean()) < .95:
        raise ValueError('Less than 95% coverage: choose another validation block or matching mosaic.')
    backgrounds = dict(nisp=sky(nisp, nvalid), jwst=sky(jwst, jvalid))
    arrays = dict(nisp=np.where(nvalid, nisp-backgrounds['nisp'], 0).astype('f4'),
                  jwst=np.where(jvalid, jwst-backgrounds['jwst'], 0).astype('f4'),
                  nisp_valid=nvalid, jwst_valid=jvalid)
    compass = {}
    for label, offset in [('N', [0, 1/3600]), ('E', [1/3600/np.cos(np.deg2rad(local[0, 1])), 0])]:
        compass[label] = (cw.all_world2pix([local[0]+offset], 0)[0]-[mid, mid]).tolist()
    print('Hashing source mosaics for export provenance...', flush=True)
    sources = {str(path.resolve()): digest(path) for path in [npath, jpath]}
    for path in [npath, jpath]:
        expected = next((r['sha256'] for r in manifest['source_files']
                         if Path(r['path']).name == path.name), None)
        if expected and sources[str(path.resolve())] != expected:
            raise ValueError(f'Source checksum differs from the sample manifest: {path.name}')
    meta = dict(signature=signature, split=region_split, spatial_block=block,
                center_object_id=signature['center_object_id'],
                units='sky-subtracted MJy/sr', native_size=native_size,
                center_ra_dec_deg=local[0].tolist(), corners_ra_dec_deg=corners.tolist(),
                block_ra_edges_deg=ra_edges.tolist(), block_dec_edges_deg=dec_edges.tolist(),
                nisp_pixel_scale_arcsec=float(np.sqrt(pixel_sr)*180/np.pi*3600),
                compass_native_pixel_vectors=compass, sources_sha256=sources,
                background_mjy_sr=backgrounds,
                valid_fractions=dict(nisp=float(nvalid.mean()), jwst=float(jvalid.mean())),
                jwst_resampling='bilinear surface brightness; all contributors valid',
                limitations='Different bands/PSFs; JWST finite/nonzero coverage proxy; display only.')
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    meta['npz_sha256'] = digest(npz)
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    return arrays, meta
