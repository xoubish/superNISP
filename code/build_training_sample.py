"""Build portable NIR-Y/F115W pairs from local mosaics, without network access.

The NISP stamp is unchanged. JWST surface brightness is bilinearly sampled at
the sky positions of a grid exactly 5x finer than that stamp. Masks remain
instrument-specific. Existing arrays and checkpoints are never overwritten.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import time
from contextlib import ExitStack

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
LR_SIZE, FACTOR = 41, 5
HR_SIZE = LR_SIZE * FACTOR
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_position(ra, dec, seed=20261016, block=180., buffer=10.):
    """Fixed tangent-plane approximation, with conservative footprint buffer."""
    x = (ra - 150.) * np.cos(np.deg2rad(2.)) * 3600
    y = (dec - 2.) * 3600
    bx, by = int(np.floor(x / block)), int(np.floor(y / block))
    u, v = x % block, y % block
    inside = min(u, block-u, v, block-v) >= buffer
    key = f'{seed}:{bx}:{by}'.encode()
    draw = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
    split = 0 if draw < .8 else (1 if draw < .9 else 2)
    return split, f'{bx}:{by}', inside


def finer_wcs(wcs, factor=FACTOR):
    """Preserve pixel edges and footprint: LR x=(HR x+.5)/factor-.5."""
    if wcs.sip is not None or wcs.has_distortion:
        raise ValueError('Distorted NISP WCS needs explicit distortion handling')
    fine = wcs.deepcopy()
    fine.wcs.crpix = factor * wcs.wcs.crpix - (factor - 1) / 2
    if fine.wcs.has_cd():
        fine.wcs.cd = fine.wcs.cd / factor
    else:
        fine.wcs.cdelt = fine.wcs.cdelt / factor
    return fine


def resample_jwst(data, source_wcs, target_wcs, size=HR_SIZE):
    """Bilinear surface-brightness interpolation with strict support validity.

    Science-only JWST mosaics lack DQ/weight maps here. Nonfinite and exact
    zero native pixels are conservatively treated as uncovered. Every
    contributing source pixel must be valid. Do not infer a noise map.
    """
    yy, xx = np.mgrid[:size, :size]
    ra, dec = target_wcs.all_pix2world(xx, yy, 0)
    sx, sy = source_wcs.all_world2pix(ra, dec, 0)
    x0, y0 = int(np.floor(sx.min())) - 1, int(np.floor(sy.min())) - 1
    x1, y1 = int(np.ceil(sx.max())) + 2, int(np.ceil(sy.max())) + 2
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(data.shape[1], x1), min(data.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('JWST footprint outside mosaic')
    patch = np.asarray(data[y0:y1, x0:x1], dtype=np.float32)
    valid = np.isfinite(patch) & (patch != 0)
    coords = np.array([sy-y0, sx-x0])
    support = map_coordinates(valid.astype(np.float32), coords, order=1,
                              mode='constant', cval=0, prefilter=False)
    output = map_coordinates(np.where(valid, patch, 0), coords, order=1,
                             mode='constant', cval=0, prefilter=False)
    good = support >= 1 - 1.e-6
    output[~good] = 0
    return output.astype(np.float32), good


def sky_stats(image, valid, exclusion_fraction=.5):
    yy, xx = np.indices(image.shape)
    cy, cx = (np.array(image.shape)-1) / 2
    border = (np.abs(xx-cx) > cx*exclusion_fraction) | (np.abs(yy-cy) > cy*exclusion_fraction)
    pixels = image[valid & border]
    if len(pixels) < 100:
        raise ValueError('Too few valid sky pixels')
    _, med, sigma = sigma_clipped_stats(pixels, sigma=3, maxiters=5)
    if not np.isfinite(med+sigma) or sigma <= 0:
        raise ValueError('Invalid sky estimate')
    return float(med), float(sigma)


def native_cutout(hdul, coord):
    sci = hdul['SCI']
    cut = Cutout2D(sci.data, coord, LR_SIZE, wcs=WCS(sci.header), mode='strict', copy=True)
    sl = cut.slices_original
    rms = np.asarray(hdul['RMS'].data[sl], dtype=np.float32)
    dq_hdu = hdul['DQ']
    # FITS uint32 is stored as signed int32 plus BZERO; memmap raw storage.
    dq = (np.asarray(dq_hdu.data[sl], dtype=np.int64) * dq_hdu.header.get('BSCALE', 1)
          + dq_hdu.header.get('BZERO', 0)).astype(np.uint32)
    arr = np.asarray(cut.data, dtype=np.float32)
    valid = (dq == 0) & np.isfinite(arr) & np.isfinite(rms) & (rms > 0) & (rms < 1.e7)
    return cut, arr, rms, dq, valid


def candidate_catalog():
    candidate_path = ROOT / 'catalog/matched_cat_cosmos_2_20251124.csv'
    cosmos_path = ROOT / 'catalog/COSMOS2020_CLASSIC_R1_v2.2_p3.fits'
    cat = pd.read_csv(candidate_path)
    if not cat.id_classic.is_unique:
        raise ValueError('Duplicate candidate IDs')
    with fits.open(cosmos_path, memmap=True) as hdul:
        data = hdul[1].data
        ids = np.asarray(data['ID'])
        order = np.argsort(ids)
        pos = np.searchsorted(ids[order], cat.id_classic.to_numpy())
        if np.any(pos >= len(ids)):
            raise ValueError('Unknown catalog ID')
        rows = order[pos]
        if not np.array_equal(ids[rows], cat.id_classic.to_numpy()):
            raise ValueError('ID lookup failed')
        for key in ['ACS_F814W_MAG', 'FLUX_RADIUS', 'lp_type', 'ez_z_phot']:
            cat[key] = np.asarray(data[key][rows], dtype=float)
        if not (np.allclose(cat.ra, data['ALPHA_J2000'][rows], atol=1.e-9, rtol=0)
                and np.allclose(cat.dec, data['DELTA_J2000'][rows], atol=1.e-9, rtol=0)):
            raise ValueError('Catalog coordinate mismatch')
    selected = ((cat.lp_type == 0) & (cat.ACS_F814W_MAG < 25) & (cat.ez_z_phot > .01)
                & (cat.ez_z_phot < 3) & (cat.FLUX_RADIUS > 4) & (cat.FLUX_RADIUS < 500))
    if not selected.all():
        raise ValueError('Candidate list does not match documented selection')
    return cat, [candidate_path, cosmos_path]


def expand_nisp(path, cache):
    if path.suffix != '.gz':
        return path
    dest = cache / path.stem
    if not dest.exists():
        temporary = dest.with_suffix('.partial')
        print(f'Decompressing {path.name}', flush=True)
        with gzip.open(path, 'rb') as inp, open(temporary, 'wb') as out:
            shutil.copyfileobj(inp, out, length=8*1024**2)
        temporary.rename(dest)
    return dest


class Writer:
    def __init__(self, handle):
        self.handle = handle
        self.n = 0
        self.buffer = []
        self.shapes = {'nisp': (41,41), 'nisp_rms': (41,41), 'nisp_dq': (41,41),
                       'nisp_valid': (41,41), 'jwst': (205,205), 'jwst_valid': (205,205)}
        for key, shape in self.shapes.items():
            dtype = 'u1' if key.endswith('valid') else ('u4' if key == 'nisp_dq' else 'f4')
            handle.create_dataset(key, shape=(0,*shape), maxshape=(None,*shape),
                                  chunks=(1,*shape), dtype=dtype, compression='gzip',
                                  compression_opts=4, shuffle=True)

    def add(self, arrays):
        index = self.n + len(self.buffer)
        self.buffer.append(arrays)
        if len(self.buffer) == 32:
            self.flush()
        return index

    def flush(self):
        if not self.buffer:
            return
        end = self.n + len(self.buffer)
        for key in self.shapes:
            dataset = self.handle[key]
            dataset.resize(end, axis=0)
            dataset[self.n:end] = np.stack([row[key] for row in self.buffer])
        self.n = end
        self.buffer.clear()
        self.handle.flush()


def build(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    (out / 'source_headers').mkdir()
    (out / 'nisp_psf').mkdir()
    cache = ROOT / 'data/sample_build_cache'
    cache.mkdir(exist_ok=True)
    catalog, source_paths = candidate_catalog()
    if args.limit:
        # Random pilot across source pairs, not the first galaxies in a tile.
        catalog = catalog.sample(n=min(args.limit, len(catalog)), random_state=42).sort_values('id_classic')
    print(f'Candidates: {len(catalog)}', flush=True)
    catalog.to_csv(out / 'candidates.csv', index=False)
    files = {p.name: p for folder in ['data/NISP-Y/cosmos','data/JWST/cosmos'] for p in (ROOT / folder).iterdir()
             if p.is_file()
             and (p.name.endswith('.fits') or p.name.endswith('.fits.gz'))}
    records, rejected = [], []
    source_set = set(source_paths)
    source_headers = {}
    start = time.monotonic()
    with h5py.File(out / 'sample.h5', 'w') as h5:
        writer = Writer(h5)
        h5.attrs['format_version'] = 'supernisp-paired-v1'
        h5.attrs['nisp_unit'] = 'microJy/pixel'
        h5.attrs['jwst_unit'] = 'MJy/sr'
        h5.attrs['factor'] = FACTOR
        h5.attrs['jwst_interpolation'] = 'bilinear surface brightness; all contributing pixels valid'
        for nisp_name, ncat in catalog.groupby('nisp_image', sort=True):
            nisp_path = files[Path(nisp_name).name]
            source_set.add(nisp_path)
            local_path = expand_nisp(nisp_path, cache)
            print(f'Processing {nisp_path.name}: {len(ncat)} candidates', flush=True)
            with fits.open(local_path, memmap=True, do_not_scale_image_data=True) as nh:
                if str(nh['SCI'].header.get('DATAUNIT','')).strip() != 'microJy':
                    raise ValueError('Unexpected NISP science units')
                nw = WCS(nh['SCI'].header)
                pix_sr = float(proj_plane_pixel_area(nw) * (np.pi / 180)**2)
                psf_name = Path(ncat.iloc[0].psf_image).name
                if psf_name not in files:
                    raise FileNotFoundError(psf_name)
                psf_path = files[psf_name]
                shutil.copy2(psf_path, out / 'nisp_psf' / psf_name)
                source_set.add(psf_path)
                header_path = f'source_headers/{nisp_path.name}.json'
                (out / header_path).write_text(json.dumps({h.name:h.header.tostring() for h in nh}, indent=2))
                source_headers[nisp_path.name] = header_path
                for jwst_name, jcat in ncat.groupby('jwst_image', sort=True):
                    jwst_path = files[Path(jwst_name).name]
                    source_set.add(jwst_path)
                    with fits.open(jwst_path, memmap=True) as jh:
                        jscience = jh[0]
                        if jscience.header.get('BUNIT') != 'MJy/sr':
                            raise ValueError('Unexpected JWST units')
                        jw = WCS(jscience.header)
                        hp = f'source_headers/{jwst_path.name}.json'
                        if jwst_path.name not in source_headers:
                            (out / hp).write_text(json.dumps({'SCI':jscience.header.tostring()}, indent=2))
                            source_headers[jwst_path.name] = hp
                        for row in jcat.itertuples(index=False):
                            record = row._asdict()
                            split, block, inside = split_position(row.ra,row.dec)
                            if not inside:
                                rejected.append(dict(id_classic=row.id_classic,reason='spatial_buffer'))
                                continue
                            try:
                                coord = nw.all_world2pix(row.ra,row.dec,0)
                                cut, lr, rms, dq, lm = native_cutout(nh, tuple(float(x) for x in coord))
                                # The 10-arcsec buffer must enclose the entire stamp.
                                corners = np.array([[-20.5,-20.5],[-20.5,20.5],[20.5,-20.5],[20.5,20.5]])
                                radius = np.linalg.norm(corners @ nw.pixel_scale_matrix.T,axis=1).max()*3600+.3
                                if radius > 10:
                                    raise RuntimeError('Split buffer too small for source WCS')
                                hr, hm = resample_jwst(jscience.data,jw,finer_wcs(cut.wcs))
                                if lm.mean() < .95 or hm.mean() < .95:
                                    raise ValueError('coverage_below_95_percent')
                                if not lm[18:23,18:23].all() or not hm[90:115,90:115].all():
                                    raise ValueError('invalid_central_5_nisp_pixels')
                                lb, ls = sky_stats(lr,lm)
                                hb, hs = sky_stats(hr,hm)
                            except (ValueError, fits.verify.VerifyError) as error:
                                rejected.append(dict(id_classic=row.id_classic,reason=str(error)))
                                continue
                            index = writer.add(dict(nisp=np.where(lm,lr,0),nisp_rms=np.where(lm,rms,0),
                                                    nisp_dq=dq,nisp_valid=lm,jwst=hr,jwst_valid=hm))
                            x0,y0=cut.origin_original
                            # Local tangent-plane area at this object's native pixel.
                            cx,cy=map(float,coord)
                            sky=nw.all_pix2world([[cx,cy],[cx+1,cy],[cx,cy+1]],0)
                            delta=sky[1:]-sky[0]
                            delta[:,0]=(delta[:,0]+180)%360-180
                            delta[:,0]*=np.cos(np.deg2rad(sky[0,1]))
                            local_pix_sr=float(abs(np.linalg.det(delta))*(np.pi/180)**2)
                            record.update(row=index,split=SPLITS[split],spatial_block=block,nisp_x0=x0,nisp_y0=y0,
                                          target_x_lr=float(coord[0]-x0),target_y_lr=float(coord[1]-y0),
                                          nisp_pixel_sr=local_pix_sr,nisp_background=lb,nisp_sky_sigma=ls,
                                          nisp_median_rms=float(np.median(rms[lm])),jwst_background=hb,jwst_sky_sigma=hs,
                                          nisp_valid_fraction=float(lm.mean()),jwst_valid_fraction=float(hm.mean()))
                            # Portable file names; full original paths stay in candidates.csv.
                            record.update(nisp_image=nisp_path.name,jwst_image=jwst_path.name,psf_image=psf_name)
                            records.append(record)
                            if len(records) % 500 == 0:
                                print(f'Accepted {len(records)}; rejected {len(rejected)}; elapsed {time.monotonic()-start:.0f}s',flush=True)
            # Expanded cache is local only, reusable by pilot/full runs.
        writer.flush()
        table = pd.DataFrame(records)
        if len(table) == 0:
            raise RuntimeError('No accepted pairs')
        table.to_csv(out / 'objects.csv',index=False)
        pd.DataFrame(rejected, columns=['id_classic','reason']).to_csv(out / 'rejected.csv',index=False)
        h5.create_dataset('object_id',data=table.id_classic.to_numpy(dtype='i8'))
        h5.create_dataset('split',data=table.split.map({s:i for i,s in enumerate(SPLITS)}).to_numpy(dtype='u1'))
        for name in ['nisp_pixel_sr','nisp_background','nisp_median_rms','jwst_background','jwst_sky_sigma']:
            h5.create_dataset(name,data=table[name].to_numpy(dtype='f8'))
        for i, name in enumerate(SPLITS):
            h5.create_dataset(f'{name}_indices',data=np.flatnonzero(table.split.to_numpy()==name))
        train = table[table.split=='train']
        normalization = dict(nisp_scale=float(np.median(train.nisp_median_rms*1.e-12/train.nisp_pixel_sr)),
                             jwst_scale=float(np.median(train.jwst_sky_sigma)),
                             units='MJy/sr',background='per-stamp sigma-clipped outer region',
                             scale_fit='training stamps only',representation='unbounded affine; optional asinh(z/3) in loader')
        h5.attrs['normalization_json'] = json.dumps(normalization)
    (out / 'normalization.json').write_text(json.dumps(normalization,indent=2)+'\n')
    print('Hashing source files for provenance...',flush=True)
    inventory = []
    for p in sorted(source_set):
        inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                              sha256=None if args.limit else sha256(p)))
    manifest = dict(format_version='supernisp-paired-v1',build_date='2026-10-16',pilot=bool(args.limit),
                    candidates=len(catalog),accepted=len(table),split_counts=table.split.value_counts().to_dict(),
                    rejection_counts=pd.Series([r['reason'] for r in rejected]).value_counts().to_dict(),
                    source_files=inventory,source_headers=source_headers,
                    selection='lp_type==0; ACS_F814W_MAG<25; .01<ez_z_phot<3; 4<FLUX_RADIUS<500; existing dated coverage candidates',
                    quality='DQ==0; finite SCI and positive finite RMS<1e7; >=95% coverage each instrument; central 5x5 NISP /25x25 HR fully valid',
                    spatial_split=dict(seed=20261016,block_arcsec=180,buffer_arcsec=10,probabilities=[.8,.1,.1],origin_deg=[150,2]),
                    limitations=['JWST SCI-only: validity from finite nonzero pixels, no JWST RMS/DQ available.',
                                 'WCS registration only; residual astrometric offsets need validation.',
                                 'Different filters; no color correction or PSF deconvolution applied.',
                                 'Selected optical-bright COSMOS subset, not a NISP-complete galaxy sample.',
                                 'Noise correlations from input mosaics and JWST resampling are not modeled.'])
    (out / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'build_training_sample.py')
    for name in ['paired_sample.py','validate_training_sample.py']:
        shutil.copy2(ROOT/'code'/name,out/name)
    print(f'Finished {out}: {len(table)} pairs, {(out/"sample.h5").stat().st_size/1e9:.2f} GB',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0,help='Seeded pilot candidate count; zero builds full sample')
    build(parser.parse_args())
