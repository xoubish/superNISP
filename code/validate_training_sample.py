"""Validate a portable corrected sample without its original mosaics."""
from pathlib import Path
import argparse
import json
import hashlib
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


def validate(directory, diagnostics=128):
    out = Path(directory)
    table = pd.read_csv(out/'objects.csv')
    rejected = pd.read_csv(out/'rejected.csv')
    manifest = json.loads((out/'manifest.json').read_text())
    assert table.id_classic.is_unique
    assert set(table.id_classic).isdisjoint(set(rejected.id_classic))
    assert len(table)+len(rejected) == manifest['candidates']
    assert np.array_equal(table.row,np.arange(len(table)))
    sky = SkyCoord(table.ra.to_numpy(),table.dec.to_numpy(),unit='deg')
    xyz = sky.cartesian.xyz.value.T
    pairs = cKDTree(xyz).query_pairs(2*np.sin(np.deg2rad(20/3600)/2),output_type='ndarray')
    cross = int(np.sum(table.split.to_numpy()[pairs[:,0]] != table.split.to_numpy()[pairs[:,1]])) if len(pairs) else 0
    assert cross == 0, f'{cross} cross-split neighbors within 20 arcsec'
    shifts, previews = [], []
    norm = json.loads((out/'normalization.json').read_text())
    with h5py.File(out/'sample.h5','r') as f:
        n=len(table)
        assert np.array_equal(f['object_id'][:],table.id_classic)
        for split in ['train','validation','test']:
            assert np.array_equal(f[f'{split}_indices'][:],np.flatnonzero(table.split==split))
        # Check every stored stamp, not merely the diagnostic subset.
        for start in range(0,n,128):
            for key in ['nisp','nisp_rms','jwst']:
                a=f[key][start:start+128]
                mask=f['jwst_valid' if key=='jwst' else 'nisp_valid'][start:start+128].astype(bool)
                assert np.isfinite(a).all(),key
                assert np.all(a[~mask]==0),key
            lm=f['nisp_valid'][start:start+128].astype(bool)
            hm=f['jwst_valid'][start:start+128].astype(bool)
            assert np.all(lm.mean(axis=(1,2))>=.95)
            assert np.all(hm.mean(axis=(1,2))>=.95)
            assert lm[:,18:23,18:23].all() and hm[:,90:115,90:115].all()
            assert (f['nisp_dq'][start:start+128][lm]==0).all()
        training=table[table.split=='train']
        assert np.isclose(norm['nisp_scale'],np.median(training.nisp_median_rms*1e-12/training.nisp_pixel_sr))
        assert np.isclose(norm['jwst_scale'],np.median(training.jwst_sky_sigma))
        ids=np.sort(np.random.default_rng(42).choice(n,min(diagnostics,n),replace=False))
        geometry=[]
        for i in ids:
            row=table.iloc[i]
            hdrs=json.loads((out/'source_headers'/f'{row.nisp_image}.json').read_text())
            w=WCS(fits.Header.fromstring(hdrs['SCI']))
            pix=w.all_world2pix(row.ra,row.dec,0)
            geometry.append(max(abs(pix[0]-row.nisp_x0-row.target_x_lr),abs(pix[1]-row.nisp_y0-row.target_y_lr)))
            lr=f['nisp'][i].astype(float)-row.nisp_background
            hr=f['jwst'][i].astype(float)-row.jwst_background
            lm=f['nisp_valid'][i].astype(bool)
            hm=f['jwst_valid'][i].astype(bool)
            if len(previews)<12:
                previews.append((i,lr,hr,lm,hm))
            # Exploratory alignment diagnostic only, no shift applied to data.
            # Gaussian matching is approximate and not a calibrated PSF model.
            if not (lm[7:34,7:34].all() and hm[35:170,35:170].all()):
                continue
            if np.max(lr[17:24,17:24])/row.nisp_median_rms < 10:
                continue
            low=gaussian_filter(hr,3.5).reshape(41,5,41,5).mean(axis=(1,3))
            from scipy.ndimage import shift
            def corr(offset):
                a=lr[10:31,10:31].ravel()
                b=shift(low,offset,order=1,mode='nearest')[10:31,10:31].ravel()
                return float(np.corrcoef(a,b)[0,1])
            fit=minimize(lambda z:-corr(z),[0.,0.],method='Powell',bounds=[(-2,2),(-2,2)])
            shifts.append(dict(row=int(i),object_id=int(row.id_classic),dy=float(fit.x[0]),dx=float(fit.x[1]),
                               correlation_zero=corr([0,0]),correlation_fit=-float(fit.fun),nisp_image=row.nisp_image))
        assert max(geometry)<1e-6
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(previews),4,figsize=(10,len(previews)*2.3),squeeze=False)
    for axes_row,(idx,lr,hr,lm,hm) in zip(axes,previews):
        for ax,img,title in zip(axes_row,[lr,hr,lm,hm],[f'NISP row {idx}','JWST on NISP WCS','NISP valid','JWST valid']):
            kwargs=dict(cmap='gray',origin='lower')
            if img.dtype==bool:
                kwargs.update(vmin=0,vmax=1)
            else:
                kwargs.update(vmin=np.percentile(img,5),vmax=np.percentile(img,99.7))
            ax.imshow(img,**kwargs);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'preview.png',dpi=120);plt.close(fig)
    report=dict(accepted=len(table),split_counts=table.split.value_counts().to_dict(),
                full_array_validation=True,cross_split_neighbors_within_20_arcsec=cross,
                max_catalog_position_roundtrip_error_pixels=float(max(geometry)),
                alignment_diagnostics=shifts,
                alignment_note='Approximate PSF-smoothed image correlation; not proof of subpixel astrometric accuracy; no shifts fitted into data.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='alignment_diagnostics'},indent=2))
    if shifts:
        a=pd.DataFrame(shifts)
        good=a[a.correlation_fit>.7]
        print('Alignment fits:',len(a),'with correlation >.7:',len(good))
        if len(good): print('Median shifts dy/dx in LR pixels:',good[['dy','dx']].median().to_dict())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--diagnostics',type=int,default=128)
    args=p.parse_args()
    validate(args.directory,args.diagnostics)
