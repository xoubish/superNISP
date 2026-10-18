"""Build input-only train/validation PSF associations and discrete kernel proxies."""
import json
from pathlib import Path
import shutil

import galsim
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import map_coordinates

from audit_shape_psfs import ROOT,DATA,sha


def kernel_proxy(array,psf_cd,native_cd,centroid,side=11):
    """Sample the empirical effective PSF; do not integrate another pixel response.

    Use the supplied linear CD matrices as local bases. This is an explicit
    proxy convention, not a claim of alias-free physical deconvolution.
    """
    axis=np.arange(side)-(side-1)/2
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    coords=np.linalg.solve(psf_cd,native_cd)@np.stack([xx.ravel(),yy.ravel()])
    coords+=np.asarray(centroid)[:,None]
    sampled=map_coordinates(array,[coords[1],coords[0]],order=3,mode='constant',cval=0.,prefilter=True).reshape(side,side)
    if not np.isfinite(sampled).all() or sampled.sum()<=0:
        raise ValueError('Invalid sampled PSF')
    return sampled/sampled.sum()


def main():
    out=ROOT/'runs/psf-features-v01'
    out.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(DATA/'objects.csv')
    # Drop reserved rows before joining/processing: no test image reads.
    frame=frame[frame.split.isin(['train','validation'])].copy()
    manifest=json.loads((DATA/'manifest.json').read_text())
    records=[]; kernels=[]; names=sorted(frame.psf_image.unique())
    for name in names:
        psf_path=DATA/'nisp_psf'/name
        expected=next(r['sha256'] for r in manifest['source_files'] if Path(r['path']).name==name)
        assert sha(psf_path)==expected
        images=frame.loc[frame.psf_image==name,'nisp_image'].unique()
        assert len(images)==1
        header_path=DATA/'source_headers'/(images[0]+'.json')
        hdr=fits.Header.fromstring(json.loads(header_path.read_text())['SCI'])
        with fits.open(psf_path) as hdus:
            array=hdus[1].data.astype(float); ph=hdus[1].header
            assert hdus[0].header['OVERSAMP']==6 and ph['POLNAXIS']==0
        native_cd=np.array([[hdr[f'CD{i}_{j}'] for j in [1,2]] for i in [1,2]])
        psf_cd=np.array([[ph[f'CD{i}_{j}'] for j in [1,2]] for i in [1,2]])
        fit=galsim.hsm.FindAdaptiveMom(galsim.ImageD(array),strict=True)
        center=[fit.moments_centroid.x-1,fit.moments_centroid.y-1]
        kernel=kernel_proxy(array,psf_cd,native_cd,center)
        kernels.append(kernel)
        records.append(dict(filename=name,sha256=expected,image=images[0],
            source_header_sha256=sha(header_path),native_cd=native_cd.tolist(),psf_cd=psf_cd.tolist(),
            centered_at_psf_pixels_xy=center,kernel_sum=float(kernel.sum()),
            kernel_minimum=float(kernel.min()),native_psf_axis_transform=np.linalg.solve(psf_cd,native_cd).tolist(),
            train_objects=int(((frame.psf_image==name)&(frame.split=='train')).sum()),
            validation_objects=int(((frame.psf_image==name)&(frame.split=='validation')).sum())))
    arrays=dict(kernels=np.array(kernels,dtype='float32'))
    with h5py.File(DATA/'sample.h5','r') as f:
        for split in ['train','validation']:
            rows=f[split+'_indices'][:]
            subset=frame.set_index('row').loc[rows]
            assert (subset.split==split).all()
            ids=f['object_id'][rows]
            np.testing.assert_array_equal(ids,subset.id_classic)
            arrays.update({split+'_rows':rows,split+'_ids':ids,
                split+'_kernel_index':np.array([names.index(n) for n in subset.psf_image],dtype='int32')})
    np.savez_compressed(out/'features.npz',**arrays)
    report=dict(complete=True,objects_csv_sha256=sha(DATA/'objects.csv'),records=records,
        features_sha256=sha(out/'features.npz'),sources={n:sha(ROOT/'code'/n) for n in ['prepare_psf_features.py','audit_shape_psfs.py']},
        convention='Recenter the signed supplied PSF on its adaptive centroid; map native pixel-center offsets through inverse(PSF CD)*native CD; cubic sample 11x11 and normalize signed sum. No additional pixel convolution. Header CD axes are an explicit proxy assumption; no calibrated intrinsic-image claim.',
        inverse_feature='Native-grid regularized linear inverse proxy, lambda .05, DC gain 1, maximum transfer gain <=2.35, 20-pixel reflection pad. Original image, mask and RMS retained. No JWST information in model inputs.',
        limitations=['Only four nearly identical stack PSFs; cannot establish generalization across broad PSF variation.',
            'Empirical PSF/stack response and undersampling remain approximate; no physically exact deconvolution claim.',
            'No matched local JWST PSF supplied; target remains the observed resampled F115W image.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name in report['sources']:shutil.copy2(ROOT/'code'/name,out/name)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
