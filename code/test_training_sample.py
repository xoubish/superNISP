"""Geometric and mask regression tests for sample extraction."""
import unittest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from build_training_sample import finer_wcs, resample_jwst, native_cutout, split_position


def wcs(scale=.3,angle=16,handedness=1):
    w=WCS(naxis=2)
    a=np.deg2rad(angle)
    w.wcs.cd=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@np.diag([scale*handedness,scale])/3600
    w.wcs.crpix=[51,51];w.wcs.crval=[150,2];w.wcs.ctype=['RA---TAN','DEC--TAN']
    return w


class GeometryTests(unittest.TestCase):
    def test_fivefold_grid_preserves_edges_and_centers_both_parities(self):
        for sign in [-1,1]:
            w=wcs(handedness=sign); fine=finer_wcs(w)
            xy=np.array([[-.5,-.5],[0,0],[100,100],[250,250],[500.5,500.5]])
            expected=w.all_pix2world((xy+.5)/5-.5,0)
            np.testing.assert_allclose(fine.all_pix2world(xy,0),expected,rtol=0,atol=1e-10)

    def test_bilinear_constant_surface_brightness_and_independent_mask(self):
        w=wcs();target=w.deepcopy();target.wcs.crpix=[6,6]
        data=np.ones((101,101),dtype='f4')*7
        image,valid=resample_jwst(data,w,target,size=11)
        self.assertTrue(valid.all());np.testing.assert_allclose(image,7)
        data[50,50]=np.nan
        image,valid=resample_jwst(data,w,target,size=11)
        self.assertFalse(valid[5,5]);self.assertEqual(image[5,5],0)

    def test_rotated_gaussian_keeps_sky_location(self):
        source=wcs(scale=.06,angle=-20,handedness=-1)
        yy,xx=np.mgrid[:101,:101]
        ra,dec=source.all_pix2world(xx,yy,0)
        x=(ra-150)*np.cos(np.deg2rad(2))*3600;y=(dec-2)*3600
        data=np.exp(-((x-.3)**2+(y+.15)**2)/(.3**2*2)).astype('f4')
        target=wcs(scale=.06,angle=16,handedness=1)
        target.wcs.crpix=[26,26]
        pred,valid=resample_jwst(data,source,target,size=51)
        yy,xx=np.mgrid[:51,:51];ra,dec=target.all_pix2world(xx,yy,0)
        x=(ra-150)*np.cos(np.deg2rad(2))*3600;y=(dec-2)*3600
        expected=np.exp(-((x-.3)**2+(y+.15)**2)/(.3**2*2))
        self.assertLess(np.max(np.abs(pred[valid]-expected[valid])),.011)

    def test_named_dq_not_rms_controls_mask(self):
        w=wcs();science=np.ones((101,101),dtype='f4')
        rms=np.ones_like(science)*2 # Would be wrongly masked by the old RMS>=1 rule.
        dq=np.zeros((101,101),dtype='i4');dq[50,49]=1
        hdul=fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(science,w.to_header(),name='SCI'),
                          fits.ImageHDU(rms,name='RMS'),fits.ImageHDU(dq,name='DQ')])
        cut,arr,error,flags,valid=native_cutout(hdul,(50.,50.))
        self.assertEqual(valid.sum(),41*41-1)
        self.assertFalse(valid[20,19]);self.assertTrue(valid[20,21])
        np.testing.assert_array_equal(arr,1);np.testing.assert_array_equal(error,2)

    def test_spatial_split_reproducible_and_buffered(self):
        self.assertFalse(split_position(150.,2.)[2])
        ra=150+90/(3600*np.cos(np.deg2rad(2)))
        dec=2+90/3600
        self.assertTrue(split_position(ra,dec)[2])
        self.assertEqual(split_position(ra,dec),split_position(ra,dec))


if __name__=='__main__':unittest.main()
