"""Boundary and scaling checks for input-only S/N stratification."""
import unittest
import numpy as np
from analyze_shape_snr import bin_indices, input_snr


class ShapeSnrTest(unittest.TestCase):
    def test_bin_boundaries_and_nonfinite(self):
        values=[-1,0,4.999,5,10,20,50,500,np.nan,np.inf,-np.inf]
        np.testing.assert_array_equal(bin_indices(values),[0,0,0,1,2,3,4,4,-1,-1,-1])

    def test_weighting_and_unit_scaling(self):
        image=np.full((2,41,41),2.)
        rms=np.full_like(image,3.)
        valid=np.ones_like(image,dtype=bool)
        axis=np.arange(-20,21)*5
        yy,xx=np.meshgrid(axis,axis)
        w=np.exp(-(xx**2+yy**2)/200)*(xx**2+yy**2<=625)
        expected=2*w.sum()/(3*np.sqrt((w*w).sum()))
        np.testing.assert_allclose(input_snr(image,rms,valid),expected)
        np.testing.assert_allclose(input_snr(image*1e-4,rms*1e-4,valid),expected)
        np.testing.assert_allclose(input_snr(-image,rms,valid),-expected)

    def test_invalid_pixels_and_zero_variance(self):
        image=np.ones((1,41,41));rms=np.ones_like(image);valid=np.zeros_like(image,dtype=bool)
        valid[:,20,20]=True
        image[:,0,0]=np.nan
        np.testing.assert_allclose(input_snr(image,rms,valid),1)
        self.assertTrue(np.isnan(input_snr(image,rms*0,valid)[0]))


if __name__=='__main__':
    unittest.main()
