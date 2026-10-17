"""Analytical tests distinguish spectral amplitude, phase and masking."""
import unittest
import numpy as np
from spatial_spectra import transform, paired_spectrum, frequency_masks, block_summary


class SpectrumTests(unittest.TestCase):
    def test_identity_amplitude_and_sign(self):
        image=np.random.default_rng(4).normal(size=(105,105))
        window=np.ones_like(image); f=transform(image,window); masks=frequency_masks()
        for scale,ratio,correlation in [(1,1,1),(2,4,1),(-1,1,-1)]:
            result=paired_spectrum(transform(scale*image,window),f,masks)
            np.testing.assert_allclose(result['log_power_ratio'],np.log10(ratio),atol=1e-14)
            np.testing.assert_allclose(result['correlation'],correlation,atol=1e-14)

    def test_equal_power_wrong_phase(self):
        _,x=np.indices((105,105)); phase=2*np.pi*10*x/105
        a=np.cos(phase);b=np.sin(phase);window=np.ones_like(a)
        f=np.fft.fftfreq(105);radius=np.hypot(f[:,None],f[None,:]);mask=(radius>=9.5/105)&(radius<10.5/105)
        result=paired_spectrum(transform(a,window),transform(b,window),[mask])
        self.assertAlmostEqual(result['log_power_ratio'][0],0,places=12)
        self.assertAlmostEqual(result['correlation'][0],0,places=12)

    def test_invalid_pixels_do_not_enter_transform(self):
        image=np.random.default_rng(8).normal(size=(105,105));window=np.ones_like(image)
        window[:5]=0; changed=image.copy();changed[:5]=np.nan
        np.testing.assert_allclose(transform(image,window),transform(changed,window))
        with self.assertRaises(ValueError):transform(changed,np.ones_like(window))

    def test_block_weighting_and_paired_difference(self):
        values=np.array([[1.,3.],[3.,5.],[12.,14.]])
        result=block_summary(values,['a','a','b'],repeats=100)
        np.testing.assert_allclose(result['mean'],[16/3,22/3])
        difference=block_summary(values[:,1]-values[:,0],['a','a','b'],repeats=100)
        np.testing.assert_allclose(difference['ci95'],[2,2])


if __name__=='__main__':unittest.main()
