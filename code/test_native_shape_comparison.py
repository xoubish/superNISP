"""Check contour probability semantics and paired native-relative errors."""
import unittest
import numpy as np
from add_native_shape_comparison import density_thresholds, native_gains


class NativeComparisonTest(unittest.TestCase):
    def test_gaussian_probability_contours(self):
        axis = np.linspace(-6, 6, 601)
        xx, yy = np.meshgrid(axis, axis)
        density = np.exp(-(xx**2 + yy**2)/2) / (2*np.pi)
        thresholds = density_thresholds(density)
        # For a circular Gaussian, enclosed p implies density (1-p)/(2*pi).
        np.testing.assert_allclose(thresholds, (1-np.array([.68,.95]))/(2*np.pi), rtol=.005)
        for threshold, probability in zip(thresholds, [.68,.95]):
            mass = density[density >= threshold].sum()/density.sum()
            self.assertAlmostEqual(mass, probability, delta=.001)

    def test_gains_preserve_sign_and_pairing(self):
        ref = np.arange(24).reshape(12,2)/100
        errors = np.arange(1,25).reshape(12,2)/100
        measurements = {'target': {'e': ref}, 'native': {'e': ref+errors},
                        'better': {'e': ref+.5*errors}, 'worse': {'e': ref+1.5*errors}}
        gains = native_gains(measurements, np.ones(12,bool), np.repeat([0,1,2],4),
                             ['native','better','worse'], repeats=100)
        for name, expected in [('native',0),('better',50),('worse',-50)]:
            np.testing.assert_allclose(gains[name]['reduction_percent'], expected, atol=1e-12)
            np.testing.assert_allclose(gains[name]['ci95_percent'], expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
