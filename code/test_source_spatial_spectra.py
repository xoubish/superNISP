"""Check exact source exclusion, including invalid pixels and mask holes."""
import unittest
import numpy as np
from source_spatial_spectra import windows
from spatial_spectra import transform


class SourceWindowTests(unittest.TestCase):
    def test_outside_source_invariance(self):
        rng = np.random.default_rng(62)
        image = rng.normal(size=(105, 105))
        source = np.zeros_like(image, dtype=bool)
        source[30:65, 35:75] = True
        source[15:25, 80:90] = True  # A separate neighbour is retained.
        source[40:45, 50:55] = False  # An internal masked hole is excluded.
        valid = np.ones_like(source)
        valid[50:55, 60:65] = False
        support = source & valid
        changed = image.copy()
        changed[~support] = np.nan
        for case, window in windows(valid, source).items():
            if not case.startswith('source'):
                continue
            with self.subTest(case=case):
                self.assertTrue(np.all(window[~support] == 0))
                self.assertTrue(np.all(window[support] > 0))
                self.assertTrue(np.all((window >= 0) & (window <= 1)))
                np.testing.assert_array_equal(transform(image, window),
                                              transform(changed, window))
                altered = image.copy()
                altered[20, 85] += 100
                self.assertFalse(np.array_equal(transform(image, window),
                                                transform(altered, window)))

    def test_taper_rises_inward(self):
        source = np.zeros((105, 105), dtype=bool)
        source[30:75, 30:75] = True
        weights = windows(np.ones_like(source), source)
        self.assertEqual(weights['source3'][29, 52], 0)
        np.testing.assert_allclose(weights['source3'][30:33, 52], [.25, .75, 1])
        for width in (3, 5):
            self.assertLess(weights[f'source{width}'][30, 52], weights['source1'][30, 52])


if __name__ == '__main__':
    unittest.main()
