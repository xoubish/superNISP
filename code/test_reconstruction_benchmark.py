"""Analytical metric and unequal-block bootstrap regression checks (CPU only)."""
import unittest
import numpy as np
import torch
from reconstruction_benchmark import pixel_errors, summarize
from source_masks import reference_masks


class BenchmarkTests(unittest.TestCase):
    def test_reference_mask_includes_neighbour_and_respects_invalid_pixels(self):
        rng = np.random.default_rng(314159)
        yy, xx = np.indices((205,205))
        target = rng.normal(0,.01,(205,205))
        for cy,cx in [(102,102),(110,135)]:
            target += .3*np.exp(-((yy-cy)**2+(xx-cx)**2)/18)
        valid = np.ones((205,205),bool)
        valid[70:80,70:80] = False
        target[~valid] = np.nan
        masks, diagnostic = reference_masks(target,valid)
        self.assertTrue(masks[:,102,102].all())
        self.assertTrue(masks[:,110,135].all())
        self.assertFalse(masks[:,~valid].any())
        self.assertTrue((~masks[2] | masks[1]).all())
        self.assertTrue((~masks[1] | masks[0]).all())
        self.assertGreater(diagnostic['sky_sigma'],0)

    def test_source_background_errors_partition_squared_and_absolute_error(self):
        target = torch.zeros(1,1,2,3)
        prediction = torch.tensor([[[[-2.,1.,3.],[4.,-5.,6.]]]])
        valid = torch.ones_like(target,dtype=torch.bool)
        source = torch.tensor([[[[True,False,False],[False,True,False]]]])
        whole = pixel_errors(prediction,target,valid)
        a = pixel_errors(prediction,target,source)
        b = pixel_errors(prediction,target,valid & ~source)
        self.assertAlmostEqual(a['mae_mjy_sr'].item(),3.5)
        self.assertAlmostEqual(a['rmse_mjy_sr'].item(),np.sqrt(14.5))
        for metric,power in [('mae_mjy_sr',1),('rmse_mjy_sr',2)]:
            combined = (a[metric]**power*2+b[metric]**power*4)/6
            self.assertAlmostEqual(combined.item(),(whole[metric]**power).item())

    def test_signed_masked_errors_and_invalid_support(self):
        target = torch.tensor([[[[-2., 2.], [float('nan'), 4.]]]])
        prediction = torch.tensor([[[[-1., -1.], [float('nan'), 6.]]]])
        valid = torch.tensor([[[[True, True], [False, True]]]])
        metrics = pixel_errors(prediction, target, valid)
        self.assertAlmostEqual(metrics['mae_mjy_sr'].item(), 2.)
        self.assertAlmostEqual(metrics['rmse_mjy_sr'].item(), np.sqrt(14/3))
        self.assertEqual(metrics['valid_pixels'].item(), 3)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            pixel_errors(prediction, target, torch.zeros_like(valid))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            pixel_errors(prediction, target, torch.ones_like(valid))

    def test_paired_gains_and_object_weighting(self):
        # Unequal populations per block must not become equal block means.
        values = np.array([1., 3., 12.])
        methods = {name: {metric: values*factor for metric in ['mae_mjy_sr','rmse_mjy_sr']}
                   for name, factor in [('bilinear',1),('half',.5),('same',1)]}
        report = summarize(methods, ['a','a','b'], repeats=500)
        self.assertAlmostEqual(report['methods']['bilinear']['mae_mjy_sr']['mean'],16/3)
        for metric in ['mae_mjy_sr','rmse_mjy_sr']:
            self.assertEqual(report['methods']['half'][metric]['gain_vs_bilinear_percent'],50.)
            np.testing.assert_allclose(report['methods']['half'][metric]['gain_ci95_percent'],[50,50])
            np.testing.assert_allclose(report['methods']['same'][metric]['gain_ci95_percent'],[0,0])
        with self.assertRaisesRegex(ValueError,'unmatched'):
            summarize(methods, ['a','b'])


if __name__ == '__main__':
    unittest.main()
