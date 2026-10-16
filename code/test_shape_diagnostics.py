"""Scientific regression checks for the fixed validation diagnostics."""
import unittest
import numpy as np
import torch
from shape_diagnostics import diagnose_moments, interpolation, paired_summary, REASONS
from train_paired import shape_moments


class ShapeTests(unittest.TestCase):
    def test_target_matches_existing_diagnostic(self):
        gen=torch.Generator().manual_seed(8)
        images=torch.randn(5,1,205,205,generator=gen)+.1
        mask=torch.ones_like(images,dtype=torch.bool)
        mask[0,0,102,102]=False
        old_e,old_good=shape_moments(images,mask)
        result=diagnose_moments(images,mask)
        torch.testing.assert_close(result['e'],old_e,rtol=0,atol=0)
        self.assertTrue(torch.equal(result['good'],old_good))

    def test_same_angular_gaussian_and_failures(self):
        def gaussian(side,step):
            axis=(torch.arange(side,dtype=torch.float64)-(side-1)/2)*step
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            return torch.exp(-xx.square()/200-yy.square()/98)[None,None]
        high=gaussian(205,1)
        low=gaussian(41,5)
        a=diagnose_moments(high,torch.ones_like(high,dtype=torch.bool))
        b=diagnose_moments(low,torch.ones_like(low,dtype=torch.bool),5)
        torch.testing.assert_close(a['e'],b['e'],atol=.002,rtol=0)
        self.assertTrue(a['good'].all() and b['good'].all())
        invalid=diagnose_moments(-high,torch.ones_like(high,dtype=torch.bool))
        self.assertFalse(invalid['good'].any())
        self.assertTrue(invalid['flags'][0,REASONS.index('nonpositive_flux')])
        mask=torch.ones_like(high,dtype=torch.bool)
        mask[0,0,102,102]=False
        self.assertTrue(diagnose_moments(high,mask)['flags'][0,1])
        high[0,0,102,102]=float('nan')
        self.assertTrue(diagnose_moments(high,torch.ones_like(mask))['flags'][0,0])

    def test_interpolation_masks_and_constant_intensity(self):
        data=torch.full((1,1,41,41),7.)
        valid=torch.ones_like(data,dtype=torch.bool)
        for mode in ['bilinear','bicubic']:
            output,mask=interpolation(data,valid,mode)
            torch.testing.assert_close(output,torch.full_like(output,7.),atol=1e-5,rtol=0)
            self.assertTrue(mask.all())
        valid[0,0,20,20]=False
        for mode in ['bilinear','bicubic']:
            _,mask=interpolation(data,valid,mode)
            self.assertFalse(mask[0,0,102,102])
            self.assertTrue(mask[0,0,10,10])

    def test_common_population_and_paired_block_interval(self):
        n=8
        truth=np.full((n,2),.2)
        measurements={m:dict(e=truth.copy(),good=np.ones(n,dtype=bool),flags=np.zeros((n,len(REASONS)),dtype=bool))
                      for m in ['jwst','native','bilinear','bicubic','sr']}
        for method in ['native','bilinear','bicubic']:
            measurements[method]['e']+=.2
        measurements['sr']['e']+=.1
        measurements['sr']['good'][0]=False
        measurements['sr']['flags'][0,2]=True
        measurements['native']['good'][1]=False
        measurements['native']['flags'][1,1]=True
        summary=paired_summary(measurements,np.array(['a','a','b','b','c','c','d','d']))
        self.assertEqual(summary['common_successes'],6)
        self.assertEqual(summary['reference_baselines_eligible'],7)
        self.assertEqual(summary['sr_failures_on_eligible'],1)
        compare=summary['methods']['native']['sr_comparison']
        self.assertAlmostEqual(compare['improvement_percent'],50)
        np.testing.assert_allclose(compare['ci95_percent'],[50,50])
        for values in measurements.values(): values['good'][:]=False
        self.assertEqual(paired_summary(measurements,np.array(['a']*n))['methods'],{})


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
