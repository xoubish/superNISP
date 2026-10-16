"""Geometry, HSM failure handling, and crowding-indicator checks."""
import unittest
import numpy as np
import torch
from shape_diagnostics import diagnose_moments
from robust_shape_measurements import fixed_moments,adaptive_moments,nearby_peak_proxy,summary,HSM_REASONS


class RobustnessTests(unittest.TestCase):
    def gaussian(self,side,step,shift=0.):
        axis=(torch.arange(side,dtype=torch.float64)-(side-1)/2)*step
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        return torch.exp(-(xx-shift).square()/200-yy.square()/98)[None,None]

    def test_nominal_exact_and_center_shift(self):
        image=self.gaussian(205,1)
        mask=torch.ones_like(image,dtype=torch.bool)
        old=diagnose_moments(image,mask)
        new=fixed_moments(image,mask)
        torch.testing.assert_close(old['e'],new['e'],rtol=0,atol=0)
        self.assertTrue(torch.equal(old['good'],new['good']))
        shifted=fixed_moments(self.gaussian(205,1,2),mask,offset=(2.,0.))
        torch.testing.assert_close(new['e'],shifted['e'],atol=1e-10,rtol=0)

    def test_adaptive_known_shape_scales_and_mask(self):
        expected=(100-49)/(100+49)
        for side,step in [(205,1),(41,5)]:
            image=self.gaussian(side,step)
            mask=torch.ones_like(image,dtype=torch.bool)
            result=adaptive_moments(image,mask,step)
            self.assertTrue(result['good'][0],result['error'][0])
            np.testing.assert_allclose(result['e'][0],[expected,0],atol=2e-3)
            mask[0,0,side//2,side//2]=False
            result=adaptive_moments(image,mask,step)
            self.assertFalse(result['good'][0])
            self.assertTrue(result['flags'][0,1])
        image=torch.zeros((1,1,205,205),dtype=torch.float64)
        result=adaptive_moments(image,torch.ones_like(image,dtype=torch.bool))
        self.assertFalse(result['good'][0])

    def test_neighbor_capture_and_peak_proxy(self):
        image=self.gaussian(205,1,15)
        result=adaptive_moments(image,torch.ones_like(image,dtype=torch.bool))
        self.assertFalse(result['good'][0])
        native=torch.zeros(1,1,41,41)
        native[0,0,20,20]=10
        rms=torch.ones_like(native)
        valid=torch.ones_like(native,dtype=torch.bool)
        self.assertFalse(nearby_peak_proxy(native,rms,valid)[0])
        native[0,0,20,25]=8
        self.assertTrue(nearby_peak_proxy(native,rms,valid)[0])

    def test_hsm_reason_labels_not_moment_labels(self):
        measurements={m:dict(e=np.zeros((2,2)),good=np.ones(2,dtype=bool),flags=np.zeros((2,7),dtype=bool))
                      for m in ['jwst','native','bilinear','bicubic','sr']}
        measurements['sr']['good'][0]=False
        measurements['sr']['flags'][0,2]=True
        report=summary(measurements,np.array(['a','b']),HSM_REASONS)
        self.assertEqual(report['failure_reasons']['sr']['hsm_failure'],1)
        self.assertNotIn('nonpositive_flux',report['failure_reasons']['sr'])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
