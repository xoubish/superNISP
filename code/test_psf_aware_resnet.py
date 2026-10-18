"""Controls for PSF preprocessing, warm-start equivalence, and trainability."""
import copy
import unittest
import numpy as np
import torch

from psf_aware_resnet import PSFAwareResNet,inverse_feature,initialize_parent,is_context
from residual_baseline import EuclidToJWSTSuperResolution
from prepare_psf_features import kernel_proxy


class PSFControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_inverse_identity_dc_and_rotation(self):
        torch.manual_seed(13)
        x=torch.randn(2,1,41,41,dtype=torch.float64)
        delta=torch.zeros(2,1,11,11,dtype=torch.float64);delta[:,:,5,5]=1
        torch.testing.assert_close(inverse_feature(x,delta),x,atol=1e-12,rtol=1e-12)
        yy,xx=torch.meshgrid(torch.arange(-5,6,dtype=torch.float64),torch.arange(-5,6,dtype=torch.float64),indexing='ij')
        kernel=torch.exp(-((xx-.2)**2/1.4+(yy+.1)**2/2.1))[None,None].repeat(2,1,1,1)
        torch.testing.assert_close(inverse_feature(torch.ones_like(x),kernel),torch.ones_like(x),atol=1e-12,rtol=1e-12)
        for k in range(4):
            def transform(t): return torch.rot90(t,k,(-2,-1)).flip(-1)
            torch.testing.assert_close(inverse_feature(transform(x),transform(kernel)),transform(inverse_feature(x,kernel)),atol=1e-11,rtol=1e-11)

    def test_native_kernel_geometry(self):
        y,x=np.mgrid[-30:31,-30:31]
        a=np.exp(-(x*x/72+y*y/162))
        k=kernel_proxy(a,np.eye(2),6*np.eye(2),[30,30])
        expected=a[::6,::6]; expected/=expected.sum()
        np.testing.assert_allclose(k,expected,atol=1e-12)
        rotated=kernel_proxy(a,np.eye(2),6*np.array([[0,-1],[1,0]]),[30,30])
        np.testing.assert_allclose(rotated,np.rot90(k),atol=1e-12)

    def test_parent_equivalence_and_context_gradient(self):
        torch.manual_seed(7)
        base=EuclidToJWSTSuperResolution(num_rrdb=1,features=8)
        opt=torch.optim.Adam(base.parameters(),lr=1e-5)
        x=torch.randn(1,1,41,41)
        base(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        parent=dict(model=copy.deepcopy(base.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
        packed=torch.cat([x,torch.ones_like(x),torch.ones_like(x),torch.zeros_like(x)],1)
        packed[:,3,20,20]=1
        for branch in ['conditioned','deconv']:
            model=PSFAwareResNet(branch,num_rrdb=1,features=8)
            optimizer,extended=initialize_parent(model,parent)
            torch.testing.assert_close(model(packed),base(x),rtol=0,atol=0)
            model(packed).square().mean().backward()
            self.assertGreater(float(model.context_adapter[-1].weight.grad.abs().sum()),0)
            optimizer.step()
            self.assertTrue(torch.isfinite(model(packed)).all())
            # Loading/updating the new optimizer must not mutate the parent.
            for state in parent['optimizer']['state'].values():self.assertEqual(float(state['step']),1)
            for state in extended['optimizer']['state'].values():self.assertEqual(float(state['step']),1)


if __name__=='__main__': unittest.main()
