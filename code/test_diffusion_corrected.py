"""Analytic and integration checks, executed in tmux with online W&B by launcher."""
import unittest
import torch
from torch import nn
from diffusion_corrected import (CorrectedDiffusion,cosine_alpha_bar,ddim_step,
    velocity_to_x0_eps,masked_mean,fixed_noise)


class DiffusionTests(unittest.TestCase):
    def test_oracle_jumps_and_endpoint_unbounded(self):
        x0=torch.tensor([[[[-4.,2.5],[.3,9.]]]],dtype=torch.float64)
        eps=torch.tensor([[[[.2,-.7],[1.,-.4]]]],dtype=torch.float64)
        alpha=cosine_alpha_bar(100).double()
        for times in [list(range(99,-1,-1)),[99,73,29,0]]:
            xt=eps.clone()
            for i,t in enumerate(times):
                a=alpha[t]
                # Analytic oracle: v=sqrt(alpha)*epsilon-sqrt(1-alpha)*x0.
                v=a.sqrt()*eps-(1-a).sqrt()*x0
                nxt=alpha[times[i+1]] if i+1<len(times) else a.new_tensor(1.)
                xt=ddim_step(xt,v,a,nxt)
                torch.testing.assert_close(xt,nxt.sqrt()*x0+(1-nxt).sqrt()*eps)
            torch.testing.assert_close(xt,x0)

    def test_schedule_and_velocity_roundtrip(self):
        alpha=cosine_alpha_bar(1000)
        self.assertEqual(alpha[-1].item(),0.)
        self.assertTrue(bool((alpha[1:]<alpha[:-1]).all()))
        x0,noise=torch.randn(3,1,5,5),torch.randn(3,1,5,5)
        a=alpha[torch.tensor([0,500,999]),None,None,None]
        xt=a.sqrt()*x0+(1-a).sqrt()*noise
        v=a.sqrt()*noise-(1-a).sqrt()*x0
        recovered,eps=velocity_to_x0_eps(xt,v,a)
        torch.testing.assert_close(recovered,x0)
        torch.testing.assert_close(eps,noise)

    def test_masked_loss_and_seed_partition(self):
        values=torch.tensor([[[[2.,float('nan')]]]],requires_grad=True)
        valid=torch.tensor([[[[True,False]]]])
        loss=masked_mean(values,valid)
        self.assertEqual(loss.item(),2.)
        loss.backward()
        self.assertEqual(values.grad[0,0,0,1],0.)
        with self.assertRaises(ValueError): masked_mean(values,~torch.ones_like(valid))
        all_noise=fixed_noise([1,20,3],77,'cpu',5)
        torch.testing.assert_close(all_noise,torch.cat([fixed_noise([1],77,'cpu',5),fixed_noise([20,3],77,'cpu',5)]))
        self.assertFalse(torch.equal(all_noise,fixed_noise([1,20,3],78,'cpu',5)))

    def test_sampler_calls_both_endpoints_and_masks_condition(self):
        model=CorrectedDiffusion(width=8,timesteps=20)
        class Oracle(nn.Module):
            def __init__(self,alpha):
                super().__init__(); self.alpha=alpha; self.times=[]
            def forward(self,x,t,cond):
                self.times.append(int(t[0]))
                a=self.alpha[t,None,None,None]
                target=torch.full_like(x,3.5)
                eps=(x-a.sqrt()*target)/(1-a).sqrt()
                return a.sqrt()*eps-(1-a).sqrt()*target
        oracle=Oracle(model.alpha_bar)
        model.unet=oracle
        x=torch.zeros(1,1,5,5); valid=torch.ones_like(x,dtype=torch.bool)
        valid[:,:,0,0]=False; x[:,:,0,0]=float('nan')
        self.assertTrue(torch.isfinite(model.condition(x,valid)).all())
        noise=fixed_noise([5],11,'cpu',25)
        first=model.sample(x,valid,noise,4)
        self.assertEqual(oracle.times,[19,13,6,0])
        torch.testing.assert_close(first,torch.full_like(first,3.5),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(model.sample(x,valid,noise,4),first)

    def test_unet_geometry_gradient_and_unknown_targets(self):
        model=CorrectedDiffusion(width=8,timesteps=20)
        x=torch.randn(1,1,9,9); mask=torch.ones_like(x,dtype=torch.bool)
        y=torch.randn(1,1,45,45); valid=torch.ones_like(y,dtype=torch.bool)
        valid[:,:,:5,:5]=False
        noise=torch.randn_like(y); t=torch.tensor([19])
        loss,_=model.training_loss(x,mask,y,valid,t,noise)
        other=y.clone(); other[~valid]=float('nan')
        other_loss,_=model.training_loss(x,mask,other,valid,t,noise)
        torch.testing.assert_close(loss,other_loss)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(model.sample(x,mask,noise,2).shape,y.shape)


if __name__=='__main__':
    raise SystemExit('Run via check_diffusion_corrected.py inside tmux with online W&B')
