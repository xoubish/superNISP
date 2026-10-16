"""Mask-conditioned SR3 UNet with v prediction and jump-aware deterministic DDIM.

Historical SR3 sources are preserved. This fresh-training path uses exact zero
terminal signal, no output clipping, and takes no HR data at inference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from diffusion.model_sr3 import SR3UNet


def cosine_alpha_bar(timesteps):
    if timesteps < 2:
        raise ValueError('At least two diffusion timesteps required')
    t=torch.arange(1,timesteps+1,dtype=torch.float64)/timesteps
    alpha=(torch.cos((t+.008)/1.008*math.pi/2)/math.cos(.008/1.008*math.pi/2)).square()
    alpha[-1]=0.  # Exact terminal prior: pure N(0,I), including during training.
    return alpha.float()


def masked_mean(values,mask):
    mask=mask.bool()
    count=mask.sum((1,2,3))
    if (count==0).any():
        raise ValueError('Empty target mask')
    return (torch.where(mask,values,0.).sum((1,2,3))/count).mean()


def velocity_to_x0_eps(xt,velocity,alpha):
    a,b=alpha.sqrt(),(1-alpha).sqrt()
    return a*xt-b*velocity, b*xt+a*velocity


def ddim_step(xt,velocity,alpha_current,alpha_next):
    """Eta=0 DDIM using BOTH endpoints; alpha_next=1 is the final clean image."""
    x0,eps=velocity_to_x0_eps(xt,velocity,alpha_current)
    return alpha_next.sqrt()*x0+(1-alpha_next).sqrt()*eps


class CorrectedDiffusion(nn.Module):
    def __init__(self,width=32,timesteps=1000):
        super().__init__()
        self.unet=SR3UNet(in_channels=1,cond_channels=2,base_channels=width,
                         channel_mults=(1,2,4),time_emb_dim=256)
        self.register_buffer('alpha_bar',cosine_alpha_bar(timesteps))
        self.timesteps=timesteps

    def condition(self,nisp,valid):
        clean=torch.where(valid,nisp,0.)
        return torch.cat([F.interpolate(clean,scale_factor=5,mode='bilinear',align_corners=False),
                          F.interpolate(valid.float(),scale_factor=5,mode='nearest')],1)

    def training_loss(self,nisp,nisp_valid,target,target_valid,t,noise,x0_weight=.1):
        if target.shape[-2:]!=tuple(5*s for s in nisp.shape[-2:]):
            raise ValueError('Expected exact 5x paired geometry; no silent target resize')
        target=torch.where(target_valid,target,0.)
        alpha=self.alpha_bar[t,None,None,None]
        a,b=alpha.sqrt(),(1-alpha).sqrt()
        xt=a*target+b*noise
        velocity=self.unet(xt,t,self.condition(nisp,nisp_valid))
        truth=a*noise-b*target
        x0,_=velocity_to_x0_eps(xt,velocity,alpha)
        v_loss=masked_mean((velocity-truth).square(),target_valid)
        x0_loss=masked_mean((x0-target).abs(),target_valid)
        return v_loss+x0_weight*x0_loss,dict(v_mse=v_loss.detach(),x0_l1=x0_loss.detach())

    @torch.no_grad()
    def sample(self,nisp,nisp_valid,initial_noise,steps=50):
        if not 2<=steps<=self.timesteps:
            raise ValueError('Sampling steps must be between 2 and timesteps')
        expected=(len(nisp),1,5*nisp.shape[-2],5*nisp.shape[-1])
        if tuple(initial_noise.shape)!=expected:
            raise ValueError('Initial noise has wrong shape')
        condition=self.condition(nisp,nisp_valid)
        times=torch.linspace(self.timesteps-1,0,steps,device=nisp.device).round().long()
        xt=initial_noise.clone()
        for i,t in enumerate(times):
            batch_t=t.expand(len(nisp))
            velocity=self.unet(xt,batch_t,condition)
            next_alpha=self.alpha_bar[times[i+1]] if i+1<len(times) else xt.new_tensor(1.)
            xt=ddim_step(xt,velocity,self.alpha_bar[t],next_alpha)
        if not torch.isfinite(xt).all():
            raise FloatingPointError('Nonfinite generated image')
        return xt


def fixed_noise(rows,seed,device,side=205):
    """Object-specific streams make evaluation independent of batch partitioning."""
    return torch.stack([torch.randn((1,side,side),device=device,
        generator=torch.Generator(device=device).manual_seed((int(row)*1000003+seed)%(2**63-1)))
        for row in rows])
