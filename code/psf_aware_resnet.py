"""PSF/noise-conditioned extension of the frozen-compatible legacy ResNet.

The original image always reaches the original first convolution unchanged.
A zero-initialized context residual makes initialization exactly the parent.
The optional bounded inverse-filter feature is not a calibrated deconvolution.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution


def inverse_feature(image, kernel, regularization=.05):
    """DC-preserving Tikhonov inverse, reflect padding, <=2.35 Fourier gain.

    Image B,1,41,41; centered discrete effective-PSF proxy B,1,11,11.
    No division by small transfer values; signed intensities remain signed.
    """
    if regularization <= 0:
        raise ValueError('Positive regularization required')
    padded=F.pad(image,(20,20,20,20),mode='reflect')
    kernel=kernel/kernel.sum((-2,-1),keepdim=True)
    padded_kernel=F.pad(kernel,(35,35,35,35))
    transfer=torch.fft.rfft2(torch.fft.ifftshift(padded_kernel,dim=(-2,-1)))
    inverse=(1+regularization)*transfer.conj()/(transfer.abs().square()+regularization)
    filtered=torch.fft.irfft2(torch.fft.rfft2(padded)*inverse,s=(81,81))
    return filtered[:,:,20:61,20:61]


class PSFAwareResNet(EuclidToJWSTSuperResolution):
    def __init__(self, branch='conditioned',num_rrdb=8,features=64):
        super().__init__(num_rrdb=num_rrdb,features=features,head='legacy')
        if branch not in ['conditioned','deconv']:
            raise ValueError(branch)
        self.branch=branch
        self.psf_encoder=nn.Sequential(nn.Flatten(),nn.Linear(121,32),nn.LeakyReLU(.2),
                                       nn.Linear(32,16),nn.LeakyReLU(.2))
        self.context_adapter=nn.Sequential(nn.Conv2d(20,32,3,padding=1),nn.LeakyReLU(.2),
            nn.Conv2d(32,32,3,padding=1),nn.LeakyReLU(.2),nn.Conv2d(32,features,3,padding=1))
        nn.init.zeros_(self.context_adapter[-1].weight)
        nn.init.zeros_(self.context_adapter[-1].bias)

    def forward(self, packed):
        if packed.ndim!=4 or packed.shape[1:]!=(4,41,41):
            raise ValueError('Expected image, log RMS, mask, centered PSF: B,4,41,41')
        x=packed[:,:1]
        kernel=packed[:,3:4,15:26,15:26]
        code=self.psf_encoder(kernel).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,41,41)
        residual=inverse_feature(x,kernel)-x if self.branch=='deconv' else torch.zeros_like(x)
        residual=residual*packed[:,2:3]
        context=torch.cat([x,packed[:,1:3],residual,code],dim=1)
        features=self.conv_first(x)+self.context_adapter(context)
        trunk=features
        for block in self.rrdb_blocks:
            trunk=block(trunk)
        features=self.detail_enhancer(features+self.trunk_conv(trunk))
        for up in self.upsampling_blocks:
            features=up(features)
        return F.interpolate(self.final_conv(features),size=(205,205),mode='bilinear',align_corners=False)


def is_context(name):
    return name.startswith(('psf_encoder.','context_adapter.'))


def initialize_parent(model,parent):
    """Preserve parent Adam moments; initialize only new context parameters."""
    missing,unexpected=model.load_state_dict(parent['model'],strict=False)
    assert not unexpected and missing and all(is_context(n) for n in missing)
    original=[p for n,p in model.named_parameters() if not is_context(n)]
    new=[p for n,p in model.named_parameters() if is_context(n)]
    optimizer=torch.optim.Adam(original,lr=1e-5)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    group=optimizer.param_groups[0]
    optimizer.add_param_group(dict(params=new,lr=1e-4,betas=group['betas'],eps=group['eps'],
                                  weight_decay=group['weight_decay'],amsgrad=group['amsgrad']))
    extended=copy.deepcopy(parent)
    extended['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    extended['optimizer']=copy.deepcopy(optimizer.state_dict())
    return optimizer,extended


def context_schedule(optimizer,total_steps):
    for group,lr in zip(optimizer.param_groups,[1e-5,1e-4]):
        for name in ['initial_lr','max_lr','min_lr','base_momentum','max_momentum']:
            group.pop(name,None)
        group['lr']=lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:.1+.9*(1+math.cos(math.pi*min(step,total_steps)/total_steps))/2)


class PSFSample(PairedSample):
    def __init__(self,directory,split,features_path):
        if split not in ['train','validation']:
            raise ValueError('Reserved test objects are unavailable to redesign selection')
        super().__init__(directory,split,'linear')
        with np.load(features_path) as f:
            np.testing.assert_array_equal(f[split+'_rows'],self.indices)
            self.ids=f[split+'_ids']
            self.kernel_index=f[split+'_kernel_index']
            self.kernels=f['kernels'].astype('float32')

    def __getitem__(self,i):
        item=super().__getitem__(i)
        assert item['object_id']==self.ids[i]
        x=item['nisp']; valid=item['nisp_valid']
        rms=item['nisp_rms_linear']
        if not torch.isfinite(rms[valid]).all() or (rms[valid]<=0).any():
            raise ValueError('Invalid RMS on valid pixels')
        noise=torch.log1p(torch.where(valid,rms,0.))
        psf=torch.zeros_like(x)
        psf[:,15:26,15:26]=torch.from_numpy(self.kernels[self.kernel_index[i]])
        item['nisp']=torch.cat([x,noise,valid.to(x.dtype),psf],dim=0)
        return item
