"""Run integration checks in a persistent session with online W&B monitoring."""
import argparse
import ast
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import wandb

from paired_sample import masked_l1
from residual_baseline import EuclidToJWSTSuperResolution
from train_paired import (persistent_session_required, seed_everything, save_checkpoint,
                          restore_rng, paired_augment, shape_moments)
from smoke_test_sample import main as loader_smoke


def check_architecture():
    source = Path(__file__).with_name('claude_model_NIR_2.py').read_text()
    names = {'ResidualDenseBlock','DetailEnhancementModule','EuclidToJWSTSuperResolution'}
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name in names]
    scope = {'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<legacy classes>','exec'),scope)
    old = scope['EuclidToJWSTSuperResolution'](num_rrdb=8,features=64).cuda()
    new = EuclidToJWSTSuperResolution().cuda()
    new.load_state_dict(old.state_dict(),strict=True)
    assert sum(p.numel() for p in new.parameters()) == 2332177
    x = torch.randn(1,1,41,41,device='cuda')
    with torch.no_grad():
        torch.testing.assert_close(old(x),new(x),rtol=0,atol=0)
    for head in ['legacy','direct5']:
        model = EuclidToJWSTSuperResolution(head=head).cuda()
        y = model(x)
        assert y.shape == (1,1,205,205) and torch.isfinite(y).all()
        y.abs().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def check_mask_and_geometry():
    prediction = torch.zeros(2,1,4,4,device='cuda',requires_grad=True)
    target = torch.ones_like(prediction)
    valid = torch.ones_like(prediction,dtype=torch.bool)
    valid[:,:,0,0] = False
    target[:,:,0,0] = 1e20
    loss = masked_l1(prediction,target,valid)
    assert loss.item() == 1
    loss.backward()
    assert (prediction.grad[:,:,0,0] == 0).all()
    try:
        masked_l1(prediction,target,torch.zeros_like(valid))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty masks must fail')
    x = torch.arange(41*41).reshape(1,1,41,41).float()
    for seed in range(8):
        random.seed(seed)
        batch = paired_augment(dict(nisp=x,jwst=x.repeat_interleave(5,-1).repeat_interleave(5,-2),
                                    nisp_valid=x>10,nisp_rms_linear=x+1))
        torch.testing.assert_close(batch['nisp'],batch['jwst'][:,:,2::5,2::5])
        assert torch.equal(batch['nisp_valid'],batch['nisp']>10)
        torch.testing.assert_close(batch['nisp_rms_linear'],batch['nisp']+1)
    axis = torch.arange(205,device='cuda')-102
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    circular = torch.exp(-(xx.square()+yy.square())/100)[None,None]
    e,ok = shape_moments(circular,torch.ones_like(circular,dtype=torch.bool))
    assert ok.all() and e.abs().max()<1e-6
    _,ok = shape_moments(-circular,torch.ones_like(circular,dtype=torch.bool))
    assert not ok.any()


def check_resume():
    seed_everything(123)
    def setup():
        model = nn.Sequential(nn.Conv2d(1,25,3,padding=1),nn.PixelShuffle(5)).cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-4,total_steps=6)
        return model,optimizer,scheduler
    def step(model,opt,sched,generator):
        x = torch.randn(2,1,41,41,device='cuda')*random.random()+float(np.random.rand())
        x *= torch.rand((),generator=generator).item()
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
        sched.step()
    model,opt,sched = setup()
    generator = torch.Generator().manual_seed(8)
    step(model,opt,sched,generator)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'resume.pt'
        save_checkpoint(path,model,opt,sched,generator)
        step(model,opt,sched,generator)
        expected = {k:v.clone() for k,v in model.state_dict().items()}
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        restored,ropt,rsched = setup()
        restored.load_state_dict(checkpoint['model'])
        ropt.load_state_dict(checkpoint['optimizer'])
        rsched.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'],generator)
        step(restored,ropt,rsched,generator)
        for key,value in restored.state_dict().items():
            torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
        assert rsched.state_dict() == sched.state_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    persistent_session_required()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    seed_everything(42)
    with wandb.init(project='superNISP-corrected',name=args.output.name,job_type='integration-check',
                    mode='online',dir=str(args.output),settings=wandb.Settings(init_timeout=60)) as run:
        if run.settings.mode!='online' or not run.url:
            raise RuntimeError('Online W&B required')
        (args.output/'wandb_url.txt').write_text(run.url+'\n')
        results = {}
        for name,fn in [('loader_smoke',lambda:loader_smoke(args.data)),
                        ('legacy_equivalence_and_cuda',check_architecture),
                        ('masks_augmentation_moments',check_mask_and_geometry),
                        ('exact_resume',check_resume)]:
            fn()
            results[name] = 'passed'
            run.log({f'checks/{name}':1})
            print(f'{name}: passed',flush=True)
        (args.output/'checks.json').write_text(json.dumps(results,indent=2))
        run.summary.update(results)


if __name__=='__main__':
    main()
