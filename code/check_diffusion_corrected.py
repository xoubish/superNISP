"""Run mathematical and CUDA integration checks only after online W&B starts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import torch
import wandb
from train_paired import persistent_session_required,seed_everything
from diffusion_corrected import CorrectedDiffusion,fixed_noise
from paired_sample import PairedSample


def main(args):
    persistent_session_required()
    torch.set_num_threads(2)
    device=torch.device(args.device); torch.cuda.set_device(device)
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['diffusion_corrected.py','test_diffusion_corrected.py','check_diffusion_corrected.py']:
        shutil.copy2(Path(__file__).parent/name,args.output/name)
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name=args.output.name,
                    job_type='diffusion-checks',mode='online',dir=str(args.output)) as run:
        if run.settings.mode!='online' or not run.url: raise RuntimeError('Online W&B required')
        seed_everything(20261016)
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_diffusion_corrected.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): raise RuntimeError('Diffusion regression tests failed')
        dataset=PairedSample(args.data,'train','asinh')
        batch={k:dataset[0][k][None].to(device) for k in ['nisp','jwst','nisp_valid','jwst_valid']}
        model=CorrectedDiffusion().to(device)
        noise=fixed_noise([int(dataset[0]['row'])],42,device)
        loss,_=model.training_loss(batch['nisp'],batch['nisp_valid'],batch['jwst'],batch['jwst_valid'],
                                  torch.tensor([999],device=device),noise)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('CUDA forward/backward failed')
        model.eval()
        sample=model.sample(batch['nisp'],batch['nisp_valid'],noise,5)
        torch.testing.assert_close(sample,model.sample(batch['nisp'],batch['nisp_valid'],noise,5),rtol=0,atol=0)
        if sample.shape!=(1,1,205,205): raise ValueError('Output geometry failed')
        physical=dataset.jwst_to_mjy_sr(batch['jwst'])
        linear=PairedSample(args.data,'train','linear')
        torch.testing.assert_close(physical.cpu(),linear.jwst_to_mjy_sr(linear[0]['jwst'][None]),rtol=2e-5,atol=1e-6)
        report=dict(complete=True,regression_tests=result.testsRun,cuda_geometry='41x41 -> 205x205',
            cuda_loss=float(loss),parameters=sum(p.numel() for p in model.parameters()),
            deterministic_repeat='bitwise equal',normalization_roundtrip='passed',wandb_url=run.url,
            source_sha256=hashlib.sha256((Path(__file__).parent/'diffusion_corrected.py').read_bytes()).hexdigest())
        (args.output/'checks.json').write_text(json.dumps(report,indent=2))
        run.summary.update(report)
        dataset.close(); linear.close()
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    main(p.parse_args())
