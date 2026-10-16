"""Verified fresh full diffusion training plus automatic matched shape assessment."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import torch
import wandb
from train_paired import persistent_session_required,seed_everything,sha256
from train_diffusion_full import SOURCES
from diffusion_full_runtime import CooperativeDiffusion
from diffusion_corrected import fixed_noise
from paired_sample import PairedSample

ROOT=Path(__file__).resolve().parents[1]


def main():
    persistent_session_required()
    output=ROOT/'runs/diffusion-full50-pipeline'
    training=ROOT/'runs/diffusion-full50-s20261016'
    comparison=ROOT/'runs/diffusion-full50-comparison'
    for path in [output,training,comparison]:
        if path.exists(): raise RuntimeError(f'Refusing to overwrite {path}')
    previous=json.loads((ROOT/'runs/diffusion-corrected-check/checks.json').read_text())
    if not previous['complete'] or previous['source_sha256']!=sha256(ROOT/'code/diffusion_corrected.py'):
        raise RuntimeError('Diffusion core differs from the passed integration checks')
    data=ROOT.parent/'supernisp_nir_y_f115w_v1'
    baseline=json.loads((ROOT/'runs/residual-full50-s20261016/provenance.json').read_text())
    for name,expected in baseline['data_hashes'].items():
        if sha256(data/name)!=expected: raise RuntimeError('Sample metadata differs from baseline')
    output.mkdir(parents=True); (output/'source').mkdir()
    sources=SOURCES+['run_diffusion_full.py','test_diffusion_full.py','compare_diffusion_full.py']
    hashes={name:sha256(ROOT/'code'/name) for name in sources}
    for name in sources:
        target=output/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'code'/name,target)
    command=[sys.executable,'-u',str(ROOT/'code/train_diffusion_full.py'),
        '--data',str(data),'--output',str(training),'--device','cuda:1',
        '--mode','train','--train-limit','0','--validation-limit','256','--epochs','50','--batch-size','8']
    protocol=dict(command=command,source_hashes=hashes,data_hashes=baseline['data_hashes'],
        budget='Fresh 50 epochs, all 17816 training objects, batch 8, 111350 optimizer updates; pilot weights discarded.',
        monitoring='256 fixed validation galaxies; 50-step NISP-only DDIM; seed 314159; physical MAE selection, shape/failure metrics and 8 galaxy panels every epoch.',
        final='Full 2118 validation galaxies, 3 seeds at selected checkpoint, linear-intensity mean image, matched shape comparison with frozen residual and completed direct5.',
        gpu='GPU 1; yield at each training/sampling batch boundary while existing direct5 analyses use GPU 1; resume automatically.',
        limitations='Same exposures/update budget as residual, but representation, objective, capacity, validation-selection population and compute differ. Single training seed. No test data.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(2); torch.cuda.set_device(1)
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name='diffusion-full50-preflight',
                    mode='online',job_type='diffusion-full-preflight',dir=str(output),config=protocol) as run:
        if run.settings.mode!='online' or not run.url: raise RuntimeError('Online W&B required')
        seed_everything(20261016)
        suite=unittest.defaultTestLoader.discover(str(ROOT/'code'),pattern='test_diffusion_full.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): raise RuntimeError('Full-run regression checks failed')
        dataset=PairedSample(data,'train','asinh')
        items=[dataset[i] for i in range(8)]
        batch={key:torch.stack([item[key] for item in items]).to('cuda:1') for key in ['nisp','nisp_valid','jwst','jwst_valid']}
        model=CooperativeDiffusion().to('cuda:1')
        noise=fixed_noise([item['row'] for item in items],42,'cuda:1')
        torch.cuda.reset_peak_memory_stats(1)
        loss,_=model.training_loss(batch['nisp'],batch['nisp_valid'],batch['jwst'],batch['jwst_valid'],
            torch.tensor([0,100,300,500,700,800,900,999],device='cuda:1'),noise)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Batch-8 GPU check failed')
        model.eval(); generated=model.sample(batch['nisp'],batch['nisp_valid'],noise,5)
        if generated.shape!=(8,1,205,205): raise RuntimeError('Batch-8 sampling geometry failed')
        checks=dict(passed=True,tests=result.testsRun,batch_size=8,peak_cuda_gib=torch.cuda.max_memory_allocated(1)/2**30,
            training_objects=len(dataset),planned_updates=50*((len(dataset)+7)//8),wandb_url=run.url)
        if checks['planned_updates']!=111350: raise RuntimeError('Unexpected training budget')
        (output/'checks.json').write_text(json.dumps(checks,indent=2)); run.summary.update(checks)
        dataset.close()
        del model,noise,batch,generated,loss
        torch.cuda.empty_cache()
    stages=[('full50',command),('comparison',[sys.executable,'-u',str(ROOT/'code/compare_diffusion_full.py'),
        '--training',str(training),'--output',str(comparison)])]
    for name,cmd in stages:
        for source,expected in hashes.items():
            if sha256(ROOT/'code'/source)!=expected: raise RuntimeError(f'Queued source changed: {source}')
        (output/'status.json').write_text(json.dumps(dict(complete=False,stage=name),indent=2))
        subprocess.run(cmd,check=True,cwd=ROOT)
    (output/'status.json').write_text(json.dumps(dict(complete=True,stage='all_completed'),indent=2))


if __name__=='__main__': main()
