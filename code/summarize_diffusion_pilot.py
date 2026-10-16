"""Summarize completed pilot on fixed populations; no model execution."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from train_paired import persistent_session_required,sha256
from shape_diagnostics import METHODS


def read(path): return json.loads(path.read_text())
def arrays(path):
    with np.load(path) as data: return dict(data)


def main(args):
    persistent_session_required()
    status=read(args.pilot/'status.json')
    if not status['complete']: raise ValueError('Pilot must finish before assessment')
    records=[json.loads(line) for line in (args.pilot/'metrics.jsonl').read_text().splitlines()]
    best=min(records,key=lambda r:r['generated/mae_mjy_sr'])
    epochs=[arrays(args.pilot/f"epoch-{r['epoch']:03d}-measurements.npz") for r in records]
    repeats=[arrays(args.pilot/f'best-sample-{i}-measurements.npz') for i in range(3)]
    all_arrays=epochs+repeats
    for a in all_arrays[1:]:
        for key in ['object_ids','rows','blocks','jwst_e','jwst_good','jwst_flags']:
            np.testing.assert_array_equal(a[key],epochs[0][key])
    def common_for(collection):
        mask=collection[0]['jwst_good'].copy()
        for a in collection:
            for m in METHODS: mask &= a[m+'_good']
        return mask
    def rmse(a,method,mask):
        return float(np.sqrt(((a[method+'_e'][mask]-a['jwst_e'][mask])**2).mean())) if mask.any() else None
    common=common_for(epochs)
    repeat_common=common_for(repeats)
    # Compare with the frozen fully trained reference only as a contextual bar.
    # The pilot has much less training and a different recipe; no architecture inference.
    legacy=arrays(args.baseline/'measurements.npz')
    lookup={v:i for i,v in enumerate(legacy['object_ids'])}
    order=[lookup[v] for v in repeats[0]['object_ids']]
    legacy={k:v[order] for k,v in legacy.items()}
    np.testing.assert_allclose(legacy['jwst_e'],repeats[0]['jwst_e'],rtol=2e-5,atol=2e-6)
    np.testing.assert_array_equal(legacy['rows'],repeats[0]['rows'])
    context=repeat_common.copy()
    for m in ['jwst',*METHODS]: context &= legacy[m+'_good']
    initial=read(args.pilot/'initial_metrics.json')['mae_mjy_sr']
    report=dict(wandb_training_url=status['wandb_url'],best_epoch=best['epoch'],updates=status['updates'],
        initial_generated_mae_mjy_sr=initial,best_generated_mae_mjy_sr=status['best'],
        generated_mae_reduction_percent=100*(1-status['best']/initial),
        fixed_epoch_population=dict(objects=int(common.sum()),object_ids=epochs[0]['object_ids'][common].tolist(),
            epochs=[r['epoch'] for r in records],sr_rmse=[rmse(a,'sr',common) for a in epochs],
            native_rmse=rmse(epochs[0],'native',common)),
        repeated_samples=read(args.pilot/'sample_variation.json'),
        matched_context=dict(objects=int(context.sum()),object_ids=repeats[0]['object_ids'][context].tolist(),
            native_rmse=rmse(repeats[0],'native',context),bilinear_rmse=rmse(repeats[0],'bilinear',context),
            legacy_full50_rmse=rmse(legacy,'sr',context),diffusion_sample_rmse=[rmse(a,'sr',context) for a in repeats]),
        limitations=['64 validation objects; 512 training objects and 640 updates versus the legacy full run with 111350 updates.',
            'Best checkpoint selected by fixed-seed generated pixel MAE, not shapes. Same IDs used for every displayed comparison.',
            'Physical target shapes agree within 2e-5 relative / 2e-6 absolute after asinh roundtrip.',
            'No test data. Sampling scatter is not calibrated uncertainty. This pilot cannot rank architectures.'])
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),args.output/Path(__file__).name)
    inputs=[args.pilot/'provenance.json',args.pilot/'status.json',args.pilot/'sample_variation.json',args.baseline/'measurements.npz']
    inputs+=list(args.pilot.glob('*measurements.npz'))
    (args.output/'provenance.json').write_text(json.dumps(dict(inputs={str(p.resolve()):sha256(p) for p in inputs},
        source_sha256=sha256(__file__)),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot([0]+[r['epoch'] for r in records],[initial]+[r['generated/mae_mjy_sr'] for r in records],marker='o')
    axes[0].set(xlabel='Epoch',ylabel='Generated MAE (MJy/sr)',title='NISP-only generation, fixed seed')
    axes[1].plot([r['epoch'] for r in records],[rmse(a,'sr',common) for a in epochs],marker='o',label='Diffusion')
    if common.any(): axes[1].axhline(rmse(epochs[0],'native',common),color='gray',label='Native')
    axes[1].set(xlabel='Epoch',ylabel='Shape component RMSE',title=f'Same {int(common.sum())} successful galaxies at every epoch')
    axes[1].legend()
    values=report['matched_context']
    labels=['Native','Bilinear','Legacy\nfull50','Diffusion\nseed 0','Diffusion\nseed 1','Diffusion\nseed 2']
    scores=[values['native_rmse'],values['bilinear_rmse'],values['legacy_full50_rmse'],*values['diffusion_sample_rmse']]
    axes[2].bar(labels,[np.nan if v is None else v for v in scores]); axes[2].tick_params(axis='x',labelsize=8)
    axes[2].set(ylabel='Shape component RMSE',title=f'Same {int(context.sum())} galaxies; unequal training budgets')
    fig.tight_layout(); fig.savefig(args.output/'summary.png',dpi=140); plt.close(fig)
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name=args.output.name,
                    job_type='diffusion-pilot-assessment',mode='online',dir=str(args.output)) as run:
        if run.settings.mode!='online' or not run.url: raise RuntimeError('Online W&B required')
        report['wandb_url']=run.url
        run.log({'pilot/assessment':wandb.Image(str(args.output/'summary.png'))})
        run.summary.update(report)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=Path('runs/diffusion-corrected-pilot'))
    parser.add_argument('--baseline',type=Path,default=Path('runs/validation-shapes-full50-best'))
    parser.add_argument('--output',type=Path,default=Path('runs/diffusion-pilot-assessment'))
    main(parser.parse_args())
