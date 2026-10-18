"""Wait for independent baseline seeds, then evaluate frozen validation diagnostics.

GPU inference requires tmux and online W&B; comparison plots stay local.
"""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from residual_refinement import FrozenDiagnostics
from train_paired import persistent_session_required, seed_everything, make_loader
from train_residual_refinement import evaluate
from compare_residual_refinements import wait_for_runs, adaptive, shape_difference, save_plot
from cached_shape_comparison import summarize
from spatial_spectra import block_summary, EDGES
from shape_diagnostics import REASONS
from robust_shape_measurements import HSM_REASONS
from probe_shared_gpu import sha

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[20261016,20261018,20261019]


def read(path):return json.loads(path.read_text())


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(args):
    persistent_session_required()
    paths={seed:ROOT/f'runs/residual-full50-s{seed}' for seed in SEEDS}
    contract=read(ROOT/'runs/resnet-seed-replication-v01/contract.json')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sources=['evaluate_residual_seed_replicates.py','residual_refinement.py','train_residual_refinement.py',
             'compare_residual_refinements.py','cached_shape_comparison.py','robust_shape_measurements.py',
             'shape_diagnostics.py','spatial_spectra.py','probe_shared_gpu.py',*contract['source_hashes']]
    (output/'source').mkdir()
    for name in sources:shutil.copy2(ROOT/'code'/name,output/'source'/name)
    write(output/'provenance.json',dict(contract=contract,source_hashes={n:sha(output/'source'/n) for n in sources},
        split='validation',test_used=False,selection='Historical full-field masked MAE; best and last shown',
        bootstrap='2000 paired spatial-block draws, seed 20261017; conditional on each trained checkpoint',
        seed_variation='Descriptive mean and sample standard deviation across three independent training seeds'))
    if args.wait:
        print('Waiting for both 50-epoch baseline replications.',flush=True)
        wait_for_runs([paths[s] for s in SEEDS[1:]])
    for path in paths.values():
        status=read(path/'status.json')
        assert status['complete'] and status['epoch']==50 and status['updates']==111350
    for path in paths.values():
        provenance=read(path/'provenance.json')
        for key in ['source_hashes','data_hashes']:assert provenance[key]==contract[key]
    for name,digest in contract['source_hashes'].items():assert sha(ROOT/'code'/name)==digest
    data=Path(read(paths[SEEDS[0]]/'provenance.json')['config']['data'])
    for name,digest in contract['data_hashes'].items():assert sha(data/name)==digest
    torch.set_num_threads(2);seed_everything(20261016)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    frozen=FrozenDiagnostics();dataset=PairedSample(data,'validation','linear')
    generator=torch.Generator().manual_seed(20261017)
    loader=make_loader(dataset,list(range(len(dataset))),8,0,generator,False)
    images={'legacy':frozen.baseline};measurements={};metrics={};checkpoints={}
    metrics['legacy'],measurements['legacy']=frozen.measure(frozen.baseline)
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name='resnet-three-seed-validation',
                    job_type='seed-replication-evaluation',mode='online',dir=str(output),
                    config=dict(seeds=SEEDS,split='validation',contract=contract),
                    settings=wandb.Settings(init_timeout=60)) as run:
        if run.settings.mode!='online' or not run.url:raise RuntimeError('Online monitoring required')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        try:
            model=EuclidToJWSTSuperResolution(head='legacy').cuda()
            for seed,path in paths.items():
                for suffix in ['best','last']:
                    checkpoint_path=path/(suffix+'.pt')
                    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                    name='legacy' if seed==SEEDS[0] and suffix=='best' else f'seed{seed}_{suffix}'
                    checkpoints[name]=dict(path=str(checkpoint_path),sha256=sha(checkpoint_path),
                        epoch=checkpoint['epoch'],updates=checkpoint['step'],seed=seed,selection=suffix)
                    if name=='legacy':
                        assert checkpoints[name]['sha256']==contract.get('original_checkpoint_sha256',
                            'be747f0cd0b02bb81ff0ab42a9ce8ea81a0e473c28c58d8406a428f758ea6b56')
                        continue
                    assert checkpoint['signature']['data_hashes']==contract['data_hashes']
                    assert checkpoint['validation_indices']==list(range(2118))
                    model.load_state_dict(checkpoint['model']);del checkpoint
                    metrics[name],measurements[name],images[name]=evaluate(model,loader,dataset,frozen)
                    np.save(output/(name+'_predictions.npy'),images[name])
                    np.savez_compressed(output/(name+'_measurements.npz'),object_ids=frozen.ids,**measurements[name])
                    run.log({name+'/'+key:value for key,value in metrics[name].items()})
                    print(name,metrics[name],flush=True)
            del model
        finally:dataset.close()
    methods=list(images)
    errors={name:{key:block_summary(values[key],frozen.blocks) for key in
                  ['source_mae','source_rmse','outside_mae','outside_rmse']} for name,values in measurements.items()}
    spectra={name:{key:block_summary(values[key],frozen.blocks[frozen.primary]) for key in
                   ['log_power_ratio','correlation']} for name,values in measurements.items()}
    paired={name:{key:block_summary(values[key]-measurements['legacy'][key],frozen.blocks)
                  for key in errors[name]} for name,values in measurements.items() if name!='legacy'}
    shapes={};shape_arrays={}
    for kind,reasons in [('fixed',REASONS),('adaptive',HSM_REASONS)]:
        print('Measuring '+kind+' shapes for all seeds.',flush=True)
        values={'target':frozen.reference_shapes if kind=='fixed' else adaptive(frozen.target,frozen.valid)}
        for name in methods:
            values[name]=({key:measurements[name]['shape_'+key] for key in ['e','good','flags']}
                          if kind=='fixed' else adaptive(images[name],frozen.valid))
        shapes[kind]=summarize(values,frozen.blocks,frozen.ids,methods,reasons)
        common=np.logical_and.reduce([v['good'] for v in values.values()])
        shapes[kind]['paired_vs_original']={name:shape_difference(values[name]['e'],values['legacy']['e'],
            values['target']['e'],common,frozen.blocks) for name in methods if name!='legacy'}
        shape_arrays.update({kind+'_'+name+'_'+key:value for name,v in values.items() for key,value in v.items()})
    best_methods=['legacy',*[f'seed{s}_best' for s in SEEDS[1:]]]
    seed_summary={}
    for key in ['source_mae','source_rmse','outside_rmse']:
        vals=[errors[name][key]['mean'] for name in best_methods]
        seed_summary[key]=dict(seeds=SEEDS,values=vals,mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)))
    result=dict(complete=False,checkpoints=checkpoints,errors=errors,spectra=spectra,paired=paired,
        shapes=shapes,seed_summary=seed_summary,objects=len(frozen.ids),source_spectrum_objects=int(frozen.primary.sum()),
        limitations=['Validation-selected best checkpoints; no held-out test results.',
                    'Three independent seeds give a limited estimate of training variation.',
                    'Spatial-block intervals do not include training-seed or checkpoint-selection uncertainty.',
                    'Shape RMSE conditional on all six variants and reference succeeding; no PSF/shear calibration.'])
    write(output/'comparison.json',result)
    np.savez_compressed(output/'shapes.npz',object_ids=frozen.ids,**shape_arrays)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for j,(seed,name) in enumerate(zip(SEEDS,best_methods)):
        color=['0.35','#0072B2','#D55E00'][j]
        for region,marker in [('source','o'),('outside','s')]:
            v=errors[name][region+'_rmse'];avg=v['mean']
            axes[0].errorbar(j,avg,yerr=np.abs(np.asarray(v['ci95'])-avg)[:,None],color=color,
                marker=marker,mfc=color if region=='source' else 'white',capsize=3)
        for axis,key in [(axes[1],'log_power_ratio'),(axes[2],'correlation')]:
            v=spectra[name][key];avg=np.asarray(v['mean'])[:14];ci=np.asarray(v['ci95'])[:14]
            if key=='log_power_ratio':avg=10**avg;ci=10**ci
            x=np.sqrt(EDGES[:-1]*EDGES[1:]);axis.plot(x,avg,color=color,label=str(seed))
            axis.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.15)
    axes[0].set_xticks(range(3),[str(s) for s in SEEDS],rotation=20)
    axes[0].set_ylabel('Mean object RMSE (MJy/sr)')
    for region,marker in [('Source','o'),('Outside source','s')]:
        axes[0].plot([],[],color='0.4',marker=marker,mfc='0.4' if marker=='o' else 'white',ls='',label=region)
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel('Source power / reference');axes[1].set_yscale('log');axes[1].axhline(1,color='0.5',ls=':')
    axes[2].set_ylabel('Source signed Fourier correlation');axes[2].legend(fontsize=9)
    for axis in axes[1:]:axis.set_xscale('log');axis.set_xlabel('Frequency (cycles / target pixel)')
    fig.tight_layout();save_plot(fig,output/'seed_comparison')
    lines=['# Independent-seed baseline comparison','',
        'Same original 50-epoch recipe; only the training seed changes. Best uses the original full-field '
        'validation MAE selection. These checkpoints have no refinement updates.','',
        '| Variant | Epoch | Source MAE | Source RMSE | Outside RMSE | Fixed shape RMSE | Adaptive shape RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in methods:
        vals=[errors[name][k]['mean'] for k in ['source_mae','source_rmse','outside_rmse']]
        vals += [shapes[k]['methods'][name]['component_rmse']['mean'] for k in ['fixed','adaptive']]
        lines.append(f'| {name} | {checkpoints[name]["epoch"]} | '+' | '.join(f'{v:.7f}' for v in vals)+' |')
    lines += ['', '![Best-checkpoint comparison across seeds](seed_comparison.png)', '',
              'Full paired spatial intervals, fixed/adaptive shape success and failure counts, '
              'and descriptive between-seed variation are in [comparison.json](comparison.json).', '',
              *['- '+v for v in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    result['complete']=True;write(output/'comparison.json',result)
    print('Complete: '+str(output/'report.md'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/resnet-seed-replication-v01/evaluation')
    parser.add_argument('--wait',action='store_true')
    main(parser.parse_args())
