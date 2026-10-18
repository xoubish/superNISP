"""Local, validation-only comparison after both matched refinements complete.

No GPU/model execution or W&B upload. Frozen Figure 6 inputs remain unchanged.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from residual_refinement import FrozenDiagnostics, fixed_shapes
from probe_shared_gpu import sha
from spatial_spectra import block_summary, EDGES, BANDS
from cached_shape_comparison import summarize
from shape_diagnostics import REASONS
from robust_shape_measurements import adaptive_moments, HSM_REASONS

ROOT=Path(__file__).resolve().parents[1]
METHODS=['legacy','control_best','balanced_best','control_last','balanced_last']
LABELS=['Original','Control best','Balanced best','Control last','Balanced last']
COLORS=['0.35','#0072B2','#D55E00','#56B4E9','#E69F00']


def read(path):return json.loads(path.read_text())


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def wait_for_runs(paths):
    while True:
        complete=[]
        for path in paths:
            status=read(path/'status.json') if (path/'status.json').exists() else {}
            if 'error' in status:raise RuntimeError(f'{path.name}: {status["error"]}')
            exit_path=path.with_name(path.name+'-console')/'exit_status'
            if exit_path.exists() and int(exit_path.read_text())!=0:
                raise RuntimeError(f'{path.name} exited with {exit_path.read_text().strip()}')
            complete.append(status.get('complete',False))
        if all(complete):return
        time.sleep(30)


def adaptive(images,valid):
    parts=[adaptive_moments(torch.from_numpy(images[i:i+32,None].astype(np.float64)),
                           torch.from_numpy(valid[i:i+32,None]))
           for i in range(0,len(images),32)]
    return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}


def shape_difference(a,b,reference,good,blocks):
    """Difference of population component RMSE, with paired block resampling."""
    squared=np.stack([((v[good]-reference[good])**2).mean(1) for v in [a,b]],axis=1)
    labels,inverse=np.unique(blocks[good],return_inverse=True)
    counts=np.bincount(inverse)
    sums=np.stack([squared[inverse==i].sum(0) for i in range(len(labels))])
    draws=np.random.default_rng(20261017).integers(len(labels),size=(2000,len(labels)))
    boot=np.sqrt(sums[draws].sum(1)/counts[draws].sum(1)[:,None])
    means=np.sqrt(squared.mean(0))
    return dict(mean=float(means[0]-means[1]),
                ci95=np.percentile(boot[:,0]-boot[:,1],[2.5,97.5]).tolist())


def save_plot(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def plots(output,result,epochs,measurements,images,frozen):
    fig,axes=plt.subplots(1,3,figsize=(13,3.4))
    for j,name in enumerate(METHODS[:3]):
        for region,marker in [('source','o'),('outside','s')]:
            value=result['errors'][name][region+'_rmse'];mean=value['mean']
            axes[0].errorbar(j,mean,yerr=np.abs(np.asarray(value['ci95'])-mean)[:,None],
                marker=marker,mfc=COLORS[j] if region=='source' else 'white',color=COLORS[j],capsize=3)
        for axis,key,transform in [(axes[1],'log_power_ratio',lambda x:10**x),(axes[2],'correlation',lambda x:x)]:
            value=result['spectra'][name][key]
            mean=transform(np.asarray(value['mean'])[:14]);ci=transform(np.asarray(value['ci95'])[:14])
            x=np.sqrt(EDGES[:-1]*EDGES[1:])
            axis.plot(x,mean,color=COLORS[j],label=LABELS[j])
            axis.fill_between(x,ci[:,0],ci[:,1],color=COLORS[j],alpha=.15)
    axes[0].set_xticks(range(3),LABELS[:3]);axes[0].set_ylabel('Mean object RMSE (MJy/sr)')
    axes[0].plot([],[],color='0.4',marker='o',ls='',label='Source')
    axes[0].plot([],[],color='0.4',marker='s',mfc='white',ls='',label='Outside source');axes[0].legend()
    axes[1].set_ylabel('Source power / reference');axes[1].set_yscale('log');axes[1].axhline(1,color='0.5',ls=':')
    axes[2].set_ylabel('Source signed Fourier correlation');axes[2].legend(fontsize=9)
    for axis in axes[1:]:axis.set_xscale('log');axis.set_xlabel('Frequency (cycles / target pixel)')
    fig.tight_layout();save_plot(fig,output/'comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for branch,records in epochs.items():
        j=1 if branch=='control' else 2
        for axis,key in zip(axes,['source_mae','outside_rmse']):
            axis.plot([r['epoch'] for r in records],[r['validation'][key] for r in records],
                      '-o',color=COLORS[j],label=branch)
            axis.axhline(records[0]['validation'][key],color='0.5',ls=':');axis.set_xlabel('Extra epochs')
    axes[0].set_ylabel('Source MAE (selection metric)');axes[1].set_ylabel('Outside-source RMSE')
    axes[0].legend();fig.tight_layout();save_plot(fig,output/'learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for axis,kind in zip(axes,['fixed','adaptive']):
        shape=result['shapes'][kind]
        for j,name in enumerate(METHODS):
            value=shape['methods'][name]['component_rmse'];mean=value['mean']
            axis.errorbar(j,mean,yerr=np.abs(np.asarray(value['ci95'])-mean)[:,None],
                          fmt='o',color=COLORS[j],capsize=3)
        axis.set_xticks(range(5),LABELS,rotation=30,ha='right')
        axis.set_ylabel(f'{kind.capitalize()} ellipticity component RMSE')
        axis.set_title(f'{shape["common_successes"]} common successes')
    fig.tight_layout();save_plot(fig,output/'shapes')
    # Pre-existing, observed-image-selected Figure 5 examples; no selection on
    # refinement predictions/errors. Not a population performance estimate.
    selection_path=ROOT/'paper_figures/inputs/galaxy_examples_selection_v06.json'
    selection=read(selection_path);write(output/'gallery_selection.json',selection)
    indices=[frozen.ids.tolist().index(i) for i in selection['object_ids']]
    fig,axes=plt.subplots(len(indices),4,figsize=(10,2.4*len(indices)))
    from matplotlib.colors import AsinhNorm
    for row,(i,limits) in enumerate(zip(indices,selection['limits_mjy_sr'])):
        axes[row,0].imshow(np.where(frozen.valid[i],frozen.target[i],np.nan),origin='lower',cmap='gray',
            norm=AsinhNorm(linear_width=limits['stretch'],vmin=limits['low'],vmax=limits['high']))
        for col,name in enumerate(METHODS[:3],1):
            error=np.where(frozen.valid[i],images[name][i]-frozen.target[i],np.nan)
            axes[row,col].imshow(error,origin='lower',cmap='RdBu_r',
                                vmin=-limits['residual'],vmax=limits['residual'])
        axes[row,0].set_ylabel(str(frozen.ids[i]))
        for axis in axes[row]:axis.set_xticks([]);axis.set_yticks([])
    for axis,title in zip(axes[0],['JWST reference','Original − reference','Control − reference','Balanced − reference']):axis.set_title(title)
    fig.tight_layout();save_plot(fig,output/'fixed_examples')


def main(args):
    torch.set_num_threads(2)
    paths={'control':args.control.resolve(),'balanced':args.balanced.resolve()}
    if args.wait:wait_for_runs(paths.values())
    statuses={k:read(p/'status.json') for k,p in paths.items()}
    assert all(s['complete'] and s['updates']==22270 and s['parent_unchanged'] for s in statuses.values())
    contracts={k:read(p/'contract.json') for k,p in paths.items()}
    for key in ['source_hashes','data_hashes','parent_sha256','masks_sha256']:
        assert contracts['control'][key]==contracts['balanced'][key]
    epochs={k:[read(p/f'epoch-{i:02d}.json') for i in range(11)] for k,p in paths.items()}
    for i in range(1,11):
        assert epochs['control'][i]['training_stream_sha256']==epochs['balanced'][i]['training_stream_sha256'],f'Training streams differ at epoch {i}'
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ['compare_residual_refinements.py','residual_refinement.py','cached_shape_comparison.py',
                 'robust_shape_measurements.py','shape_diagnostics.py','spatial_spectra.py']:
        (output/'source').mkdir(exist_ok=True);shutil.copy2(ROOT/'code'/name,output/'source'/name)
    frozen=FrozenDiagnostics();images={'legacy':frozen.baseline};measurements={}
    _,measurements['legacy']=frozen.measure(frozen.baseline)
    for branch,path in paths.items():
        best=statuses[branch]['best_epoch']
        expected=min(range(11),key=lambda i:epochs[branch][i]['validation']['source_mae'])
        assert best==expected
        for suffix,epoch,file in [('best',best,'best_predictions.npy'),('last',10,'predictions-10.npy')]:
            checkpoint=torch.load(path/(suffix+'.pt'),map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==epoch and checkpoint['step']==epoch*2227
            assert all(float(v['step'])==109123+checkpoint['step'] for v in checkpoint['optimizer']['state'].values())
            del checkpoint
            name=branch+'_'+suffix;images[name]=np.load(path/file)
            with np.load(path/f'measurements-{epoch:02d}.npz') as saved:
                np.testing.assert_array_equal(saved['object_ids'],frozen.ids)
                measurements[name]={key:saved[key] for key in saved.files if key!='object_ids'}
    errors={name:{key:block_summary(values[key],frozen.blocks) for key in
                  ['source_mae','source_rmse','outside_mae','outside_rmse']} for name,values in measurements.items()}
    spectra={name:{key:block_summary(values[key],frozen.blocks[frozen.primary]) for key in
                   ['log_power_ratio','correlation']} for name,values in measurements.items()}
    comparisons=[(m,'legacy') for m in METHODS[1:]]+[('balanced_best','control_best'),('balanced_last','control_last')]
    paired={}
    for a,b in comparisons:
        record={key:block_summary(measurements[a][key]-measurements[b][key],frozen.blocks) for key in errors[a]}
        for key in ['log_power_ratio','correlation']:
            record[key]=block_summary(measurements[a][key]-measurements[b][key],frozen.blocks[frozen.primary])
        record['absolute_log_power_error']=block_summary(np.abs(measurements[a]['log_power_ratio'])-
            np.abs(measurements[b]['log_power_ratio']),frozen.blocks[frozen.primary])
        paired[a+'__minus__'+b]=record
    shapes={};shape_arrays={}
    for kind,method,reasons in [('fixed',fixed_shapes,REASONS),('adaptive',adaptive,HSM_REASONS)]:
        print('Measuring '+kind+' shapes',flush=True)
        values={'target':method(frozen.target,frozen.valid)}
        for name in METHODS:
            values[name]=({key:measurements[name]['shape_'+key] for key in ['e','good','flags']}
                          if kind=='fixed' else method(images[name],frozen.valid))
        shapes[kind]=summarize(values,frozen.blocks,frozen.ids,METHODS,reasons)
        common=np.logical_and.reduce([v['good'] for v in values.values()])
        shapes[kind]['paired_all']={a+'__minus__'+b:shape_difference(values[a]['e'],values[b]['e'],
            values['target']['e'],common,frozen.blocks) for a,b in comparisons}
        shape_arrays.update({kind+'_'+name+'_'+key:value for name,v in values.items() for key,value in v.items()})
    result=dict(complete=False,validation_objects=len(frozen.ids),source_spectrum_objects=int(frozen.primary.sum()),
        status=statuses,errors=errors,spectra=spectra,paired=paired,shapes=shapes,
        frequency_edges=EDGES.tolist(),bands=list(BANDS),training_streams_match=True,
        bootstrap=dict(draws=2000,seed=20261017,unit='spatial block',weighting='equal objects'),
        input_hashes={str(p/f):sha(p/f) for p in paths.values() for f in
            ['contract.json','best.pt','last.pt','best_predictions.npy','predictions-10.npy']},
        limitations=['One parent training seed; validation selection, not held-out test performance.',
                    'Observed shapes/spectra, no PSF or noise correction.',
                    'Shape RMSE conditional on reference and all five model measurements succeeding.',
                    'Shared GPU load; no guaranteed compute share or competing-job throughput measurement.'])
    write(output/'comparison.json',result)
    np.savez_compressed(output/'shapes.npz',object_ids=frozen.ids,**shape_arrays)
    plots(output,result,epochs,measurements,images,frozen)
    lines=['# Paired ResNet refinement results','',
        'Validation only; both branches share the same parent, batches and augmentation stream. '
        'All ten extra epochs completed; best is selected by source MAE with epoch zero eligible.','',
        '| Model | Epoch | Source MAE | Source RMSE | Outside RMSE | Fixed shape RMSE | Adaptive shape RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in zip(METHODS,LABELS):
        epoch=0 if name=='legacy' else statuses[name.split('_')[0]]['best_epoch'] if name.endswith('best') else 10
        vals=[errors[name][key]['mean'] for key in ['source_mae','source_rmse','outside_rmse']]
        vals += [shapes[k]['methods'][name]['component_rmse']['mean'] for k in ['fixed','adaptive']]
        lines.append(f'| {label} | {epoch} | '+' | '.join(f'{v:.7f}' for v in vals)+' |')
    lines += ['', 'Paired changes below include 95% spatial-block intervals. Negative error changes '
              'and positive correlation changes favor the first model. These exploratory intervals '
              'do not account for checkpoint selection or training-seed variation.', '',
              '| Difference | Source MAE | Outside RMSE | Fine-scale correlation | Fixed shape RMSE |',
              '|---|---:|---:|---:|---:|']
    for a,b in [('control_best','legacy'),('balanced_best','legacy'),('balanced_best','control_best')]:
        key=a+'__minus__'+b;record=paired[key]
        fine=record['correlation']
        values=[record['source_mae'],record['outside_rmse'],
                dict(mean=fine['mean'][-2],ci95=fine['ci95'][-2]),shapes['fixed']['paired_all'][key]]
        cells=[f'{v["mean"]:+.4g} [{v["ci95"][0]:+.4g}, {v["ci95"][1]:+.4g}]' for v in values]
        lines.append('| '+a+' − '+b+' | '+' | '.join(cells)+' |')
    lines += ['', 'Pixel errors are MJy/sr; source/outside RMSE is the mean of per-object RMSE. '
              'Shape errors use a common success population across all five models and the reference.',
              '', '![Reconstruction comparison](comparison.png)','![Learning curves](learning_curves.png)',
              '![Observed shapes](shapes.png)','![Pre-existing observed-image-selected examples](fixed_examples.png)',
              '', 'Full paired spatial-bootstrap intervals, per-method success/failure counts and provenance '
              'are in [comparison.json](comparison.json). The example sheet uses the existing Figure 5 '
              'editorial selection and display limits; it is illustrative, not a performance sample.', '',
              *['- '+limit for limit in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    result['complete']=True
    write(output/'comparison.json',result)
    print('Completed local comparison: '+str(output/'report.md'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,default=ROOT/'runs/resnet-refine-control-v03')
    parser.add_argument('--balanced',type=Path,default=ROOT/'runs/resnet-refine-balanced-v03')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/resnet-refinement-comparison-v01')
    parser.add_argument('--wait',action='store_true')
    main(parser.parse_args())
