"""Validation-only reconstruction, sharpness, morphology and visual comparison.

Reads frozen predictions after training; no GPU inference or W&B uploads.
"""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
import torch

from residual_refinement import FrozenDiagnostics,fixed_shapes
from spatial_spectra import block_summary,frequency_masks,paired_spectrum,transform,taper,validity_taper,BANDS
from cached_shape_comparison import summarize
from shape_diagnostics import REASONS
from robust_shape_measurements import HSM_REASONS
from compare_residual_refinements import read,write,wait_for_runs,adaptive,shape_difference,save_plot
from probe_shared_gpu import sha

ROOT=Path(__file__).resolve().parents[1]
PATHS=dict(control=ROOT/'runs/resnet-refine-control-v03',
           conditioned=ROOT/'runs/psf-redesign-conditioned-v01',
           deconv=ROOT/'runs/psf-redesign-deconv-v01')
DISPLAY=['legacy','control_best','conditioned_best','deconv_best']
LABELS=['Original ResNet','Extra-training control','PSF + noise','PSF + noise + inverse feature']
COLORS=['0.4','#aa4499','#4477aa','#228833']


def background_spectra(images,frozen):
    weights=[taper()*validity_taper(v & ~s,3) for v,s in zip(frozen.valid,frozen.source)]
    keep=frozen.primary & np.array([np.sum(w*w)>16 and w.sum()**2/max(np.sum(w*w),1e-30)>=64 for w in weights])
    reference=[transform(im,w) for im,w,k in zip(frozen.target,weights,keep) if k]
    masks=frequency_masks(); result={}
    for name,values in images.items():
        records=[paired_spectrum(transform(im,w),ref,masks) for im,w,ref in zip(values[keep],np.array(weights)[keep],reference)]
        result[name]={key:np.stack([r[key] for r in records]) for key in ['log_power_ratio','correlation']}
    return keep,result


def plots(out,report,images,arrays,frozen,epochs):
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,key,label in zip(axes,['source_rmse','outside_rmse','field_rmse'],['Source RMSE','Outside-source RMSE','Full common-field RMSE']):
        for j,name in enumerate(DISPLAY):
            rec=report['errors'][name][key]
            ax.errorbar(j,rec['mean'],yerr=np.abs(np.array(rec['ci95'])-rec['mean'])[:,None],fmt='o',color=COLORS[j],capsize=3)
        ax.set_xticks(range(4),LABELS,rotation=30,ha='right',fontsize=8)
        ax.set_ylabel(label+' [MJy/sr]');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save_plot(fig,out/'rmse')

    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    x=np.arange(4)
    for row,region in enumerate(['source','outside']):
        for name,label,color in zip(DISPLAY,LABELS,COLORS):
            for col,key in enumerate(['log_power_ratio','correlation']):
                rec=report['spectra'][region][name][key]
                y=np.array(rec['mean'])[-4:];ci=np.array(rec['ci95'])[-4:]
                if key=='log_power_ratio':y=10**y;ci=10**ci
                axes[row,col].plot(x,y,'o-',label=label,color=color,ms=4)
                axes[row,col].fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.1)
        axes[row,0].axhline(1,color='black',ls=':',lw=.8)
        axes[row,0].set_ylabel(region.capitalize()+' power / JWST')
        axes[row,1].set_ylabel(region.capitalize()+' structural correlation')
        for ax in axes[row]:ax.set_xticks(x,['15–30','6–15','3–6','2–3']);ax.grid(alpha=.15)
    for ax in axes[1]:ax.set_xlabel('Spatial scale [target pixels]')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.9));save_plot(fig,out/'sharpness_structure')

    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,kind in zip(axes,['fixed','adaptive']):
        table=report['shapes'][kind]
        for j,name in enumerate(DISPLAY):
            rec=table['methods'][name]['component_rmse']
            ax.errorbar(j,rec['mean'],yerr=np.abs(np.array(rec['ci95'])-rec['mean'])[:,None],fmt='o',color=COLORS[j],capsize=3)
        ax.set_xticks(range(4),LABELS,rotation=30,ha='right',fontsize=8)
        ax.set_ylabel(kind.capitalize()+' ellipticity RMSE')
        ax.set_title(f"{table['common_successes']} common successful objects",fontsize=10)
        ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save_plot(fig,out/'shapes')

    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for name,color in zip(PATHS,COLORS[1:]):
        rows=epochs[name]
        for ax,key in zip(axes,['source_rmse','outside_rmse','fine_3_6px_correlation']):
            ax.plot([r['epoch'] for r in rows],[r['validation'][key] for r in rows],'-o',label=name,color=color,ms=3)
            ax.set_xlabel('Additional epochs');ax.set_ylabel(key.replace('_',' '));ax.grid(alpha=.15)
    axes[0].legend();fig.tight_layout();save_plot(fig,out/'learning_curves')

    selection=read(ROOT/'paper_figures/inputs/galaxy_examples_selection_v06.json')
    write(out/'gallery_selection.json',selection)
    indices=[frozen.ids.tolist().index(i) for i in selection['object_ids']]
    for residual in [False,True]:
        fig,axes=plt.subplots(len(indices),5,figsize=(12,2.35*len(indices)),squeeze=False)
        for row,(i,limits) in enumerate(zip(indices,selection['limits_mjy_sr'])):
            for col,name in enumerate(['target',*DISPLAY]):
                im=frozen.target[i] if name=='target' else images[name][i]
                im=np.ma.array(im,mask=~frozen.valid[i])
                if residual and name!='target':
                    axes[row,col].imshow(im-frozen.target[i],origin='lower',cmap='RdBu_r',vmin=-limits['residual'],vmax=limits['residual'])
                else:
                    axes[row,col].imshow(im,origin='lower',cmap='gray',norm=AsinhNorm(linear_width=limits['stretch'],vmin=limits['low'],vmax=limits['high']))
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if col==0:axes[row,col].set_ylabel(str(frozen.ids[i]))
        for ax,label in zip(axes[0],['JWST reference',*LABELS]):ax.set_title(label,fontsize=9)
        fig.tight_layout();save_plot(fig,out/('fixed_residuals' if residual else 'fixed_images'))


def main(args):
    torch.set_num_threads(2)
    if args.wait:wait_for_runs(PATHS.values())
    statuses={name:read(p/'status.json') for name,p in PATHS.items()}
    assert all(s['complete'] and s['updates']==22270 and s['parent_unchanged'] for s in statuses.values())
    contracts={name:read(p/'contract.json') for name,p in PATHS.items()}
    epochs={name:[read(p/f'epoch-{i:02d}.json') for i in range(11)] for name,p in PATHS.items()}
    for name in ['conditioned','deconv']:
        for key in ['parent_sha256','data_hashes','masks_sha256']:
            assert contracts[name][key]==contracts['control'][key]
        for i in range(1,11):
            assert epochs[name][i]['training_stream_sha256']==epochs['control'][i]['training_stream_sha256']
    assert contracts['conditioned']['source_hashes']==contracts['deconv']['source_hashes']
    assert contracts['conditioned']['features_sha256']==contracts['deconv']['features_sha256']
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ['compare_psf_redesign.py','residual_refinement.py','spatial_spectra.py',
                 'cached_shape_comparison.py','compare_residual_refinements.py','robust_shape_measurements.py','shape_diagnostics.py']:
        shutil.copy2(ROOT/'code'/name,out/name)
    frozen=FrozenDiagnostics();images={'legacy':frozen.baseline};arrays={}
    _,arrays['legacy']=frozen.measure(images['legacy'])
    for name,p in PATHS.items():
        best=statuses[name]['best_epoch']
        assert best==min(range(11),key=lambda i:epochs[name][i]['validation']['source_mae'])
        for suffix,epoch,filename in [('best',best,'best_predictions.npy'),('last',10,'predictions-10.npy')]:
            assert sha(p/(suffix+'.pt'))==statuses[name][suffix+'_checkpoint_sha256']
            key=name+'_'+suffix
            images[key]=np.load(p/filename)
            with np.load(p/f'measurements-{epoch:02d}.npz') as saved:
                np.testing.assert_array_equal(saved['object_ids'],frozen.ids)
                arrays[key]={k:saved[k] for k in saved.files if k!='object_ids'}
            # Recompute pixel errors directly from the saved predictions.
            for region,mask in [('source',frozen.source),('outside',frozen.valid & ~frozen.source)]:
                delta=np.where(mask,images[key].astype(float)-frozen.target,0.)
                np.testing.assert_allclose(np.sqrt((delta**2).sum((1,2))/mask.sum((1,2))),arrays[key][region+'_rmse'],rtol=1e-12,atol=1e-12)
    for name in images:
        delta=np.where(frozen.valid,images[name].astype(float)-frozen.target,0.)
        arrays[name]['field_rmse']=np.sqrt((delta**2).sum((1,2))/frozen.valid.sum((1,2)))
    methods=list(images)
    pairs=[(m,b) for m in methods if m.startswith(('conditioned','deconv')) for b in ['legacy','control_best' if m.endswith('best') else 'control_last']]
    pairs += [('deconv_best','conditioned_best'),('deconv_last','conditioned_last')]
    errors={name:{k:block_summary(a[k],frozen.blocks) for k in ['source_mae','source_rmse','outside_rmse','field_rmse']} for name,a in arrays.items()}
    bg_keep,bg=background_spectra(images,frozen)
    spectra={'source':{name:{k:block_summary(a[k],frozen.blocks[frozen.primary]) for k in ['log_power_ratio','correlation']} for name,a in arrays.items()},
             'outside':{name:{k:block_summary(a[k],frozen.blocks[bg_keep]) for k in ['log_power_ratio','correlation']} for name,a in bg.items()}}
    paired={}
    for a,b in pairs:
        row={k:block_summary(arrays[a][k]-arrays[b][k],frozen.blocks) for k in errors[a]}
        for region,values,keep in [('source',arrays,frozen.primary),('outside',bg,bg_keep)]:
            row[region+'_correlation_difference']=block_summary(values[a]['correlation']-values[b]['correlation'],frozen.blocks[keep])
            row[region+'_absolute_log_power_error_difference']=block_summary(np.abs(values[a]['log_power_ratio'])-np.abs(values[b]['log_power_ratio']),frozen.blocks[keep])
        paired[a+'__minus__'+b]=row
    shapes={};measurements={};shape_arrays={}
    for kind,fn,reasons in [('fixed',fixed_shapes,REASONS),('adaptive',adaptive,HSM_REASONS)]:
        print('Measuring',kind,'shapes for all selected/final checkpoints',flush=True)
        values={'target':fn(frozen.target,frozen.valid)}
        for name in methods:
            values[name]={k:arrays[name]['shape_'+k] for k in ['e','good','flags']} if kind=='fixed' else fn(images[name],frozen.valid)
        common=np.logical_and.reduce([v['good'] for v in values.values()])
        shapes[kind]=summarize(values,frozen.blocks,frozen.ids,methods,reasons)
        shapes[kind]['paired_all']={a+'__minus__'+b:shape_difference(values[a]['e'],values[b]['e'],values['target']['e'],common,frozen.blocks) for a,b in pairs}
        # Pairwise-success sensitivity check, keeping the population explicit.
        shapes[kind]['pairwise_success_check']={}
        for a,b in pairs:
            good=values[a]['good'] & values[b]['good'] & values['target']['good']
            shapes[kind]['pairwise_success_check'][a+'__minus__'+b]=dict(objects=int(good.sum()),**shape_difference(values[a]['e'],values[b]['e'],values['target']['e'],good,frozen.blocks))
        measurements[kind]=values
        shape_arrays.update({kind+'__'+name+'__'+k:v for name,data in values.items() for k,v in data.items()})
    np.savez_compressed(out/'shapes.npz',object_ids=frozen.ids,**shape_arrays)
    snr_path=ROOT/'paper_figures/inputs/shape_snr_v02.npz'
    with np.load(snr_path) as saved:
        np.testing.assert_array_equal(saved['object_ids'],frozen.ids)
        snr_bin=saved['bin_index']
    with np.load(ROOT/'runs/psf-features-v01/features.npz') as saved:
        np.testing.assert_array_equal(saved['validation_ids'],frozen.ids)
        stack=saved['validation_kernel_index']
    subgroups={}
    for group,selection in [(f'snr_bin_{i}',snr_bin==i) for i in range(5)]+[(f'psf_stack_{i}',stack==i) for i in np.unique(stack)]:
        rec=dict(objects=int(selection.sum()),errors={},shapes={})
        for name in DISPLAY:
            rec['errors'][name]={k:float(arrays[name][k][selection].mean()) for k in ['source_rmse','outside_rmse']}
        for kind,values in measurements.items():
            good=selection & np.logical_and.reduce([values[n]['good'] for n in ['target',*DISPLAY]])
            rec['shapes'][kind]=dict(common_success=int(good.sum()),success_counts={n:int((selection & values[n]['good']).sum()) for n in ['target',*DISPLAY]},
                rmse={n:float(np.sqrt(np.mean((values[n]['e'][good]-values['target']['e'][good])**2))) if good.any() else None for n in DISPLAY})
        subgroups[group]=rec
    report=dict(complete=True,validation_objects=len(frozen.ids),source_spectrum_objects=int(frozen.primary.sum()),
        outside_spectrum_objects=int(bg_keep.sum()),status=statuses,methods=methods,errors=errors,spectra=spectra,
        paired=paired,shapes=shapes,subgroups=subgroups,training_streams_match=True,
        bands=list(BANDS),bootstrap=dict(draws=2000,seed=20261017,unit='spatial block'),
        input_hashes={str(p/f):sha(p/f) for p in PATHS.values() for f in ['contract.json','best.pt','last.pt','best_predictions.npy','predictions-10.npy']},
        source_hashes={p.name:sha(p) for p in out.glob('*.py')},
        snr_inputs_sha256=sha(snr_path),shapes_sha256=sha(out/'shapes.npz'),
        limitations=['Validation selection and one parent seed; bootstrap does not account for checkpoint/model selection or training variance.',
            'Same initial parent; improvements versus the control are more informative than versus the parent alone.',
            'PSF proxy assumptions and only four near-identical kernels; no claim of calibrated deconvolution or shear.',
            'Source/background power includes reference noise and mask mixing; more power alone is not sharper recovered structure.',
            'Morphology RMSE is conditional on common measurement success. Full denominators and pairwise-success checks retained.'])
    write(out/'comparison.json',report)
    plots(out,report,images,arrays,frozen,epochs)
    lines=['# PSF-aware ResNet comparison','',
        'All ten epochs completed. Selection uses source MAE with the unchanged parent eligible. Validation only.','',
        '| Model | Epoch | Source RMSE | Outside RMSE | Fixed shape RMSE | Adaptive shape RMSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in methods:
        epoch=0 if name=='legacy' else statuses[name.split('_')[0]]['best_epoch'] if name.endswith('best') else 10
        vals=[errors[name][k]['mean'] for k in ['source_rmse','outside_rmse']]+[shapes[k]['methods'][name]['component_rmse']['mean'] for k in ['fixed','adaptive']]
        lines.append(f'| {name} | {epoch} | '+' | '.join(f'{v:.7f}' for v in vals)+' |')
    lines+=['','Compare against both original and extra-training control; inspect paired intervals and failure counts in comparison.json.','',
        '![RMSE](rmse.png)','![Sharpness and structure](sharpness_structure.png)','![Shapes](shapes.png)',
        '![Learning curves](learning_curves.png)','![Fixed examples](fixed_images.png)','![Residuals](fixed_residuals.png)','',
        *['- '+s for s in report['limitations']]]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(out/'status.json',dict(complete=True,comparison_sha256=sha(out/'comparison.json'),
        artifacts={p.name:sha(p) for p in out.glob('*.png')}))
    print('Complete:',out/'report.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'runs/psf-redesign-comparison-v01')
    main(p.parse_args())
