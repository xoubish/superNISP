"""Completed-checkpoint validation robustness; models require tmux and online W&B."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import unittest

import galsim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from shape_diagnostics import METHODS,REASONS,interpolation
from robust_shape_measurements import (HSM_REASONS,fixed_moments,adaptive_moments,
    common_mask,summary,nearby_peak_proxy,scenarios)
from train_paired import persistent_session_required

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convergence(training,output):
    records=[json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines()]
    best=min(records,key=lambda r:r['validation/mae'])
    checkpoint=torch.load(training/'last.pt',map_location='cpu',weights_only=False)
    metrics=dict(best_epoch=best['epoch'],best_validation_mae=best['validation/mae'],
        last_learning_rate=checkpoint['optimizer']['param_groups'][0]['lr'],
        improvements_to_best_percent={str(epoch):100*(1-best['validation/mae']/records[epoch-1]['validation/mae']) for epoch in [30,40,45,48]},
        interpretation='Inspect the measured late-epoch improvements to assess convergence under the completed OneCycle schedule; the best epoch alone does not establish convergence. A continuation requires a separately recorded schedule.')
    (output/'convergence.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    epochs=[r['epoch'] for r in records]
    for ax in axes:
        ax.plot(epochs,[r['train_epoch_mae'] for r in records],label='Training MAE')
        ax.plot(epochs,[r['validation/mae'] for r in records],label='Validation MAE')
        ax.axvline(best['epoch'],color='gray',linestyle=':',label=f"Best epoch {best['epoch']}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Masked normalized-intensity MAE')
        ax.legend()
    axes[1].set_xlim(40,50)
    late=[r[key] for r in records if r['epoch']>=40 for key in ['train_epoch_mae','validation/mae']]
    padding=max((max(late)-min(late))*.1,1e-4)
    axes[1].set_ylim(min(late)-padding,max(late)+padding)
    axes[0].set_title('Completed 50-epoch run')
    axes[1].set_title('Late training')
    fig.tight_layout()
    fig.savefig(output/'convergence.png',dpi=130)
    plt.close(fig)
    return metrics


def plot_results(report,path):
    names=list(report['scenarios'])
    fig,axes=plt.subplots(1,2,figsize=(16,max(8,.4*len(names))))
    y=np.arange(len(names))
    for baseline,shift,color in [('native',-.13,'tab:blue'),('bilinear',.13,'tab:orange')]:
        values=[]
        lower=[]
        upper=[]
        for name in names:
            result=report['scenarios'][name]['matched_with_nominal']['methods'].get(baseline,{}).get('sr_comparison',{})
            values.append(result.get('improvement_percent',np.nan))
            interval=result.get('ci95_percent') or [np.nan,np.nan]
            lower.append(interval[0]);upper.append(interval[1])
        axes[0].hlines(y+shift,lower,upper,color=color)
        axes[0].plot(values,y+shift,'o',label=baseline,color=color)
    axes[0].axvline(0,color='black',linewidth=.7)
    axes[0].set_yticks(y,names,fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('SR component-RMSE reduction (%) vs baseline\nSame galaxies within each scenario / nominal pair; 95% block interval')
    axes[0].legend()
    sr=[report['scenarios'][n]['all_objects']['success_counts']['sr']/report['objects'] for n in names]
    jwst=[report['scenarios'][n]['all_objects']['success_counts']['jwst']/report['objects'] for n in names]
    axes[1].barh(y-.15,sr,height=.3,label='SR success')
    axes[1].barh(y+.15,jwst,height=.3,label='JWST success')
    axes[1].set_yticks(y,names,fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0,1)
    axes[1].set_xlabel('Fraction of all validation objects')
    axes[1].legend()
    fig.suptitle(f"Observed-shape robustness — epoch {report['checkpoint_epoch']}, validation only",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(path,dpi=130)
    plt.close(fig)


def main(args):
    persistent_session_required()
    torch.set_num_threads(2)
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    training=args.training.resolve()
    if not json.loads((training/'status.json').read_text())['complete']:
        raise ValueError('Only completed training runs are supported')
    shutil.copy2(training/'best.pt',output/'checkpoint.pt')
    checkpoint=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
    for name,expected in checkpoint['signature']['source_hashes'].items():
        if sha(ROOT/'code'/name)!=expected: raise ValueError(f'Training source changed: {name}')
    for name,expected in checkpoint['signature']['data_hashes'].items():
        if sha(args.data/name)!=expected: raise ValueError(f'Data changed: {name}')
    nominal_reference=args.nominal_reference.resolve()
    nominal_provenance=json.loads((nominal_reference/'provenance.json').read_text())
    if nominal_provenance['checkpoint_sha256']!=sha(output/'checkpoint.pt'):
        raise ValueError('Nominal reference belongs to a different checkpoint')
    if nominal_provenance['training_signature']!=checkpoint['signature']:
        raise ValueError('Nominal reference training signature differs')
    if nominal_provenance['evaluation_sources']['shape_diagnostics.py']!=sha(ROOT/'code/shape_diagnostics.py'):
        raise ValueError('Nominal reference measurement source differs')
    suite=scenarios()
    sources=['run_shape_robustness.py','robust_shape_measurements.py','test_shape_robustness.py','shape_diagnostics.py']
    for name in sources: shutil.copy2(ROOT/'code'/name,output/name)
    protocol=dict(checkpoint_epoch=checkpoint['epoch'],checkpoint_sha256=sha(output/'checkpoint.pt'),
        sources={name:sha(ROOT/'code'/name) for name in sources},training_signature=checkpoint['signature'],
        objects_csv_sha256=sha(args.data/'objects.csv'),galsim=galsim.__version__,numpy=np.__version__,torch=torch.__version__,
        scenarios=suite,split='validation',bootstrap=1000,seed=20261016,
        nominal_reference=str(nominal_reference),nominal_measurements_sha256=sha(nominal_reference/'measurements.npz'),
        hsm='Observed adaptive moments; central 105 or 155 target-pixel crops, matched native footprints; masks; central radius 25 coverage; max centroid offset 10 target pixels. No PSF correction.',
        background='Measurement-only signed offsets ±0.1 times per-object outer sky RMS; not calibrated background uncertainty. NISP offsets affect only native/interpolated baseline measurements; SR/JWST offsets affect that image only.',
        noise='Add independent Gaussian noise of 0.25 times supplied NISP pixel RMS to valid input pixels, re-run SR and interpolation; three fixed seeds. A 6.25% variance increase under an independent-noise approximation; no JWST noise model.',
        crowding='Input-only local peak above 5 RMS at radius 3–9 NISP pixels; proxy, not a deblender or isolation certification.',
        population='Within each scenario methods share successes; compare with nominal on their common IDs; also report intersection of all scenarios. Never tune scenarios against results.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with wandb.init(project='superNISP-corrected',name=output.name,job_type='shape-robustness',
                    mode='online',dir=str(output),config=protocol) as run:
        if run.settings.mode!='online' or not run.url: raise RuntimeError('Online W&B required')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        tests=unittest.defaultTestLoader.discover(str(ROOT/'code'),pattern='test_shape_robustness.py')
        test_result=unittest.TextTestRunner(verbosity=2).run(tests)
        if not test_result.wasSuccessful(): raise RuntimeError('Robustness tests failed')
        run.summary['tests_passed']=test_result.testsRun
        conv=convergence(training,output)
        run.log({'convergence/curves':wandb.Image(str(output/'convergence.png'))})
        config=checkpoint['signature']['config']
        dataset=PairedSample(args.data,'validation',config['representation'])
        catalog={int(r['row']):r for r in csv.DictReader((args.data/'objects.csv').open()) if r['split']=='validation'}
        model=EuclidToJWSTSuperResolution(head=config['head']).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        generators={seed:torch.Generator(device=device).manual_seed(20261016+seed) for seed in range(3)}
        collected={scenario:{m:[] for m in ['jwst',*METHODS]} for scenario in suite}
        ids,rows,peak_flags=[],[],[]
        with torch.no_grad():
            for batch in DataLoader(dataset,batch_size=8,num_workers=0):
                x=batch['nisp'].to(device)
                linear=3*torch.sinh(x) if dataset.representation=='asinh' else x
                native=linear*dataset.norm['nisp_scale']
                native_mask=batch['nisp_valid'].to(device)
                rms=batch['nisp_rms_linear'].to(device)*dataset.norm['nisp_scale']
                target=dataset.jwst_to_mjy_sr(batch['jwst'].to(device))
                target_mask=batch['jwst_valid'].to(device)
                prediction=dataset.jwst_to_mjy_sr(model(x))
                images={'jwst':(target,target_mask,1.),'sr':(prediction,target_mask,1.),'native':(native,native_mask,5.)}
                for method in ['bilinear','bicubic']:
                    interp,mask=interpolation(native,native_mask,method)
                    images[method]=(interp,mask,1.)
                row_list=batch['row'].tolist()
                nisp_sky=torch.tensor([float(catalog[r]['nisp_sky_sigma'])*1e-12/float(catalog[r]['nisp_pixel_sr']) for r in row_list],device=device)[:,None,None,None]
                jwst_sky=torch.tensor([float(catalog[r]['jwst_sky_sigma']) for r in row_list],device=device)[:,None,None,None]
                for name,settings in suite.items():
                    options=settings.copy()
                    variants=images.copy()
                    if 'noise_seed' in options:
                        seed=options.pop('noise_seed')
                        noise=torch.randn(x.shape,device=device,generator=generators[seed])
                        noisy_linear=torch.where(native_mask,linear+.25*batch['nisp_rms_linear'].to(device)*noise,0.)
                        noisy_input=torch.asinh(noisy_linear/3) if dataset.representation=='asinh' else noisy_linear
                        noisy_native=noisy_linear*dataset.norm['nisp_scale']
                        variants['native']=(noisy_native,native_mask,5.)
                        variants['sr']=(dataset.jwst_to_mjy_sr(model(noisy_input)),target_mask,1.)
                        for method in ['bilinear','bicubic']:
                            interp,mask=interpolation(noisy_native,native_mask,method)
                            variants[method]=(interp,mask,1.)
                    if 'background' in options:
                        group,factor=options.pop('background')
                        methods=['native','bilinear','bicubic'] if group=='nisp' else [group]
                        delta=factor*(nisp_sky if group=='nisp' else jwst_sky)
                        for method in methods:
                            image,mask,step=variants[method]
                            variants[method]=(image+delta,mask,step)
                    adaptive=options.pop('adaptive',False)
                    for method,(image,mask,step) in variants.items():
                        if adaptive:
                            measured=adaptive_moments(image,mask,step,**options)
                        else:
                            measured={k:v.cpu().numpy() for k,v in fixed_moments(image,mask,step,**options).items()}
                        collected[name][method].append(measured)
                peak_flags.extend(nearby_peak_proxy(native,rms,native_mask).cpu().tolist())
                rows.extend(row_list)
                ids.extend(batch['object_id'].tolist())
                if len(rows)%128==0:
                    run.log({'evaluated_objects':len(rows)})
                    print(f'Robustness suite: {len(rows)}/{len(dataset)} objects',flush=True)
        all_measurements={name:{m:{k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
                           for m,parts in methods.items()} for name,methods in collected.items()}
        for row,object_id in zip(rows,ids): assert int(catalog[row]['id_classic'])==object_id
        blocks=np.array([catalog[row]['spatial_block'] for row in rows])
        crowded=np.array(peak_flags,dtype=bool)
        # Confirm that the nominal result still matches the frozen, completed report.
        with np.load(nominal_reference/'measurements.npz') as previous:
            np.testing.assert_array_equal(previous['object_ids'],ids)
            np.testing.assert_array_equal(previous['rows'],rows)
            np.testing.assert_array_equal(previous['blocks'],blocks)
            for method,measurement in all_measurements['fixed_nominal'].items():
                np.testing.assert_allclose(measurement['e'],previous[f'{method}_e'],rtol=1e-6,atol=1e-7)
                np.testing.assert_array_equal(measurement['good'],previous[f'{method}_good'])
                np.testing.assert_array_equal(measurement['flags'],previous[f'{method}_flags'])
        global_common=np.logical_and.reduce([common_mask(v) for v in all_measurements.values()])
        nominal_common=common_mask(all_measurements['fixed_nominal'])
        report=dict(wandb_url=run.url,checkpoint_epoch=checkpoint['epoch'],objects=len(rows),convergence=conv,
                    global_common_count=int(global_common.sum()),scenarios={})
        for name,measurements in all_measurements.items():
            reasons=HSM_REASONS if suite[name].get('adaptive') else REASONS
            both=nominal_common & common_mask(measurements)
            report['scenarios'][name]=dict(
                all_objects=summary(measurements,blocks,reasons),
                matched_with_nominal=summary(measurements,blocks,reasons,selection=both),
                nominal_on_same_ids=summary(all_measurements['fixed_nominal'],blocks,selection=both),
                global_matched=summary(measurements,blocks,reasons,selection=global_common),
                nearby_peak=summary(measurements,blocks,reasons,selection=crowded & both),
                no_nearby_peak=summary(measurements,blocks,reasons,selection=~crowded & both),
                common_ids=np.asarray(ids)[both].tolist())
        arrays={f'{name}__{method}__{key}':value for name,methods in all_measurements.items()
                for method,values in methods.items() for key,value in values.items()}
        np.savez_compressed(output/'measurements.npz',object_ids=ids,rows=rows,blocks=blocks,
                            nearby_peak=crowded,global_common=global_common,**arrays)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        plot_results(report,output/'robustness.png')
        table=wandb.Table(columns=['scenario','same_objects_as_nominal','SR_successes_all','JWST_successes_all',
                                  'SR_rmse','native_rmse','bilinear_rmse','gain_vs_native_percent','gain_vs_bilinear_percent'])
        for name,record in report['scenarios'].items():
            metrics=record['matched_with_nominal']
            methods=metrics['methods']
            table.add_data(name,metrics['common_successes'],record['all_objects']['success_counts']['sr'],
                record['all_objects']['success_counts']['jwst'],
                methods.get('sr',{}).get('component_rmse'),methods.get('native',{}).get('component_rmse'),
                methods.get('bilinear',{}).get('component_rmse'),
                methods.get('native',{}).get('sr_comparison',{}).get('improvement_percent'),
                methods.get('bilinear',{}).get('sr_comparison',{}).get('improvement_percent'))
        run.log({'robustness/comparison':wandb.Image(str(output/'robustness.png')),'robustness/scenarios':table,'evaluated_objects':len(rows)})
        run.summary['global_common_count']=int(global_common.sum())
        run.summary['nominal_reproduced']=True
        run.summary['completed_scenarios']=len(suite)
        (output/'status.json').write_text(json.dumps(dict(complete=True,objects=len(rows),scenarios=len(suite),wandb_url=run.url),indent=2))
        dataset.close()
        print('All robustness scenarios completed and nominal measurements reproduced.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nominal-reference',type=Path,default=ROOT/'runs/validation-shapes-full50-best')
    p.add_argument('--device',default='cuda:1')
    main(p.parse_args())
