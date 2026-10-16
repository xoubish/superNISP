"""Evaluate an immutable checkpoint on validation only, with paired baselines.

Run in tmux with online W&B. Never selects or changes the training checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from shape_diagnostics import (METHODS,REASONS,diagnose_moments,interpolation,
                               snr_proxy,paired_summary,binned_summaries)
from train_paired import persistent_session_required

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_summary(summary,bins,path):
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    methods=list(summary['methods'])
    axes[0,0].bar(methods,[summary['methods'][m]['component_rmse'] for m in methods])
    axes[0,0].set_ylabel('Ellipticity component RMSE')
    axes[0,0].set_title(f"Same {summary['common_successes']} successful objects for every method")
    success=summary['success_counts']
    axes[0,1].bar(list(success),[success[m]/summary['objects'] for m in success])
    axes[0,1].set_ylim(0,1)
    axes[0,1].set_ylabel('Success fraction of all validation objects')
    axes[0,1].set_title(f"Reference + baselines eligible: {summary['reference_baselines_eligible']}")
    for ax,key in zip(axes[1],['nisp_weighted_snr_proxy','ACS_F814W_MAG']):
        entries=bins[key]
        for method in ['native','bilinear','bicubic','round_control']:
            vals=[entry['methods'].get(method,{}).get('component_rmse',np.nan) for entry in entries]
            ax.plot(range(len(entries)),vals,'o--',label=method)
        vals=[entry['methods'].get('sr',{}).get('component_rmse',np.nan) for entry in entries]
        ax.plot(range(len(entries)),vals,'o-',label='sr',linewidth=2)
        ax.set_xticks(range(len(entries)),[f"{entry['range']}\nn={entry['common_successes']}" for entry in entries],fontsize=8)
        ax.set_title(key)
        ax.set_ylabel('Component RMSE on common successes per bin')
        ax.legend(fontsize=7)
    fig.suptitle('Validation checkpoint diagnostic — fixed signed moments; not calibrated shear',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(path,dpi=140)
    plt.close(fig)


def write_table(path,properties,measurements,rows,ids,blocks,snr):
    with path.open('w',newline='') as stream:
        fields=['row','object_id','spatial_block','nisp_weighted_snr_proxy',*properties]
        for method in measurements:
            fields += [f'{method}_{name}' for name in ['success','failure_reasons','e1','e2','flux','qxx','qyy','qxy','determinant']]
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for i,row in enumerate(rows):
            record=dict(row=int(row),object_id=int(ids[i]),spatial_block=str(blocks[i]),nisp_weighted_snr_proxy=float(snr[i]))
            record.update({k:values[i] for k,values in properties.items()})
            for method,result in measurements.items():
                record[f'{method}_success']=bool(result['good'][i])
                record[f'{method}_failure_reasons']=';'.join(reason for j,reason in enumerate(REASONS) if result['flags'][i,j])
                record[f'{method}_e1'],record[f'{method}_e2']=result['e'][i]
                for name in ['flux','qxx','qyy','qxy','determinant']:
                    record[f'{method}_{name}']=float(result[name][i])
            writer.writerow(record)


def main(args):
    persistent_session_required()
    torch.set_num_threads(2)
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    # last.pt is atomically replaced by the trainer. An open copy reads one
    # completed epoch even if the trainer replaces the filename during copying.
    snapshot=output/'checkpoint.pt'
    shutil.copyfile(args.checkpoint,snapshot)
    checkpoint=torch.load(snapshot,map_location='cpu',weights_only=False)
    source=checkpoint['signature']['source_hashes']
    for name,expected in source.items():
        if sha(ROOT/'code'/name)!=expected:
            raise ValueError(f'Checkpoint source mismatch: {name}')
    for name,expected in checkpoint['signature']['data_hashes'].items():
        if sha(args.data/name)!=expected:
            raise ValueError(f'Checkpoint sample mismatch: {name}')
    config=checkpoint['signature']['config']
    for name in ['evaluate_paired_shapes.py','shape_diagnostics.py','test_shape_diagnostics.py']:
        shutil.copy2(ROOT/'code'/name,output/name)
    provenance=dict(checkpoint_sha256=sha(snapshot),epoch=checkpoint['epoch'],updates=checkpoint['step'],
        training_wandb_id=checkpoint['wandb_id'],training_signature=checkpoint['signature'],
        evaluation_sources={name:sha(ROOT/'code'/name) for name in ['evaluate_paired_shapes.py','shape_diagnostics.py','test_shape_diagnostics.py']},
        objects_csv_sha256=sha(args.data/'objects.csv'),device=args.device,
        contract=dict(split='validation',sigma_target_pixels=10,aperture_radius_target_pixels=25,
          intensity='signed, sky-subtracted MJy/sr',native_pixel_step=5,
          common_population='JWST and all four methods pass; baseline eligibility excludes SR',
          bootstrap='paired resampling of saved spatial blocks, percentile 95% interval',
          bootstrap_replicates=args.bootstrap,seed=20261016,
          snr='Gaussian-weighted native flux / quadrature RMS; ignores correlated noise',
          catalog_radius_units='original catalog units; not claimed to be angular size',
          limitations=['Validation diagnostic of an intermediate checkpoint, not a final test result.',
            'Cross-band Y/F115W reference; no PSF correction or shear calibration.',
            'Common-success population may change between checkpoints; compare common IDs across runs.',
            'Block bootstrap is conditional on this field and these moment/selection choices.',
            'Native aperture uses pixel-center quadrature at the same angular radius.',
            'Bicubic support requires all 16 stencil pixels, even zero-weight neighbors.']))
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with wandb.init(project='superNISP-corrected',name=output.name,job_type='validation-analysis',
                    mode='online',dir=str(output),config=provenance) as run:
        if run.settings.mode!='online' or not run.url:
            raise RuntimeError('Online W&B required')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        tests=unittest.defaultTestLoader.discover(str(ROOT/'code'),pattern='test_shape_diagnostics.py')
        result=unittest.TextTestRunner(verbosity=2).run(tests)
        if not result.wasSuccessful():
            raise RuntimeError('Shape diagnostic regression tests failed')
        run.summary['regression_tests_passed']=result.testsRun
        dataset=PairedSample(args.data,'validation',config['representation'])
        catalog={int(r['row']):r for r in csv.DictReader((args.data/'objects.csv').open()) if r['split']=='validation'}
        assert len(catalog)==len(dataset) and len({r['id_classic'] for r in catalog.values()})==len(dataset)
        model=EuclidToJWSTSuperResolution(head=config['head']).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        loader=DataLoader(dataset,batch_size=8,num_workers=0)
        collected={method:[] for method in ['jwst',*METHODS]}
        rows,ids,snrs=[],[],[]
        with torch.no_grad():
            for batch in loader:
                x=batch['nisp'].to(device)
                linear=3*torch.sinh(x) if config['representation']=='asinh' else x
                native=linear*dataset.norm['nisp_scale']
                native_mask=batch['nisp_valid'].to(device)
                target=dataset.jwst_to_mjy_sr(batch['jwst'].to(device))
                target_mask=batch['jwst_valid'].to(device)
                prediction=dataset.jwst_to_mjy_sr(model(x))
                images={'jwst':(target,target_mask,1.),'sr':(prediction,target_mask,1.),'native':(native,native_mask,5.)}
                for method in ['bilinear','bicubic']:
                    image,mask=interpolation(native,native_mask,method)
                    images[method]=(image,mask,1.)
                for method,(image,mask,step) in images.items():
                    measured=diagnose_moments(image,mask,step)
                    collected[method].append({k:v.cpu().numpy() for k,v in measured.items()})
                snrs.extend(snr_proxy(native,batch['nisp_rms_linear'].to(device)*dataset.norm['nisp_scale'],native_mask).cpu().tolist())
                rows.extend(batch['row'].tolist())
                ids.extend(batch['object_id'].tolist())
                if len(rows)%256==0:
                    run.log({'evaluated_objects':len(rows)})
                    print(f'Evaluated {len(rows)}/{len(dataset)} validation objects',flush=True)
        measurements={method:{k:np.concatenate([part[k] for part in parts]) for k in parts[0]} for method,parts in collected.items()}
        for row,object_id in zip(rows,ids):
            assert int(catalog[row]['id_classic'])==object_id
        blocks=np.array([catalog[row]['spatial_block'] for row in rows])
        properties={name:np.array([float(catalog[row][name]) for row in rows]) for name in ['ACS_F814W_MAG','FLUX_RADIUS','ez_z_phot']}
        np.savez_compressed(output/'measurements.npz',rows=rows,object_ids=ids,blocks=blocks,snr_proxy=snrs,
            **properties,**{f'{m}_{k}':v for m,result in measurements.items() for k,v in result.items()})
        write_table(output/'measurements.csv',properties,measurements,rows,ids,blocks,snrs)
        summary=paired_summary(measurements,blocks,bootstrap=args.bootstrap)
        bins=binned_summaries(measurements,blocks,{**properties,'nisp_weighted_snr_proxy':snrs},bootstrap=args.bootstrap)
        report=dict(checkpoint_epoch=checkpoint['epoch'],checkpoint_sha256=provenance['checkpoint_sha256'],
                    wandb_url=run.url,summary=summary,bins=bins,contract=provenance['contract'])
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        plot_summary(summary,bins,output/'summary.png')
        metrics={'analysis/common_successes':summary['common_successes'],'analysis/objects':summary['objects']}
        for m,values in summary['methods'].items():
            metrics[f'analysis/{m}_rmse']=values['component_rmse']
        run.log(metrics|{'analysis/comparisons':wandb.Image(str(output/'summary.png'))})
        run.summary.update(metrics)
        run.summary['checkpoint_epoch']=checkpoint['epoch']
        failure_table=wandb.Table(columns=['object_id','row','sr_failure_reasons','jwst_success','nisp_snr_proxy'])
        for i in np.flatnonzero(~measurements['sr']['good']):
            failure_table.add_data(int(ids[i]),int(rows[i]),'; '.join(r for j,r in enumerate(REASONS) if measurements['sr']['flags'][i,j]),bool(measurements['jwst']['good'][i]),float(snrs[i]))
        run.log({'analysis/sr_failures':failure_table})
        (output/'status.json').write_text(json.dumps(dict(complete=True,objects=len(rows),wandb_url=run.url),indent=2))
        print(json.dumps(summary,indent=2),flush=True)
        dataset.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    if args.bootstrap<0: p.error('bootstrap must be nonnegative')
    main(args)
