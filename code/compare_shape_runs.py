"""Compare two validation checkpoints on an identical, explicitly saved population."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import wandb
from shape_diagnostics import METHODS, paired_summary


def compare(before,after,bootstrap=1000):
    before_ids=before['object_ids'].tolist()
    after_ids=after['object_ids'].tolist()
    if len(set(before_ids))!=len(before_ids) or len(set(after_ids))!=len(after_ids):
        raise ValueError('Duplicate object IDs')
    if set(before_ids)!=set(after_ids):
        raise ValueError('Validation populations differ')
    lookup={value:i for i,value in enumerate(after_ids)}
    order=[lookup[value] for value in before_ids]
    aligned={key:values[order] for key,values in after.items()}
    for key in ['rows','blocks']:
        if not np.array_equal(before[key],aligned[key]):
            raise ValueError(f'Object metadata changed: {key}')
    for method in ['jwst','native','bilinear','bicubic']:
        for field in ['e','good','flags']:
            key=f'{method}_{field}'
            if not np.array_equal(before[key],aligned[key],equal_nan=True):
                raise ValueError(f'Reference/baseline diagnostic changed: {key}')
    common=np.ones(len(before_ids),dtype=bool)
    for data in [before,aligned]:
        for method in ['jwst',*METHODS]:
            common &= data[f'{method}_good']
    def unpack(data):
        return {method:{field:data[f'{method}_{field}'] for field in ['e','good','flags']} for method in ['jwst',*METHODS]}
    first=paired_summary(unpack(before),before['blocks'],selection=common,bootstrap=bootstrap)
    second=paired_summary(unpack(aligned),before['blocks'],selection=common,bootstrap=bootstrap)
    delta=dict(improvement_percent=None,ci95_percent=None)
    if common.any():
        ref=before['jwst_e'][common]
        first_error=((before['sr_e'][common]-ref)**2).mean(1)
        second_error=((aligned['sr_e'][common]-ref)**2).mean(1)
        if first_error.mean()>0:
            delta['improvement_percent']=float(100*(1-np.sqrt(second_error.mean()/first_error.mean())))
        labels,inverse=np.unique(before['blocks'][common],return_inverse=True)
        if bootstrap and len(labels)>1:
            first_sum=np.bincount(inverse,weights=first_error)
            second_sum=np.bincount(inverse,weights=second_error)
            draws=np.random.default_rng(20261016).integers(len(labels),size=(bootstrap,len(labels)))
            denominator=first_sum[draws].sum(1)
            valid=denominator>0
            if valid.any():
                ratios=100*(1-np.sqrt(second_sum[draws].sum(1)[valid]/denominator[valid]))
                delta['ci95_percent']=np.percentile(ratios,[2.5,97.5]).tolist()
    eligible=before['jwst_good'].copy()
    for method in METHODS[:-1]: eligible &= before[f'{method}_good']
    counts=dict(total_objects=len(before_ids),reference_baselines_eligible=int(eligible.sum()),
        common_across_checkpoints=int(common.sum()),
        first_sr_failures_on_eligible=int((eligible & ~before['sr_good']).sum()),
        second_sr_failures_on_eligible=int((eligible & ~aligned['sr_good']).sum()),
        recovered_sr_measurements=int((eligible & ~before['sr_good'] & aligned['sr_good']).sum()),
        lost_sr_measurements=int((eligible & before['sr_good'] & ~aligned['sr_good']).sum()))
    return dict(counts=counts,before=first,after=second,sr_change=delta,
                common_object_ids=np.asarray(before_ids)[common].tolist())


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['compare_shape_runs.py','shape_diagnostics.py']:
        shutil.copy2(Path(__file__).parent/name,args.output/name)
    provenances=[json.loads((directory/'provenance.json').read_text()) for directory in [args.before,args.after]]
    if provenances[0]['contract']!=provenances[1]['contract']:
        raise ValueError('Diagnostic contracts differ')
    if provenances[0]['training_signature']['data_hashes']!=provenances[1]['training_signature']['data_hashes']:
        raise ValueError('Sample provenance differs')
    if provenances[0]['evaluation_sources']['shape_diagnostics.py']!=provenances[1]['evaluation_sources']['shape_diagnostics.py']:
        raise ValueError('Moment/statistics source changed')
    with np.load(args.before/'measurements.npz') as data: before=dict(data)
    with np.load(args.after/'measurements.npz') as data: after=dict(data)
    report=compare(before,after,args.bootstrap)
    report['checkpoint_epochs']=[p['epoch'] for p in provenances]
    report['checkpoint_hashes']=[p['checkpoint_sha256'] for p in provenances]
    report['comparison_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with wandb.init(project='superNISP-corrected',name=args.output.name,job_type='checkpoint-comparison',
                    mode='online',dir=str(args.output)) as run:
        if run.settings.mode!='online' or not run.url:
            raise RuntimeError('Online W&B required')
        report['wandb_url']=run.url
        run.summary.update(report['counts'])
        run.summary['sr_improvement_percent']=report['sr_change']['improvement_percent']
        table=wandb.Table(columns=['method','before_rmse','after_rmse'])
        for method in report['before']['methods']:
            table.add_data(method,report['before']['methods'][method]['component_rmse'],report['after']['methods'][method]['component_rmse'])
        run.log({'comparison/identical_population':table})
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in report.items() if k not in ['before','after','common_object_ids']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    if args.bootstrap<0: parser.error('bootstrap must be nonnegative')
    main(args)
