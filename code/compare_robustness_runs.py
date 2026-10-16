"""Compare architectures on identical objects in each frozen robustness scenario."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import wandb

from compare_shape_runs import compare
from robust_shape_measurements import HSM_REASONS
from shape_diagnostics import REASONS, METHODS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_scenario(before, after, name, adaptive, bootstrap=1000):
    def extract(data):
        arrays={key:data[key] for key in ['object_ids','rows','blocks']}
        for method in ['jwst',*METHODS]:
            for field in ['e','good','flags']:
                arrays[f'{method}_{field}']=data[f'{name}__{method}__{field}']
        return arrays
    first,second=extract(before),extract(after)
    result=compare(first,second,bootstrap)
    # compare() computes the common population and paired bootstrap correctly for
    # either estimator; replace its fixed-moment failure labels for HSM arrays.
    reasons=HSM_REASONS if adaptive else REASONS
    selected=set(result['common_object_ids'])
    for key,data in [('before',first),('after',second)]:
        mask=np.array([i in selected for i in data['object_ids']])
        result[key]['failure_reasons']={method:{reason:int((mask & data[f'{method}_flags'][:,j]).sum())
            for j,reason in enumerate(reasons)} for method in ['jwst',*METHODS]}
    return result


def main(args):
    protocols=[json.loads((directory/'protocol.json').read_text()) for directory in [args.before,args.after]]
    for key in ['scenarios','split','bootstrap','seed','hsm','background','noise','crowding','population',
                'objects_csv_sha256','galsim','numpy','torch']:
        if protocols[0][key]!=protocols[1][key]:
            raise ValueError(f'Robustness protocol differs: {key}')
    for name in ['robust_shape_measurements.py','shape_diagnostics.py']:
        if protocols[0]['sources'][name]!=protocols[1]['sources'][name]:
            raise ValueError(f'Measurement source changed: {name}')
    if protocols[0]['training_signature']['data_hashes']!=protocols[1]['training_signature']['data_hashes']:
        raise ValueError('Training data hashes differ')
    arrays=[]
    for directory in [args.before,args.after]:
        if not json.loads((directory/'status.json').read_text())['complete']:
            raise ValueError('Robustness analysis incomplete')
        with np.load(directory/'measurements.npz') as data:
            arrays.append(dict(data))
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['compare_robustness_runs.py','compare_shape_runs.py','shape_diagnostics.py','robust_shape_measurements.py']:
        shutil.copy2(Path(__file__).parent/name,args.output/name)
    provenance=dict(inputs=[str(args.before.resolve()),str(args.after.resolve())],
        protocols=protocols,measurements_sha256=[sha(d/'measurements.npz') for d in [args.before,args.after]],
        source_sha256={p.name:sha(p) for p in args.output.glob('*.py')})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with wandb.init(project='superNISP-corrected',name=args.output.name,job_type='robustness-comparison',
                    mode='online',dir=str(args.output),config=provenance) as run:
        if run.settings.mode!='online' or not run.url:
            raise RuntimeError('Online W&B required')
        report=dict(wandb_url=run.url,scenarios={})
        table=wandb.Table(columns=['scenario','common_objects','legacy_rmse','direct5_rmse',
                                  'direct5_improvement_percent','ci95_low','ci95_high','lost','recovered'])
        for name,settings in protocols[0]['scenarios'].items():
            result=compare_scenario(*arrays,name,bool(settings.get('adaptive')),protocols[0]['bootstrap'])
            report['scenarios'][name]=result
            counts=result['counts']
            interval=result['sr_change']['ci95_percent'] or [None,None]
            table.add_data(name,counts['common_across_checkpoints'],
                result['before']['methods'].get('sr',{}).get('component_rmse'),
                result['after']['methods'].get('sr',{}).get('component_rmse'),
                result['sr_change']['improvement_percent'],*interval,
                counts['lost_sr_measurements'],counts['recovered_sr_measurements'])
        run.log({'comparison/scenarios':table})
        run.summary['completed_scenarios']=len(report['scenarios'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('Paired architecture comparisons completed for all robustness scenarios.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
