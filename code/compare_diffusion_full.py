"""Final diffusion/ensemble/residual shape comparison on identical validation IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
import wandb
from train_paired import persistent_session_required,sha256
from shape_diagnostics import METHODS


def load(path):
    with np.load(path) as data: return dict(data)


def main(args):
    persistent_session_required()
    if not json.loads((args.training/'status.json').read_text())['complete']:
        raise ValueError('Full diffusion run must complete')
    paths={f'diffusion_seed_{i}':args.training/f'best-sample-{i}-measurements.npz' for i in range(3)}
    paths['diffusion_mean3']=args.training/'best-ensemble-measurements.npz'
    paths['legacy_best49']=Path('runs/validation-shapes-full50-best/measurements.npz')
    direct=Path('runs/validation-shapes-direct5-best/measurements.npz')
    if direct.exists(): paths['direct5_best']=direct
    datasets={name:load(path) for name,path in paths.items()}
    reference=datasets['diffusion_seed_0']
    identities=reference['object_ids'].tolist()
    common=np.ones(len(identities),bool)
    deviations={}
    for name,data in datasets.items():
        lookup={v:i for i,v in enumerate(data['object_ids'])}
        if len(lookup)!=len(identities) or set(lookup)!=set(identities):
            raise ValueError('Different or duplicated validation populations')
        order=[lookup[v] for v in identities]
        data={k:v[order] for k,v in data.items()}; datasets[name]=data
        for key in ['rows','blocks']: np.testing.assert_array_equal(data[key],reference[key])
        for method in ['jwst','native','bilinear','bicubic']:
            good=data[method+'_good'] & reference[method+'_good']
            np.testing.assert_allclose(data[method+'_e'][good],reference[method+'_e'][good],rtol=2e-5,atol=2e-6)
        good=data['jwst_good'] & reference['jwst_good']
        deviations[name]=float(np.max(np.abs(data['jwst_e'][good]-reference['jwst_e'][good]))) if good.any() else None
        for method in ['jwst',*METHODS]: common &= data[method+'_good']
    errors={name:((data['sr_e'][common]-data['jwst_e'][common])**2).mean(1) for name,data in datasets.items()}
    for method in ['native','bilinear','bicubic']:
        errors[method]=((reference[method+'_e'][common]-reference['jwst_e'][common])**2).mean(1)
    comparison={name:dict(component_rmse=float(np.sqrt(error.mean())) if len(error) else None) for name,error in errors.items()}
    if common.any():
        labels,inverse=np.unique(reference['blocks'][common],return_inverse=True)
        draws=np.random.default_rng(20261016).integers(len(labels),size=(1000,len(labels)))
        baseline=errors['legacy_best49']
        denominator=np.bincount(inverse,weights=baseline)[draws].sum(1)
        valid=denominator>0
        for name,error in errors.items():
            gain=100*(1-np.sqrt(error.mean()/baseline.mean())) if baseline.mean()>0 else None
            interval=None
            if len(labels)>1 and valid.any():
                sampled=np.bincount(inverse,weights=error)[draws].sum(1)
                interval=np.percentile(100*(1-np.sqrt(sampled[valid]/denominator[valid])),[2.5,97.5]).tolist()
            comparison[name].update(improvement_vs_legacy_percent=gain,ci95_percent=interval)
    report=dict(objects=len(identities),common_successes=int(common.sum()),common_object_ids=reference['object_ids'][common].tolist(),
        methods=comparison,full_population_sr_failures={n:int((~d['sr_good']).sum()) for n,d in datasets.items()},
        target_roundtrip_max_abs_difference=deviations,
        source_hash=sha256(__file__),inputs={n:dict(path=str(p),sha256=sha256(p)) for n,p in paths.items()},
        limitation='Validation only, conditional spatial-block intervals; single training seed, different representation/loss/capacity/inference cost. Mean image uses three predeclared samples in linear intensity.')
    args.output.mkdir(parents=True,exist_ok=False)
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name=args.output.name,
                    mode='online',job_type='full-diffusion-comparison',dir=str(args.output)) as run:
        if run.settings.mode!='online' or not run.url: raise RuntimeError('Online W&B required')
        report['wandb_url']=run.url
        table=wandb.Table(columns=['method','component_rmse','improvement_vs_legacy_percent','ci95_low','ci95_high'])
        for name,values in comparison.items():
            interval=values.get('ci95_percent') or [None,None]
            table.add_data(name,values['component_rmse'],values.get('improvement_vs_legacy_percent'),*interval)
        run.log({'comparison/matched_shapes':table}); run.summary.update(report)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
