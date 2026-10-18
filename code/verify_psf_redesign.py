"""Read-only independent provenance, optimizer and paired-shape verification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    out=ROOT/'runs/psf-redesign-comparison-v01'
    report=read(out/'comparison.json');status=read(out/'status.json')
    assert report['complete'] and status['complete']
    assert sha(out/'comparison.json')==status['comparison_sha256']
    for name,digest in report['input_hashes'].items(): assert sha(name)==digest,name
    for name,digest in report['source_hashes'].items(): assert sha(out/name)==digest,name
    for name,digest in status['artifacts'].items(): assert sha(out/name)==digest,name
    parent_path=ROOT/'runs/residual-full50-s20261016/best.pt'
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    verified=[]
    for branch in ['conditioned','deconv']:
        run=ROOT/f'runs/psf-redesign-{branch}-v01'
        contract=read(run/'contract.json')
        assert sha(parent_path)==contract['parent_sha256']
        for name,digest in contract['source_hashes'].items(): assert sha(run/'source'/name)==digest,name
        assert sha(run/'protocol.md')==contract['protocol_sha256']
        assert sha(ROOT/'runs/psf-features-v01/features.npz')==contract['features_sha256']
        assert sha(ROOT/'runs/psf-features-v01/report.json')==contract['features_report_sha256']
        assert read(run/'selection.json')==read(ROOT/'runs/resnet-refine-control-v03/selection.json')
        zero=torch.load(run/'checkpoint-00.pt',map_location='cpu',weights_only=False)
        for key,value in parent['model'].items():torch.testing.assert_close(zero['model'][key],value,rtol=0,atol=0)
        for epoch in range(1,11):
            a=read(run/f'epoch-{epoch:02d}.json');b=read(ROOT/f'runs/resnet-refine-control-v03/epoch-{epoch:02d}.json')
            assert a['updates']==epoch*2227 and a['training_stream_sha256']==b['training_stream_sha256']
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==10 and last['step']==22270
        groups=last['optimizer']['param_groups']
        np.testing.assert_allclose([g['lr'] for g in groups],[1e-6,1e-5],rtol=1e-10)
        for group,expected in zip(groups,[parent['step']+22270,22270]):
            for param in group['params']:
                assert float(last['optimizer']['state'][param]['step'])==expected
        verified.append(branch)
        del zero,last
    # Recompute the population RMSE difference and paired block intervals from
    # saved object shapes, independently of the comparison helper functions.
    benchmark=read(ROOT/'paper_figures/inputs/model_benchmark_v03.json')
    blocks=np.asarray(benchmark['blocks'])
    checks=0
    with np.load(out/'shapes.npz') as arrays:
        np.testing.assert_array_equal(arrays['object_ids'],benchmark['object_ids'])
        for kind in ['fixed','adaptive']:
            keep=np.logical_and.reduce([arrays[f'{kind}__{name}__good'] for name in ['target',*report['methods']]])
            assert keep.sum()==report['shapes'][kind]['common_successes']
            labels,inv=np.unique(blocks[keep],return_inverse=True)
            counts=np.bincount(inv)
            draws=np.random.default_rng(20261017).integers(0,len(labels),(2000,len(labels)))
            for pair,expected in report['shapes'][kind]['paired_all'].items():
                a,b=pair.split('__minus__')
                errors=[]
                for name in [a,b]:
                    delta=arrays[f'{kind}__{name}__e'][keep]-arrays[f'{kind}__target__e'][keep]
                    errors.append((delta[:,0]**2+delta[:,1]**2)/2)
                boot=[]
                for error in errors:
                    sums=np.bincount(inv,weights=error,minlength=len(labels))
                    boot.append(np.sqrt(sums[draws].sum(1)/counts[draws].sum(1)))
                value=np.sqrt(errors[0].mean())-np.sqrt(errors[1].mean())
                interval=np.percentile(boot[0]-boot[1],[2.5,97.5])
                np.testing.assert_allclose(value,expected['mean'],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(interval,expected['ci95'],atol=1e-12,rtol=1e-12)
                checks+=1
    result=dict(complete=True,verified_branches=verified,exact_parent_weights_at_epoch_zero=True,
        inherited_and_new_optimizer_steps_correct=True,matching_control_streams_all_epochs=True,
        source_input_and_plot_hashes_match=True,independent_paired_shape_checks=checks,
        source_sha256=sha(__file__),comparison_sha256=sha(out/'comparison.json'))
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
