"""Matched observed ellipticity summaries for all cached Figure 6 methods."""
import numpy as np


def summarize(measurements, blocks, ids, methods, reasons, selection=None,
              repeats=2000, seed=20261017):
    n=len(ids)
    selected=np.ones(n,bool) if selection is None else np.asarray(selection,bool)
    common=selected & np.logical_and.reduce([measurements[m]['good'] for m in ['target',*methods]])
    result=dict(objects=int(selected.sum()), common_successes=int(common.sum()),
        object_ids=np.asarray(ids)[common].tolist(),
        success_counts={m:int((selected & v['good']).sum()) for m,v in measurements.items()},
        failure_reasons={m:{r:int((selected & v['flags'][:,j]).sum()) for j,r in enumerate(reasons)}
                         for m,v in measurements.items()}, methods={}, paired={})
    if not common.any():return result
    labels,inverse=np.unique(np.asarray(blocks)[common],return_inverse=True)
    result['spatial_blocks']=len(labels)
    if len(labels)<2:raise ValueError('Need at least two spatial blocks')
    counts=np.bincount(inverse)
    draws=np.random.default_rng(seed).integers(len(labels),size=(repeats,len(labels)))
    denom=counts[draws].sum(1)
    def mean_ci(values,root=False):
        sums=np.stack([values[inverse==j].sum(0) for j in range(len(labels))])
        boot=sums[draws].sum(1)/denom.reshape((-1,)+(1,)*(values.ndim-1))
        avg=values.mean(0)
        if root:avg=np.sqrt(avg);boot=np.sqrt(boot)
        return dict(mean=np.asarray(avg).tolist(),ci95=np.percentile(boot,[2.5,97.5],axis=0).T.tolist()),boot
    ref=measurements['target']['e'][common]
    bootstrap_rmse={}
    for m in [*methods,'round_control']:
        pred=measurements[m]['e'][common] if m!='round_control' else np.zeros_like(ref)
        err=pred-ref
        record={}
        record['component_rmse'],bootstrap_rmse[m]=mean_ci((err**2).mean(1),root=True)
        record['bias'],_=mean_ci(err)
        record['component_scatter']=err.std(0).tolist()
        record['vector_error_above_0_2_fraction'],_=mean_ci((np.linalg.norm(err,axis=1)>.2).astype(float))
        record['component_ols_slope']=[float(np.sum((ref[:,j]-ref[:,j].mean())*(pred[:,j]-pred[:,j].mean()))/np.sum((ref[:,j]-ref[:,j].mean())**2)) if np.var(ref[:,j])>0 else None for j in range(2)]
        if m!='round_control' and 'sigma' in measurements[m]:
            record['log_size_ratio'],_=mean_ci(np.log(measurements[m]['sigma'][common]/measurements['target']['sigma'][common]))
        result['methods'][m]=record
    for m in methods:
        result['paired'][m]={}
        for baseline in ['bilinear','legacy','direct5']:
            if baseline==m or baseline not in methods:continue
            diff=bootstrap_rmse[m]-bootstrap_rmse[baseline]
            result['paired'][m][baseline]=dict(
                rmse_difference=result['methods'][m]['component_rmse']['mean']-result['methods'][baseline]['component_rmse']['mean'],
                ci95=np.percentile(diff,[2.5,97.5]).tolist())
    return result
