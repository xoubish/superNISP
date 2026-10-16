"""NISP-only generation diagnostics in linear physical intensity."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from diffusion_corrected import fixed_noise
from shape_diagnostics import METHODS,diagnose_moments,interpolation,paired_summary


class GalleryPrediction(nn.Module):
    """Render already generated NISP-only samples using the shared gallery code."""
    def __init__(self,prediction):
        super().__init__()
        self.prediction=prediction
    def forward(self,nisp):
        if len(nisp)!=len(self.prediction): raise ValueError('Gallery population mismatch')
        return self.prediction.to(nisp.device)


@torch.no_grad()
def evaluate_generation(model,dataset,indices,device,steps,seed,batch_size,output,label):
    model.eval()
    catalog={int(r['row']):r['spatial_block'] for r in csv.DictReader((dataset.directory/'objects.csv').open())}
    collected={m:[] for m in ['jwst',*METHODS]}
    rows,ids=[],[]
    predictions={}
    mae=0.
    for batch in DataLoader(Subset(dataset,indices),batch_size=batch_size,num_workers=0):
        x,valid=batch['nisp'].to(device),batch['nisp_valid'].to(device)
        # No JWST tensor/mask enters sample(); target is read only afterwards.
        generated=model.sample(x,valid,fixed_noise(batch['row'].tolist(),seed,device),steps)
        prediction=dataset.jwst_to_mjy_sr(generated)
        if not torch.isfinite(prediction).all(): raise FloatingPointError('Nonfinite physical prediction')
        target=dataset.jwst_to_mjy_sr(batch['jwst'].to(device))
        target_valid=batch['jwst_valid'].to(device)
        linear=3*torch.sinh(x) if dataset.representation=='asinh' else x
        native=linear*dataset.norm['nisp_scale']
        images={'jwst':(target,target_valid,1.),'sr':(prediction,target_valid,1.),'native':(native,valid,5.)}
        for method in ['bilinear','bicubic']:
            image,mask=interpolation(native,valid,method)
            images[method]=(image,mask,1.)
        for method,(image,mask,step) in images.items():
            values=diagnose_moments(image,mask,step)
            collected[method].append({k:values[k].cpu().numpy() for k in ['e','good','flags']})
        error=torch.where(target_valid,(prediction-target).abs(),0.)
        mae+=(error.sum((1,2,3))/target_valid.sum((1,2,3))).sum().item()
        for row,image in zip(batch['row'].tolist(),generated.cpu()): predictions[row]=image
        rows.extend(batch['row'].tolist()); ids.extend(batch['object_id'].tolist())
    measures={m:{k:np.concatenate([v[k] for v in parts]) for k in ['e','good','flags']}
              for m,parts in collected.items()}
    blocks=np.array([catalog[r] for r in rows])
    summary=paired_summary(measures,blocks,bootstrap=200)
    metrics=dict(mae_mjy_sr=mae/len(rows),objects=len(rows),common_successes=summary['common_successes'],
                 sr_failures=int((~measures['sr']['good']).sum()),reference_failures=int((~measures['jwst']['good']).sum()))
    for method,result in summary['methods'].items(): metrics[f'{method}_shape_rmse']=result['component_rmse']
    arrays=dict(object_ids=np.array(ids),rows=np.array(rows),blocks=blocks,
        **{f'{m}_{k}':v for m,values in measures.items() for k,v in values.items()})
    np.savez_compressed(output/f'{label}-measurements.npz',**arrays)
    (output/f'{label}-summary.json').write_text(json.dumps(dict(metrics=metrics,shapes=summary,seed=seed,
        sampling_steps=steps,scope='Small debugging sample; conditional shape diagnostics, not a full validation benchmark.'),indent=2,allow_nan=False))
    return metrics,arrays,predictions


def repeated_shape_report(arrays):
    reference=arrays[0]
    for current in arrays[1:]:
        for key in ['object_ids','rows','blocks','jwst_e','jwst_good','jwst_flags']:
            np.testing.assert_array_equal(reference[key],current[key])
    common=reference['jwst_good'].copy()
    for current in arrays:
        for method in METHODS: common &= current[f'{method}_good']
    shapes=np.stack([a['sr_e'] for a in arrays])
    result=dict(objects=len(common),samples=len(arrays),common_successes=int(common.sum()),
        common_object_ids=reference['object_ids'][common].tolist(),
        sr_failures_per_sample=[int((~a['sr_good']).sum()) for a in arrays],
        component_rmse_per_sample=[],shape_component_scatter_rms=None,
        interpretation='Variation across fixed initial-noise seeds; not calibrated uncertainty or a coverage guarantee.')
    if common.any():
        errors=shapes[:,common]-reference['jwst_e'][None,common]
        result['component_rmse_per_sample']=np.sqrt((errors**2).mean((1,2))).tolist()
        result['shape_component_scatter_rms']=float(np.sqrt(shapes[:,common].var(axis=0,ddof=1).mean()))
    return result


def plot_repeated_samples(predictions,gallery,dataset,path,seeds):
    import matplotlib.pyplot as plt
    meta=gallery['metadata']
    fig,axes=plt.subplots(len(meta['rows']),len(predictions)+2,figsize=(15,3*len(meta['rows'])),squeeze=False)
    for i,row in enumerate(meta['rows']):
        images=[gallery['native'][i,0],*[dataset.jwst_to_mjy_sr(p[row])[0] for p in predictions],gallery['target'][i,0]]
        masks=[gallery['native_valid'][i,0]]+[gallery['batch']['jwst_valid'][i,0]]*(len(predictions)+1)
        limit=meta['limits_mjy_sr'][i]
        titles=['NISP']+[f'SR seed {s}' for s in seeds]+['JWST']
        for j,(image,mask,title) in enumerate(zip(images,masks,titles)):
            values=np.ma.array(image[62:143,62:143].numpy(),mask=~mask[62:143,62:143].numpy())
            cmap=plt.get_cmap('gray').copy(); cmap.set_bad('#777777')
            axes[i,j].imshow(np.arcsinh(values/limit['stretch']),origin='lower',cmap=cmap,
                vmin=np.arcsinh(limit['low']/limit['stretch']),vmax=np.arcsinh(limit['high']/limit['stretch']))
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if i==0: axes[i,j].set_title(title)
            if j==0: axes[i,j].set_ylabel(f"ID {meta['object_ids'][i]}")
    fig.suptitle('Same NISP input, different initial-noise seeds | best checkpoint')
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(path,dpi=110); plt.close(fig)
