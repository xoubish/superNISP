"""Fixed scientific image comparisons for W&B; display stretches never affect loss."""
import json
import random
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import wandb


@contextmanager
def preserve_rng():
    """Media encoding must not alter the training/resume random stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def prepare_gallery(dataset, indices, count, seed, output):
    """Choose input-brightness strata in a seeded validation pool, without model outputs."""
    pool = np.random.default_rng(seed).permutation(indices)[:256]
    scored = []
    for i in pool:
        item = dataset[int(i)]
        center = item['nisp'][0,15:26,15:26]
        valid = item['nisp_valid'][0,15:26,15:26]
        score = float(torch.quantile(center[valid], .9)) if valid.any() else float('-inf')
        scored.append((score,int(i)))
    scored.sort()
    positions = np.linspace(0,len(scored)-1,min(count,len(scored))).round().astype(int)
    selected = [scored[k][1] for k in positions]
    items = [dataset[i] for i in selected]
    batch = {k:torch.stack([item[k] for item in items]) for k in
             ('nisp','jwst','nisp_valid','jwst_valid')}
    x = batch['nisp']
    linear = 3*torch.sinh(x) if dataset.representation=='asinh' else x
    native = linear*dataset.norm['nisp_scale']
    target = dataset.jwst_to_mjy_sr(batch['jwst'])
    support = F.interpolate(batch['nisp_valid'].float(),size=(205,205),mode='bilinear',align_corners=False)
    interpolated = F.interpolate(native,size=(205,205),mode='bilinear',align_corners=False)
    limits = []
    for image,mask in zip(target[:,0],batch['jwst_valid'][:,0]):
        values = image[62:143,62:143][mask[62:143,62:143]].numpy()
        low,high = np.percentile(values,[.5,99.5])
        high = max(float(high),float(low)+1e-6)
        low = min(float(low),0.)
        stretch = max(dataset.norm['jwst_scale'],(high-low)*.03)
        limits.append(dict(low=low,high=high,stretch=stretch,residual=max(stretch,(high-low)*.3)))
    metadata = dict(selection='Evenly spaced input central 90th-percentile brightness ranks in a seeded pool of up to 256 evaluation objects; display selection only.',
        seed=seed, indices=selected, rows=[item['row'] for item in items],
        object_ids=[item['object_id'] for item in items], limits_mjy_sr=limits,
        columns=['NISP Y (native sampling)','NISP Y bilinear','SR F115W prediction','JWST F115W','SR minus JWST'],
        stretch='asinh with shared fixed per-object intensity limits; residual symmetric fixed limits',
        invalid_pixels='gray; bilinear pixels require fully valid input support')
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'gallery_selection.json').write_text(json.dumps(metadata,indent=2))
    return dict(batch=batch, native=F.interpolate(native,size=(205,205),mode='nearest'),
                native_valid=F.interpolate(batch['nisp_valid'].float(),size=(205,205),mode='nearest')>.5,
                interpolated=interpolated,interpolated_valid=support>.99999,
                target=target,metadata=metadata)


@preserve_rng()
@torch.no_grad()
def log_gallery(model,dataset,gallery,device,run,output,epoch,step,shape_fn):
    model.eval()
    prediction = dataset.jwst_to_mjy_sr(model(gallery['batch']['nisp'].to(device))).cpu()
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite gallery prediction')
    target_mask = gallery['batch']['jwst_valid']
    _,pred_ok = shape_fn(prediction,target_mask)
    _,target_ok = shape_fn(gallery['target'],target_mask)
    arrays = [gallery['native'],gallery['interpolated'],prediction,gallery['target'],prediction-gallery['target']]
    masks = [gallery['native_valid'],gallery['interpolated_valid'],target_mask,target_mask,target_mask]
    meta = gallery['metadata']
    gray = plt.get_cmap('gray').copy()
    gray.set_bad('#777777')
    diverging = plt.get_cmap('RdBu_r').copy()
    diverging.set_bad('#777777')
    payload = {'epoch':epoch,'updates':step}
    directory = Path(output)/'galaxies'
    directory.mkdir(exist_ok=True)
    for view,crop in [('full',slice(None)),('central',slice(62,143))]:
        for start in range(0,len(prediction),4):
            stop = min(start+4,len(prediction))
            fig,axes = plt.subplots(stop-start,5,figsize=(15,3*(stop-start)),squeeze=False)
            for row,index in enumerate(range(start,stop)):
                limit = meta['limits_mjy_sr'][index]
                for col,(array,mask) in enumerate(zip(arrays,masks)):
                    ax = axes[row,col]
                    values = np.ma.array(array[index,0,crop,crop].numpy(),mask=~mask[index,0,crop,crop].numpy())
                    if col==4:
                        low,high = -limit['residual'],limit['residual']
                    else:
                        low,high = limit['low'],limit['high']
                    ax.imshow(np.arcsinh(values/limit['stretch']),origin='lower',interpolation='nearest',
                              vmin=np.arcsinh(low/limit['stretch']),vmax=np.arcsinh(high/limit['stretch']),
                              cmap=diverging if col==4 else gray)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row==0:
                        ax.set_title(meta['columns'][col],fontsize=10)
                    if col==0:
                        ax.set_ylabel(f"ID {meta['object_ids'][index]}\nrow {meta['rows'][index]}",fontsize=9)
                    if col==2:
                        ax.set_xlabel('Moment check: '+('pass' if pred_ok[index] else 'FAIL'),fontsize=9)
                    elif col==3:
                        ax.set_xlabel('Moment check: '+('pass' if target_ok[index] else 'FAIL'),fontsize=9)
                    elif col==4:
                        ax.set_xlabel(f"Residual scale: ±{high:.3g} MJy/sr",fontsize=8)
                    elif col==1:
                        ax.set_xlabel(f"Intensity: [{low:.3g}, {high:.3g}] MJy/sr",fontsize=8)
            label = 'full 205 × 205 footprint' if view=='full' else 'central 81 × 81 target pixels'
            fig.suptitle(f'Fixed galaxy comparisons | epoch {epoch} | {label}',fontsize=14)
            fig.text(.5,.008,'Sky-subtracted MJy/sr; fixed shared asinh display per galaxy. Gray = invalid. Y and F115W are different bands.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.025,1,.96))
            path = directory/f'epoch-{epoch:03d}-{view}-{start//4+1}.png'
            fig.savefig(path,dpi=110)
            plt.close(fig)
            payload[f'galaxies/{view}_{start//4+1}'] = wandb.Image(str(path),caption=f'Epoch {epoch}; IDs {meta["object_ids"][start:stop]}; fixed intensity limits across epochs')
    run.log(payload)
    print(f'Galaxy comparisons logged at epoch {epoch}: {directory}',flush=True)
    return prediction
