"""Corrected residual training. Launch inside tmux/screen; W&B online is required.

Checkpoints resume at epoch boundaries, including RNG, sampler, optimizer and
scheduler state. Test objects are never opened. No mixed precision or clipping
is used for the initial reference. Validation MAE alone selects the pilot model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import wandb

from paired_sample import PairedSample, masked_l1
from residual_baseline import EuclidToJWSTSuperResolution
from galaxy_visuals import prepare_gallery, log_gallery

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persistent_session_required():
    if not (os.environ.get('TMUX') or os.environ.get('STY')):
        raise RuntimeError('Launch inside tmux or screen before running models.')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def paired_augment(batch):
    """A common D4 transform for all spatial tensors, including masks and RMS."""
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            value = torch.rot90(value, k, (-2, -1))
            batch[key] = value.flip(-1) if flip else value
    return batch


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def save_checkpoint(path, model, optimizer, scheduler, generator, **metadata):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), rng=rng_state(generator), **metadata)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def select_indices(dataset, count, seed, clean=False):
    order = np.random.default_rng(seed).permutation(len(dataset))
    if not clean:
        return order[:count].tolist() if count else list(range(len(dataset)))
    chosen = []
    for i in order:
        item = dataset[int(i)]
        if item['nisp_valid'].all() and item['jwst_valid'].all():
            chosen.append(int(i))
            if len(chosen) == count:
                return chosen
    raise ValueError('Not enough fully valid training pairs')


def make_loader(dataset, indices, batch_size, workers, generator, shuffle):
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                   shuffle=shuffle, generator=generator)
    if workers:
        options.update(multiprocessing_context='spawn')
    return DataLoader(Subset(dataset, indices), **options)


def shape_moments(images, valid):
    """Fixed Gaussian-weighted central moments, signed linear intensities.

    Diagnostic only, not a calibrated shear estimator. Coordinates in target
    pixels; Gaussian sigma=10, radius=25 (about 0.59/1.48 arcsec). No clipping of
    negative intensity. Report failures and use paired successes for errors.
    """
    side = images.shape[-1]
    axis = torch.arange(side, device=images.device, dtype=images.dtype) - (side-1)/2
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    aperture = xx.square() + yy.square() <= 25**2
    weight = torch.exp(-(xx.square()+yy.square())/(2*10**2)) * aperture
    flux = images[:, 0] * valid[:, 0] * weight
    total = flux.sum((-2, -1))
    safe = torch.where(total.abs() > 1e-12, total, torch.ones_like(total))
    cx = (flux * xx).sum((-2, -1)) / safe
    cy = (flux * yy).sum((-2, -1)) / safe
    dx, dy = xx - cx[:, None, None], yy - cy[:, None, None]
    qxx = (flux * dx.square()).sum((-2, -1)) / safe
    qyy = (flux * dy.square()).sum((-2, -1)) / safe
    qxy = (flux * dx * dy).sum((-2, -1)) / safe
    trace = qxx + qyy
    denom = torch.where(trace.abs() > 1e-12, trace, torch.ones_like(trace))
    e = torch.stack(((qxx-qyy)/denom, 2*qxy/denom), -1)
    good = ((total > 1e-12) & (qxx > 0) & (qyy > 0) &
            (qxx*qyy > qxy.square()) & torch.isfinite(e).all(-1) &
            (valid[:, 0] | ~aperture).all((-2, -1)))
    return e, good


@torch.no_grad()
def evaluate(model, loader, dataset, device):
    model.eval()
    total_mae = total_mse = shape_error = 0.
    n = successes = pred_failures = target_failures = 0
    for batch in loader:
        x, y, mask = (batch[k].to(device) for k in ('nisp', 'jwst', 'jwst_valid'))
        prediction = model(x)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation output')
        b = len(x)
        total_mae += masked_l1(prediction, y, mask).item() * b
        error = dataset.jwst_to_mjy_sr(prediction) - dataset.jwst_to_mjy_sr(y)
        total_mse += ((error.square()*mask).sum((1,2,3))/mask.sum((1,2,3))).sum().item()
        pred_e, pred_ok = shape_moments(dataset.jwst_to_mjy_sr(prediction), mask)
        target_e, target_ok = shape_moments(dataset.jwst_to_mjy_sr(y), mask)
        good = pred_ok & target_ok
        shape_error += (pred_e[good]-target_e[good]).square().sum().item()
        successes += good.sum().item()
        pred_failures += (~pred_ok).sum().item()
        target_failures += (~target_ok).sum().item()
        n += b
    return dict(mae=total_mae/n, rmse_mjy_sr=(total_mse/n)**.5,
                shape_e_rmse=(shape_error/(2*successes))**.5 if successes else None,
                shape_paired_successes=successes, shape_prediction_failures=pred_failures,
                shape_target_failures=target_failures, objects=n)


def train_epoch(model, loader, optimizer, scheduler, device, augment, run, step):
    model.train()
    loss_sum = 0.
    n = 0
    for batch in loader:
        if augment:
            batch = paired_augment(batch)
        x, y, mask = (batch[k].to(device) for k in ('nisp', 'jwst', 'jwst_valid'))
        optimizer.zero_grad(set_to_none=True)
        loss = masked_l1(model(x), y, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()  # Exactly once per successful optimizer update.
        step += 1
        loss_sum += loss.item()*len(x)
        n += len(x)
        if step == 1 or step % 10 == 0:
            run.log({'updates': step, 'train/batch_mae': loss.item(),
                     'train/gradient_norm': norm.item(), 'train/lr': lr})
    return loss_sum/n, step


def main(args):
    persistent_session_required()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    data = args.data.resolve()
    output = args.output.resolve()
    config = vars(args).copy()
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in config.items()
              if k not in ('resume', 'stop_after_epochs')}
    hashes = {name: sha256(data/name) for name in ('manifest.json','normalization.json','SHA256SUMS')}
    sources = {name: sha256(Path(__file__).parent/name) for name in
               ('train_paired.py','residual_baseline.py','paired_sample.py','galaxy_visuals.py')}
    signature = dict(config=config, data_hashes=hashes, source_hashes=sources)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['signature'] != signature:
            raise ValueError('Resume config, code, or data differs from checkpoint')
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (output/'source').mkdir()
        for name in sources:
            shutil.copy2(Path(__file__).parent/name, output/'source'/name)
        for name in hashes:
            shutil.copy2(data/name, output/name)
        provenance = dict(**signature, revision=subprocess.check_output(
            ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'], cwd=ROOT,text=True),
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            h5py=h5py.__version__, wandb=wandb.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device))
        (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
        (output/'working-tree.diff').write_text(subprocess.check_output(
            ['git','diff','--no-ext-diff','--','code','HANDOFF_GPU.md','.gitignore'],cwd=ROOT,text=True))
    run_id_file = output/'wandb_id.txt'
    if args.resume:
        run_id = checkpoint['wandb_id']
    else:
        run_id = wandb.util.generate_id()
        run_id_file.write_text(run_id+'\n')
    # Explicit mode prevents WANDB_MODE=offline/disabled from silently losing monitoring.
    with wandb.init(project=args.project, entity=args.entity, id=run_id,
                    resume='must' if args.resume else 'never', name=output.name,
                    config={**config, 'data_hashes':hashes, 'source_hashes':sources},
                    mode='online', dir=str(output), settings=wandb.Settings(init_timeout=60)) as run:
        if run.settings.mode != 'online' or not run.url:
            raise RuntimeError('Online W&B monitoring required before running models')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        run.define_metric('updates')
        run.define_metric('train/*', step_metric='updates')
        run.define_metric('epoch')
        run.define_metric('validation/*', step_metric='epoch')
        run.define_metric('galaxies/*', step_metric='epoch')
        train = PairedSample(data, 'train', args.representation)
        validation = None if args.mode == 'overfit' else PairedSample(data,'validation',args.representation)
        if checkpoint:
            train_indices = checkpoint['train_indices']
            validation_indices = checkpoint['validation_indices']
        else:
            train_indices = select_indices(train,args.train_limit,args.seed,clean=args.mode=='overfit')
            validation_indices = select_indices(validation,args.validation_limit,args.seed+1) if validation else []
        (output/'selection.json').write_text(json.dumps(dict(
            train_rows=train.indices[train_indices].tolist(),
            validation_rows=validation.indices[validation_indices].tolist() if validation else []),indent=2))
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = make_loader(train,train_indices,args.batch_size,args.workers,generator,True)
        eval_dataset = validation if validation else train
        eval_indices = validation_indices if validation else train_indices
        eval_loader = make_loader(eval_dataset,eval_indices,args.batch_size,args.workers,
                                  torch.Generator().manual_seed(args.seed+1),False)
        gallery = prepare_gallery(eval_dataset,eval_indices,args.gallery_count,args.seed+2,output) if args.gallery_count else None
        model = EuclidToJWSTSuperResolution(head=args.head).to(device)
        run.summary['parameters'] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,
                     total_steps=args.epochs*len(train_loader),pct_start=.3)
        first_epoch = step = 0
        best = float('inf')
        elapsed_prior = 0.
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            restore_rng(checkpoint['rng'],generator)
            first_epoch, step, best = checkpoint['epoch'], checkpoint['step'], checkpoint['best']
            elapsed_prior = checkpoint['elapsed_seconds']
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        prefix = 'overfit_train' if args.mode == 'overfit' else 'validation'
        if not checkpoint:
            initial = evaluate(model,eval_loader,eval_dataset,device)
            run.log({f'{prefix}/{k}':v for k,v in initial.items()} | {'epoch':0,'updates':0})
            (output/'initial_metrics.json').write_text(json.dumps(initial,indent=2))
            if gallery:
                log_gallery(model,eval_dataset,gallery,device,run,output,0,0,shape_moments)
        end_epoch = min(args.epochs, first_epoch+args.stop_after_epochs) if args.stop_after_epochs else args.epochs
        for epoch in range(first_epoch,end_epoch):
            train_mae,step = train_epoch(model,train_loader,optimizer,scheduler,device,
                                         args.mode!='overfit',run,step)
            metrics = evaluate(model,eval_loader,eval_dataset,device)
            elapsed = elapsed_prior+time.monotonic()-started
            record = {f'{prefix}/{k}':v for k,v in metrics.items()} | dict(
                epoch=epoch+1, updates=step, train_epoch_mae=train_mae,
                elapsed_seconds=elapsed, peak_cuda_gib=torch.cuda.max_memory_allocated(device)/2**30)
            run.log(record)
            print(json.dumps(record),flush=True)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            improved = metrics['mae'] < best
            best = min(best,metrics['mae'])
            save_checkpoint(output/'last.pt',model,optimizer,scheduler,generator,
                            signature=signature,epoch=epoch+1,step=step,best=best,
                            wandb_id=run.id,train_indices=train_indices,
                            validation_indices=validation_indices,elapsed_seconds=elapsed)
            if improved:
                shutil.copy2(output/'last.pt',output/'best.pt')
                run.summary[f'best_{prefix}_mae'] = best
                run.summary['best_epoch'] = epoch+1
            # Save training state before rendering: an image failure cannot lose the epoch.
            if gallery and ((epoch+1)%args.image_every==0 or epoch+1==args.epochs):
                log_gallery(model,eval_dataset,gallery,device,run,output,epoch+1,step,shape_moments)
            (output/'progress.json').write_text(json.dumps(dict(
                epoch=epoch+1, planned_epochs=args.epochs, updates=step, best=best,
                elapsed_seconds=elapsed, wandb_url=run.url),indent=2))
        run.summary['completed_epochs'] = end_epoch
        run.summary['planned_epochs'] = args.epochs
        run.summary['checkpoint_sha256'] = sha256(output/'last.pt')
        (output/'status.json').write_text(json.dumps(dict(
            complete=end_epoch==args.epochs, epoch=end_epoch, updates=step, best=best,
            checkpoint_sha256=sha256(output/'last.pt'), wandb_url=run.url),indent=2))
        train.close()
        if validation:
            validation.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',default='superNISP-corrected')
    p.add_argument('--entity',default=None)
    p.add_argument('--mode',choices=['overfit','pilot','train'],default='pilot')
    p.add_argument('--head',choices=['legacy','direct5'],default='legacy')
    p.add_argument('--representation',choices=['linear','asinh'],default='linear')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--train-limit',type=int,default=0,help='0 means full saved train split')
    p.add_argument('--validation-limit',type=int,default=0,help='0 means full validation split')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--gallery-count',type=int,default=8,help='Fixed evaluation objects shown in W&B; 0 disables')
    p.add_argument('--image-every',type=int,default=1,help='Log galaxy comparisons every N epochs')
    p.add_argument('--stop-after-epochs',type=int,default=0,help='Controlled interruption for resume check')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.lr,args.image_every) <= 0 or min(args.workers,args.train_limit,args.validation_limit,args.stop_after_epochs,args.gallery_count)<0:
        p.error('Counts and learning rate must be valid positive/nonnegative values')
    if args.mode=='overfit' and not args.train_limit:
        p.error('Overfit requires an explicit small --train-limit, typically 32')
    return args


if __name__=='__main__':
    main(parse_args())
