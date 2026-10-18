"""Matched PSF/noise-aware ResNet extension; tmux and online W&B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from paired_sample import PairedSample, masked_l1
from psf_aware_resnet import PSFAwareResNet, PSFSample, initialize_parent, context_schedule
from residual_refinement import (SourceMaskedSample, balanced_l1,
    capture_rng, restore_single_gpu_rng, FrozenDiagnostics, restore_parent)
from train_paired import persistent_session_required, seed_everything, make_loader, paired_augment, shape_moments
from galaxy_visuals import prepare_gallery, log_gallery, preserve_rng
from probe_shared_gpu import gpu_snapshot, sha, PARENT_SHA

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['train_psf_redesign.py','psf_aware_resnet.py','prepare_psf_features.py','test_psf_aware_resnet.py','audit_shape_psfs.py','train_residual_refinement.py','residual_refinement.py','test_residual_refinement.py',
 'probe_shared_gpu.py','train_paired.py','paired_sample.py','residual_baseline.py','galaxy_visuals.py',
 'shape_diagnostics.py','spatial_spectra.py']


def atomic_json(path,record):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_state(path,model,optimizer,scheduler,generator,**metadata):
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    rng=capture_rng(generator),**metadata),tmp)
    tmp.replace(path)


def update(model,optimizer,scheduler,batch,branch):
    batch=paired_augment(batch)
    x,y,valid=[batch[k].cuda() for k in ['nisp','jwst','jwst_valid']]
    optimizer.zero_grad(set_to_none=True)
    prediction=model(x)
    loss=masked_l1(prediction,y,valid)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return float(loss.detach()),float(norm)


def preflight(model,optimizer,scheduler,generator,train,branch,output):
    # Use a disposable path and restore the original parent state afterwards.
    batch=next(iter(DataLoader(torch.utils.data.Subset(train,list(range(8))),batch_size=8)))
    def clone_batch():return {k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    model.train()
    for _ in range(2):update(model,optimizer,scheduler,clone_batch(),branch)
    path=output/'preflight-resume.pt'
    save_state(path,model,optimizer,scheduler,generator)
    loss,norm=update(model,optimizer,scheduler,clone_batch(),branch)
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    expected_lr=optimizer.param_groups[0]['lr']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler']);restore_single_gpu_rng(saved['rng'],generator)
    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),saved['model'][k],rtol=0,atol=0)
    restored=optimizer.state_dict()
    assert restored['param_groups']==saved['optimizer']['param_groups']
    for key,state in restored['state'].items():
        for name,value in state.items():
            original=saved['optimizer']['state'][key][name]
            if torch.is_tensor(value):torch.testing.assert_close(value.cpu(),original.cpu(),rtol=0,atol=0)
            else:assert value==original
    assert scheduler.state_dict()==saved['scheduler']
    assert random.getstate()==saved['rng']['python']
    actual_rng=capture_rng(generator)
    for key in ['torch','loader']:torch.testing.assert_close(actual_rng[key],saved['rng'][key],rtol=0,atol=0)
    torch.testing.assert_close(actual_rng['cuda'][0],saved['rng']['cuda'][0],rtol=0,atol=0)
    np.testing.assert_equal(actual_rng['numpy'],saved['rng']['numpy'])
    again,_=update(model,optimizer,scheduler,clone_batch(),branch)
    # Historical bilinear CUDA backward can differ in reduction roundoff even
    # with deterministic cuDNN. Require exact restoration above and numerical
    # next-update equivalence, without changing the parent's kernels.
    max_difference=0.
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v.cpu(),expected[k],rtol=2e-6,atol=1e-8)
        max_difference=max(max_difference,float((v.cpu()-expected[k]).abs().max()))
    assert loss==again and optimizer.param_groups[0]['lr']==expected_lr
    atomic_json(output/'preflight.json',dict(complete=True,finite_loss_and_gradients=True,
        exact_checkpoint_restoration=True,numerical_next_update_after_resume=True,
        next_update_rtol=2e-6,next_update_atol=1e-8,max_weight_difference=max_difference,
        loss=loss,gradient_norm=norm))
    path.unlink()


@torch.inference_mode()
def evaluate(model,loader,dataset,diagnostics):
    model.eval();prediction=[];ids=[];full_mae=0.;count=0
    for batch in loader:
        x,y,mask=[batch[k].cuda() for k in ['nisp','jwst','jwst_valid']]
        result=model(x)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite validation prediction')
        full_mae+=float(masked_l1(result,y,mask))*len(x);count+=len(x)
        prediction.append(dataset.jwst_to_mjy_sr(result)[:,0,50:155,50:155].cpu().numpy())
        ids+=batch['object_id'].tolist()
    assert ids==diagnostics.ids.tolist()
    prediction=np.concatenate(prediction)
    metrics,arrays=diagnostics.measure(prediction)
    metrics['full_normalized_mae']=full_mae/count
    return metrics,arrays,prediction


def main(args):
    persistent_session_required()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu_uuid
    output=args.output.resolve()
    if not args.resume:output.mkdir(parents=True,exist_ok=False)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in ['resume','stop_after_epochs']}
    parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    assert sha(args.parent)==PARENT_SHA and parent['epoch']==49 and parent['step']==109123
    assert parent['signature']['config']['head']=='legacy'
    data_hashes={name:sha(args.data/name) for name in parent['signature']['data_hashes']}
    assert data_hashes==parent['signature']['data_hashes']
    for name,digest in parent['signature']['source_hashes'].items():assert sha(ROOT/'code'/name)==digest
    maskmeta=json.loads(args.masks.with_name('metadata.json').read_text())
    assert maskmeta['complete'] and maskmeta['data_hashes']==data_hashes
    assert sha(args.masks)==maskmeta['masks_sha256']
    signature=dict(config=config,data_hashes=data_hashes,source_hashes={n:sha(ROOT/'code'/n) for n in SOURCES},
                   parent_sha256=PARENT_SHA,masks_sha256=maskmeta['masks_sha256'],
                   features_sha256=sha(args.features),features_report_sha256=sha(args.features.parent/'report.json'),
                   protocol_sha256=sha(ROOT/'paper/psf_redesign_protocol.md'))
    assert signature['features_sha256']==json.loads((args.features.parent/'report.json').read_text())['features_sha256']
    resumed=torch.load(output/'last.pt',map_location='cpu',weights_only=False) if args.resume else None
    if resumed:assert resumed['signature']==signature
    else:
        (output/'source').mkdir()
        shutil.copy2(ROOT/'paper/psf_redesign_protocol.md',output/'protocol.md')
        for name in SOURCES:shutil.copy2(ROOT/'code'/name,output/'source'/name)
        atomic_json(output/'contract.json',dict(**signature,
            objective='Original masked L1; PSF, input RMS and validity conditioning, plus bounded inverse feature only in deconv branch',
            selection='Lowest mean source MAE on frozen validation central 105x105 common support; epoch 0 eligible',
            budget='10 epochs, 22270 extra updates, inherited epoch-49 ResNet weights and Adam state',
            schedule='Parent parameters: cosine 1e-5 to 1e-6, inherited Adam; new context parameters: cosine 1e-4 to 1e-5, fresh Adam with inherited betas/eps. Zero residual initialization preserves original predictions.',
            rng='Restore parent Python/NumPy/CPU/loader state; parent physical CUDA 0 state mapped to single visible device',
            limits='Validation only; one parent seed; four similar PSFs; header-based effective-PSF proxy; not intrinsic deconvolution or shear calibration. Compare to existing exact-stream control v03 and original parent.',
            bootstrap='Final paired comparison: 2000 spatial-block draws seed 20261017, equal object weights',
            gpu_memory='Bounded allocator on a free GPU, monitor every 200 updates; epoch-boundary full-state resume'))
    before=gpu_snapshot(args.gpu_uuid)
    atomic_json(output/'gpu-before.json',before)
    if before['free_mib']<1792:raise RuntimeError('Less than 1792 MiB free before CUDA initialization')
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(args.memory_mib*2**20/torch.cuda.get_device_properties(0).total_memory,0)
    seed_everything(20261016)
    run_id=resumed['wandb_id'] if resumed else wandb.util.generate_id()
    def memory_guard():
        snap=gpu_snapshot(args.gpu_uuid)
        with (output/'gpu-memory.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
        if snap['free_mib']<256:raise RuntimeError('Free device memory below 256 MiB; retaining completed checkpoints')
        return snap
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',id=run_id,
        resume='must' if resumed else 'never',name=output.name,job_type='psf-aware-resnet-redesign',
        mode='online',dir=str(output),config=signature,settings=wandb.Settings(init_timeout=60)) as run:
        if run.settings.mode!='online' or not run.url:raise RuntimeError('Online monitoring required')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        run.define_metric('updates');run.define_metric('train/*',step_metric='updates')
        run.define_metric('epoch');run.define_metric('validation/*',step_metric='epoch')
        run.define_metric('galaxies/*',step_metric='epoch')
        train=PSFSample(args.data,'train',args.features);validation=PSFSample(args.data,'validation',args.features)
        train_indices=parent['train_indices'];val_indices=parent['validation_indices']
        assert len(train_indices)==17816 and len(val_indices)==2118
        generator=torch.Generator()
        train_loader=make_loader(train,train_indices,8,2,generator,True)
        val_loader=make_loader(validation,val_indices,8,0,torch.Generator().manual_seed(20261017),False)
        assert len(train_loader)==2227
        atomic_json(output/'selection.json',dict(train_rows=train.indices[train_indices].tolist(),
            validation_rows=validation.indices[val_indices].tolist()))
        gallery_data=PairedSample(args.data,'validation','linear')
        gallery=prepare_gallery(gallery_data,val_indices,8,20261018,output)
        gallery['batch']['nisp']=torch.stack([validation[i]['nisp'] for i in gallery['metadata']['indices']])
        gallery_data.close()
        model=PSFAwareResNet(branch=args.branch).cuda()
        optimizer,parent=initialize_parent(model,parent)
        scheduler=context_schedule(optimizer,args.epochs*len(train_loader))
        restore_single_gpu_rng(parent['rng'],generator)
        if not resumed:
            preflight(model,optimizer,scheduler,generator,train,args.branch,output)
            restore_parent(model,optimizer,parent)
            scheduler=context_schedule(optimizer,args.epochs*len(train_loader))
            restore_single_gpu_rng(parent['rng'],generator)
            optimizer.zero_grad(set_to_none=True)
            assert all(float(state['step'])==parent['step'] for state in optimizer.state.values())
            print('Preflight: finite updates, exact state restoration and numerical next-update equivalence passed',flush=True)
        else:
            model.load_state_dict(resumed['model']);optimizer.load_state_dict(resumed['optimizer'])
            scheduler.load_state_dict(resumed['scheduler']);restore_single_gpu_rng(resumed['rng'],generator)
        frozen=FrozenDiagnostics()
        start_epoch=resumed['epoch'] if resumed else 0
        step=resumed['step'] if resumed else 0
        best=resumed['best'] if resumed else None
        best_epoch=resumed['best_epoch'] if resumed else 0
        elapsed_prior=resumed['elapsed_seconds'] if resumed else 0.
        started=time.monotonic()
        def persist(epoch,metrics,arrays,prediction,is_best,stream_hash=None):
            elapsed=elapsed_prior+time.monotonic()-started
            np.savez_compressed(output/f'measurements-{epoch:02d}.npz',object_ids=frozen.ids,**arrays)
            save_state(output/'last.pt',model,optimizer,scheduler,generator,signature=signature,
                epoch=epoch,step=step,best=best,best_epoch=best_epoch,wandb_id=run.id,
                elapsed_seconds=elapsed,train_indices=train_indices,validation_indices=val_indices)
            if is_best:
                shutil.copy2(output/'last.pt',output/'best.pt')
                np.save(output/'best_predictions.npy',prediction)
            np.save(output/f'predictions-{epoch:02d}.npy',prediction)
            shutil.copy2(output/'last.pt',output/f'checkpoint-{epoch:02d}.pt')
            record=dict(epoch=epoch,updates=step,elapsed_seconds=elapsed,best=best,best_epoch=best_epoch,
                validation=metrics,training_stream_sha256=stream_hash,learning_rate=optimizer.param_groups[0]['lr'],
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            atomic_json(output/f'epoch-{epoch:02d}.json',record)
            atomic_json(output/'progress.json',record|{'planned_epochs':args.epochs,'wandb_url':run.url,'branch':args.branch})
            run.log({'epoch':epoch,'updates':step,**{f'validation/{k}':v for k,v in metrics.items()}})
            run.summary.update(dict(best_source_mae=best,best_epoch=best_epoch,completed_epochs=epoch))
            print(json.dumps(record),flush=True)
        try:
            if not resumed:
                memory_guard()
                metrics,arrays,prediction=evaluate(model,val_loader,validation,frozen)
                frozen.verify_parent(prediction,metrics)
                atomic_json(output/'baseline-verification.json',dict(complete=True,parent_predictions_match_cache=True,
                    source_outside_metrics_match=True,metrics=metrics))
                best=metrics['source_mae']
                log_gallery(model,validation,gallery,torch.device('cuda:0'),run,output,0,0,shape_moments)
                restore_single_gpu_rng(parent['rng'],generator)
                persist(0,metrics,arrays,prediction,True)
                del prediction,arrays
            del parent,resumed
            end_epoch=min(args.epochs,start_epoch+args.stop_after_epochs) if args.stop_after_epochs else args.epochs
            for epoch in range(start_epoch+1,end_epoch+1):
                model.train();digest=hashlib.sha256();train_loss=0.;count=0
                for batch in train_loader:
                    digest.update(np.asarray(batch['row']).tobytes())
                    digest.update(repr(random.getstate()).encode())
                    loss,norm=update(model,optimizer,scheduler,batch,args.branch)
                    step+=1;train_loss+=loss*len(batch['row']);count+=len(batch['row'])
                    if step%50==0:run.log({'updates':step,'train/loss':loss,'train/gradient_norm':norm,
                                          'train/lr':optimizer.param_groups[0]['lr']})
                    if step%200==0:
                        snap=memory_guard()
                        atomic_json(output/'live.json',dict(epoch=epoch,updates=step,free_mib=snap['free_mib']))
                optimizer.zero_grad(set_to_none=True);memory_guard()
                metrics,arrays,prediction=evaluate(model,val_loader,validation,frozen)
                improved=metrics['source_mae']<best
                if improved:best=metrics['source_mae'];best_epoch=epoch
                stream=digest.hexdigest()
                control_epoch=json.loads((ROOT/'runs/resnet-refine-control-v03'/f'epoch-{epoch:02d}.json').read_text())
                assert stream==control_epoch['training_stream_sha256'],'Batch/augmentation stream differs from matched control'
                persist(epoch,metrics,arrays,prediction,improved,stream)
                run.log({'epoch':epoch,'updates':step,'train/epoch_loss':train_loss/count})
                log_gallery(model,validation,gallery,torch.device('cuda:0'),run,output,epoch,step,shape_moments)
                del prediction,arrays
            assert end_epoch<args.epochs or step==22270
            if end_epoch==args.epochs:np.testing.assert_allclose(optimizer.param_groups[0]['lr'],1e-6,rtol=1e-10)
            atomic_json(output/'status.json',dict(complete=end_epoch==args.epochs,epoch=end_epoch,updates=step,
                best=best,best_epoch=best_epoch,wandb_url=run.url,parent_unchanged=sha(args.parent)==PARENT_SHA,
                best_checkpoint_sha256=sha(output/'best.pt'),last_checkpoint_sha256=sha(output/'last.pt')))
        except Exception as error:
            atomic_json(output/'status.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
                last_completed_epoch=json.loads((output/'progress.json').read_text())['epoch'] if (output/'progress.json').exists() else None,
                wandb_url=run.url))
            raise
        finally:train.close();validation.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--branch',choices=['conditioned','deconv'],required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'runs/residual-full50-s20261016/best.pt')
    p.add_argument('--masks',type=Path,default=ROOT/'runs/residual-refinement-masks-v01/source_masks.h5')
    p.add_argument('--epochs',type=int,default=10,choices=[10])
    p.add_argument('--features',type=Path,default=ROOT/'runs/psf-features-v01/features.npz')
    p.add_argument('--memory-mib',type=int,default=4096)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epochs',type=int,default=0)
    main(p.parse_args())
