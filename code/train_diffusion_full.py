"""Full corrected diffusion run; tmux/screen and online W&B required.

Best checkpoints are selected by fixed-seed NISP-only generation MAE in MJy/sr.
Asinh is invertible with saved training-only scales; no output clamp or VGG loss.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import wandb

from paired_sample import PairedSample
from diffusion_full_runtime import CooperativeDiffusion as CorrectedDiffusion
from diffusion_diagnostics import GalleryPrediction,repeated_shape_report,plot_repeated_samples
from diffusion_full_diagnostics import evaluate_generation,evaluate_ensemble
from galaxy_visuals import prepare_gallery,log_gallery
from train_paired import (persistent_session_required,seed_everything,paired_augment,select_indices,
    save_checkpoint,restore_rng,sha256,shape_moments)

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['train_diffusion_full.py','diffusion_full_runtime.py','diffusion_full_diagnostics.py','diffusion_corrected.py','diffusion_diagnostics.py',
         'diffusion/model_sr3.py','paired_sample.py','train_paired.py','shape_diagnostics.py','galaxy_visuals.py']


def render(predictions,gallery,dataset,device,run,output,epoch,updates):
    prediction=torch.stack([predictions[row] for row in gallery['metadata']['rows']])
    log_gallery(GalleryPrediction(prediction),dataset,gallery,device,run,output,epoch,updates,shape_moments)


def main(args):
    persistent_session_required()
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    device=torch.device(args.device); torch.cuda.set_device(device)
    output=args.output.resolve(); data=args.data.resolve()
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in ['resume','stop_after_epochs']}
    signature=dict(config=config,source_hashes={n:sha256(ROOT/'code'/n) for n in SOURCES},
        data_hashes={n:sha256(data/n) for n in ['manifest.json','normalization.json','SHA256SUMS']})
    checkpoint=None
    if args.resume:
        checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['signature']!=signature: raise ValueError('Resume code/data/config mismatch')
    elif output.exists(): raise ValueError(f'Refusing to overwrite {output}')
    else:
        output.mkdir(parents=True)
        (output/'source').mkdir()
        for name in SOURCES:
            path=output/'source'/name; path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'code'/name,path)
        for name in signature['data_hashes']: shutil.copy2(data/name,output/name)
        (output/'provenance.json').write_text(json.dumps(dict(**signature,
            git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            torch=torch.__version__,numpy=np.__version__,wandb=wandb.__version__,gpu=torch.cuda.get_device_name(device),
            protocol=dict(prediction='v',terminal_alpha_bar=0,prior='pure standard Gaussian',sampling='DDIM eta=0; explicit current/next alpha; no clipping',
                representation='asinh(linear normalized intensity / 3), with saved training-only scales; inverse before physical metrics',
                loss='valid-pixel v MSE + 0.1 x0 L1 in model coordinates; input validity is a separate condition channel',
                selection='lowest NISP-only generated masked MAE in MJy/sr on fixed IDs/noise seeds',
                limitations='Full training split; checkpoint selection uses a frozen validation subset, with full validation after training. Same epochs/updates as residual but different representation, loss, parameter count and compute. Masks exclude unknown targets from loss but do not solve missing-data modeling. No test data.')),
            indent=2))
    run_id=checkpoint['wandb_id'] if checkpoint else wandb.util.generate_id()
    (output/'wandb_id.txt').write_text(run_id+'\n')
    with wandb.init(project='superNISP-corrected',entity='AI-Astro',name=output.name,id=run_id,
        resume='must' if checkpoint else 'never',mode='online',dir=str(output),config=signature,
        job_type='diffusion-'+args.mode,settings=wandb.Settings(init_timeout=60)) as run:
        if run.settings.mode!='online' or not run.url: raise RuntimeError('Online W&B required')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        run.define_metric('updates'); run.define_metric('train/*',step_metric='updates')
        run.define_metric('epoch'); run.define_metric('generated/*',step_metric='epoch')
        run.define_metric('galaxies/*',step_metric='epoch')
        seed_everything(args.seed)
        train=PairedSample(data,'train','asinh')
        evaluation=train if args.mode=='overfit' else PairedSample(data,'validation','asinh')
        ti=checkpoint['train_indices'] if checkpoint else select_indices(train,args.train_limit,args.seed,clean=args.mode=='overfit')
        vi=checkpoint['validation_indices'] if checkpoint else (ti if args.mode=='overfit' else select_indices(evaluation,args.validation_limit,args.seed+1))
        (output/'selection.json').write_text(json.dumps(dict(train_rows=train.indices[ti].tolist(),evaluation_rows=evaluation.indices[vi].tolist(),
            evaluation_split='train' if args.mode=='overfit' else 'validation'),indent=2))
        generator=torch.Generator().manual_seed(args.seed)
        loader=DataLoader(Subset(train,ti),batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
        gallery=prepare_gallery(evaluation,vi,8,args.seed+2,output)
        model=CorrectedDiffusion(args.width,args.timesteps).to(device)
        run.summary['parameters']=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,total_steps=args.epochs*len(loader),pct_start=.3)
        first=updates=0; best=float('inf'); elapsed_prior=0.
        if checkpoint:
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']); restore_rng(checkpoint['rng'],generator)
            first=checkpoint['epoch']; updates=checkpoint['step']; best=checkpoint['best']; elapsed_prior=checkpoint['elapsed_seconds']
        started=time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
        if not checkpoint:
            initial,_,predictions=evaluate_generation(model,evaluation,vi,device,args.sampling_steps,args.eval_seed,args.batch_size,output,'initial')
            (output/'initial_metrics.json').write_text(json.dumps(initial,indent=2))
            run.log({'epoch':0,'updates':0,**{'generated/'+k:v for k,v in initial.items()}})
            render(predictions,gallery,evaluation,device,run,output,0,0)
        end=min(args.epochs,first+args.stop_after_epochs) if args.stop_after_epochs else args.epochs
        for epoch in range(first+1,end+1):
            model.train(); total=0.; count=0
            for batch in loader:
                if args.mode!='overfit': batch=paired_augment(batch)
                x,mask,y,ymask=(batch[k].to(device) for k in ['nisp','nisp_valid','jwst','jwst_valid'])
                t=torch.randint(args.timesteps,(len(x),),device=device)
                noise=torch.randn_like(y)
                optimizer.zero_grad(set_to_none=True)
                loss,parts=model.training_loss(x,mask,y,ymask,t,noise)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                lr=optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step(); updates+=1
                total+=loss.item()*len(x); count+=len(x)
                if updates==1 or updates%10==0:
                    run.log(dict(updates=updates,**{'train/loss':loss.item(),'train/v_mse':parts['v_mse'].item(),
                        'train/x0_l1':parts['x0_l1'].item(),'train/gradient_norm':norm.item(),'train/lr':lr}))
            record=dict(epoch=epoch,updates=updates,train_loss=total/count)
            improved=False
            if epoch%args.evaluate_every==0 or epoch==args.epochs:
                metrics,_,predictions=evaluate_generation(model,evaluation,vi,device,args.sampling_steps,args.eval_seed,args.batch_size,output,f'epoch-{epoch:03d}')
                record.update({'generated/'+k:v for k,v in metrics.items()})
                improved=metrics['mae_mjy_sr']<best
                best=min(best,metrics['mae_mjy_sr'])
            elapsed=elapsed_prior+time.monotonic()-started
            record.update(elapsed_seconds=elapsed,peak_cuda_gib=torch.cuda.max_memory_allocated(device)/2**30)
            save_checkpoint(output/'last.pt',model,optimizer,scheduler,generator,signature=signature,
                epoch=epoch,step=updates,best=best,wandb_id=run.id,train_indices=ti,validation_indices=vi,elapsed_seconds=elapsed)
            if improved:
                shutil.copy2(output/'last.pt',output/'best.pt')
                run.summary['best_epoch']=epoch; run.summary['best_generated_mae_mjy_sr']=best
            if epoch%args.evaluate_every==0 or epoch==args.epochs:
                render(predictions,gallery,evaluation,device,run,output,epoch,updates)
            run.log(record)
            with (output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record,allow_nan=False)+'\n')
            (output/'progress.json').write_text(json.dumps(dict(**record,best=best if np.isfinite(best) else None,wandb_url=run.url),indent=2))
            print(json.dumps(record),flush=True)
        if end==args.epochs:
            best_state=torch.load(output/'best.pt',map_location=device,weights_only=False)
            model.load_state_dict(best_state['model'])
            replayed,_,_=evaluate_generation(model,evaluation,vi,device,args.sampling_steps,args.eval_seed,args.batch_size,output,'best-monitor-replay')
            if not np.isclose(replayed['mae_mjy_sr'],best,rtol=1e-6,atol=1e-9):
                raise RuntimeError('Best checkpoint did not reproduce its monitor selection metric')
            final_indices=list(range(len(evaluation)))
            (output/'full_validation_selection.json').write_text(json.dumps(dict(rows=evaluation.indices[final_indices].tolist()),indent=2))
            (output/'training_status.json').write_text(json.dumps(dict(training_complete=True,final_analysis_complete=False,epoch=end,updates=updates,best_epoch=best_state['epoch']),indent=2))
            arrays=[]; samples=[]
            for i in range(args.samples):
                metrics,measurements,predictions=evaluate_generation(model,evaluation,final_indices,device,args.sampling_steps,args.eval_seed+i,args.batch_size,output,f'best-sample-{i}')
                arrays.append(measurements); samples.append(predictions)
            if args.samples>1:
                ensemble=evaluate_ensemble(evaluation,samples,arrays[0],output)
                run.summary['full_validation_ensemble']=ensemble
                report=repeated_shape_report(arrays)
                report.update(checkpoint_epoch=best_state['epoch'],seeds=[args.eval_seed+i for i in range(args.samples)])
                (output/'sample_variation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
                run.summary['sample_variation']=report
                plot_repeated_samples(samples,gallery,evaluation,output/'sample_variation.png',report['seeds'])
                run.log({'samples/fixed_input_multiple_seeds':wandb.Image(str(output/'sample_variation.png'))})
            run.summary['best_epoch']=best_state['epoch']
            (output/'training_status.json').write_text(json.dumps(dict(training_complete=True,final_analysis_complete=True,epoch=end,updates=updates,best_epoch=best_state['epoch']),indent=2))
        (output/'status.json').write_text(json.dumps(dict(complete=end==args.epochs,epoch=end,updates=updates,
            best=best if np.isfinite(best) else None,wandb_url=run.url,checkpoint_sha256=sha256(output/'last.pt')),indent=2))
        train.close()
        if evaluation is not train: evaluation.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['train'],default='train')
    p.add_argument('--train-limit',type=int,default=0); p.add_argument('--validation-limit',type=int,default=256)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--width',type=int,default=32); p.add_argument('--timesteps',type=int,default=1000)
    p.add_argument('--sampling-steps',type=int,default=50); p.add_argument('--samples',type=int,default=3)
    p.add_argument('--evaluate-every',type=int,default=1); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=20261016); p.add_argument('--eval-seed',type=int,default=314159)
    p.add_argument('--device',default='cuda:1'); p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epochs',type=int,default=0)
    args=p.parse_args()
    if args.train_limit<0 or min(args.validation_limit,args.epochs,args.batch_size,args.samples,args.evaluate_every)<=0:
        p.error('Counts and limits must be positive')
    main(args)
