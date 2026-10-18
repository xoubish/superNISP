"""Bounded, disposable ResNet training probe on an already occupied GPU.

Requires tmux/screen and online W&B, as do the existing training entrypoints.
No checkpoint is saved or overwritten; this measures feasibility, not model quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import wandb

from paired_sample import PairedSample, masked_l1
from residual_baseline import EuclidToJWSTSuperResolution
from train_paired import persistent_session_required, paired_augment, seed_everything

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = 'be747f0cd0b02bb81ff0ab42a9ce8ea81a0e473c28c58d8406a428f758ea6b56'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot(uuid):
    text = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=memory.used,memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    used, free, total, utilization = [float(x.strip()) for x in text.strip().split(',')]
    text = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    processes = []
    for line in text.strip().splitlines():
        pid, name, memory = [x.strip() for x in line.split(',', 2)]
        processes.append(dict(pid=int(pid), name=name,
                              memory_mib=float(memory) if memory not in ['[N/A]', 'N/A'] else None))
    return dict(time=time.time(), used_mib=used, free_mib=free,
                total_mib=total, utilization_percent=utilization, processes=processes)


def main(args):
    persistent_session_required()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise RuntimeError('Expose only the selected GPU UUID to this process.')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ['probe_shared_gpu.py', 'train_paired.py', 'paired_sample.py',
               'residual_baseline.py', 'galaxy_visuals.py']
    for name in sources:
        shutil.copy2(ROOT/'code'/name, args.output/name)
    config = {key:str(value.resolve()) if isinstance(value, Path) else value
              for key,value in vars(args).items()}
    config.update(parent_sha256=sha(args.checkpoint), selected_gpu_only=True,
        allocator_cap_mib=args.memory_mib, no_compute_percentage_guarantee=True,
        trial_only=True, save_weights=False, split='train', learning_rate=1e-5,
        optimizer='Inherited Adam state; discard all probe updates',
        source_hashes={name:sha(ROOT/'code'/name) for name in sources})
    (args.output/'contract.json').write_text(json.dumps(config,indent=2)+'\n')
    if config['parent_sha256'] != PARENT_SHA:
        raise RuntimeError('Unexpected parent checkpoint')
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for name,digest in parent['signature']['data_hashes'].items():
        if sha(args.data/name) != digest:
            raise RuntimeError('Data mismatch: '+name)
    for name,digest in parent['signature']['source_hashes'].items():
        if sha(ROOT/'code'/name) != digest:
            raise RuntimeError('Parent source mismatch: '+name)
    before = gpu_snapshot(args.gpu_uuid)
    (args.output/'before.json').write_text(json.dumps(before,indent=2)+'\n')
    if before['free_mib'] < args.minimum_free_start_mib:
        raise RuntimeError('Insufficient free GPU memory for the bounded trial.')
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA device')
    device = torch.device('cuda:0')
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_mib*2**20/total, device)
    # This is a separate diagnostic seed; neither refinement branch resumes it.
    seed_everything(20261017)
    report = dict(complete=False, status='starting', parent_unchanged=None,
                  pid=os.getpid(), before=before, steps=[], snapshots=[], contract=config)
    dataset = None
    def write_report():
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def terminated(signum, frame):
        raise TimeoutError('Trial terminated by external timeout')
    signal.signal(signal.SIGTERM, terminated)
    write_report()
    try:
        with wandb.init(project='superNISP-corrected', entity='AI-Astro',
            name=args.output.name, job_type='shared-gpu-feasibility-probe', mode='online',
            dir=str(args.output), config=config, settings=wandb.Settings(init_timeout=60)) as run:
            if run.settings.mode != 'online' or not run.url:
                raise RuntimeError('Online W&B required before model execution')
            report['wandb_url'] = run.url
            (args.output/'wandb_url.txt').write_text(run.url+'\n')
            dataset = PairedSample(args.data, 'train', 'linear')
            count = min(args.steps*args.batch_size, len(dataset))
            indices = np.random.default_rng(20261017).permutation(len(dataset))[:count].tolist()
            (args.output/'selection.json').write_text(json.dumps(dict(
                split='train', rows=dataset.indices[indices].tolist()),indent=2)+'\n')
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=False)
            model = EuclidToJWSTSuperResolution(head='legacy').to(device)
            model.load_state_dict(parent['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            optimizer.load_state_dict(parent['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = 1e-5
            del parent
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            loop_start = time.perf_counter()
            for step,batch in enumerate(loader,1):
                if step > args.steps:
                    break
                if time.perf_counter()-loop_start > args.max_seconds:
                    report['status'] = 'time_limit'
                    break
                batch = paired_augment(batch)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                x,y,valid = [batch[k].to(device) for k in ['nisp','jwst','jwst_valid']]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = masked_l1(prediction,y,valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter()-start
                record = dict(step=step, batch=len(x), seconds=seconds,
                    loss=float(loss), gradient_norm=float(norm),
                    allocated_mib=torch.cuda.memory_allocated(device)/2**20,
                    reserved_mib=torch.cuda.memory_reserved(device)/2**20)
                report['steps'].append(record)
                run.log(record)
                if step == 1 or step % 10 == 0:
                    snap = gpu_snapshot(args.gpu_uuid)
                    snap['step'] = step
                    report['snapshots'].append(snap)
                    write_report()
                    print(json.dumps(record|{'device_free_mib':snap['free_mib']}),flush=True)
                    if snap['free_mib'] < args.minimum_free_during_mib:
                        report['status'] = 'low_free_memory'
                        break
            report['elapsed_training_seconds'] = time.perf_counter()-loop_start
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/2**20
            timed = [r['seconds'] for r in report['steps'][args.warmup_steps:]]
            report['measured_steps_after_warmup'] = len(timed)
            if timed:
                report.update(median_step_seconds=float(np.median(timed)),
                    p90_step_seconds=float(np.percentile(timed,90)),
                    measured_objects_per_second=args.batch_size/float(np.mean(timed)),
                    estimated_training_minutes_per_10_epochs=float(np.mean(timed))*22270/60)
            report['completed_steps'] = len(report['steps'])
            report['complete'] = report['completed_steps'] == args.steps
            if report['complete']:
                report['status'] = 'passed'
            report['memory_at_end'] = gpu_snapshot(args.gpu_uuid)
            report['own_peak_sampled_process_mib'] = max(
                [p['memory_mib'] for snap in report['snapshots']
                 for p in snap['processes'] if p['pid'] == os.getpid() and p['memory_mib'] is not None],
                default=None)
            report['parent_unchanged'] = sha(args.checkpoint) == PARENT_SHA
            run.summary.update({k:v for k,v in report.items() if isinstance(v,(str,int,float,bool))})
            write_report()
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      completed_steps=len(report['steps']), parent_unchanged=sha(args.checkpoint) == PARENT_SHA)
        write_report()
        raise
    finally:
        if dataset is not None:
            dataset.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ['steps','snapshots','contract','before','memory_at_end']},indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/residual-full50-s20261016/best.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--warmup-steps',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--memory-mib',type=int,default=1024)
    p.add_argument('--minimum-free-start-mib',type=int,default=1792)
    p.add_argument('--minimum-free-during-mib',type=int,default=256)
    p.add_argument('--max-seconds',type=float,default=180)
    args=p.parse_args()
    if min(args.steps,args.batch_size,args.memory_mib,args.max_seconds)<=0 or not 0<=args.warmup_steps<args.steps:
        p.error('Invalid trial budget')
    main(args)
