"""Yield GPU 1 at batch boundaries while the existing direct5 analysis uses it."""
import json
from pathlib import Path
import time
import wandb
from diffusion_corrected import CorrectedDiffusion

ROOT=Path(__file__).resolve().parents[1]


def analysis_pending(training,pipeline,exit_file):
    if exit_file.exists():
        return False  # Child processes have exited, whether successful or failed.
    if pipeline.exists():
        try:
            state=json.loads(pipeline.read_text())
            if state.get('complete'): return False
            if state.get('stage') in ['fixed_shapes','fixed_comparison','robustness','robust_comparison']:
                return True
        except json.JSONDecodeError:
            return True  # A status write is in progress; retry conservatively.
    if training.exists():
        try: return bool(json.loads(training.read_text()).get('complete'))
        except json.JSONDecodeError: return True
    return False


def yield_to_analysis():
    paths=(ROOT/'runs/direct5-full50-s20261016/status.json',
           ROOT/'runs/direct5-comparison/status.json',ROOT/'runs/direct5-console/exit_status')
    if not analysis_pending(*paths): return
    print('Yielding GPU 1 at a batch boundary for queued direct5 analyses.',flush=True)
    wandb.run.log({'runtime/waiting_for_direct5_analysis':1})
    started=time.monotonic()
    while analysis_pending(*paths):
        if time.monotonic()-started>7200:
            raise TimeoutError('Direct5 analysis wait exceeded two hours; inspect its console')
        time.sleep(5)
    if paths[-1].exists() and paths[-1].read_text().strip()!='0':
        print('Direct5 pipeline exited unsuccessfully; see its console. GPU 1 is released.',flush=True)
        wandb.run.summary['direct5_pipeline_exit_success']=False
    waited=time.monotonic()-started
    wandb.run.log({'runtime/waiting_for_direct5_analysis':0,'runtime/direct5_wait_seconds':waited})
    print(f'Resuming diffusion after {waited:.1f} seconds waiting for direct5 analysis.',flush=True)


class CooperativeDiffusion(CorrectedDiffusion):
    def training_loss(self,*args,**kwargs):
        yield_to_analysis()
        return super().training_loss(*args,**kwargs)

    def sample(self,*args,**kwargs):
        yield_to_analysis()
        return super().sample(*args,**kwargs)
