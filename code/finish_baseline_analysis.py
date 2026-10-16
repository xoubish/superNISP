"""Wait for successful training, evaluate its best validation checkpoint, compare IDs."""
import json
from pathlib import Path
import subprocess
import sys
import time
from train_paired import persistent_session_required

ROOT=Path(__file__).resolve().parents[1]


def main():
    persistent_session_required()
    training=ROOT/'runs/residual-full50-s20261016'
    exit_file=ROOT/'runs/full50-console/exit_status'
    print('Waiting for full50 training to exit successfully; no GPU work while waiting.',flush=True)
    deadline=time.monotonic()+24*3600
    while not exit_file.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Training has not exited after 24 hours; inspect tmux/logs')
        time.sleep(30)
    if exit_file.read_text().strip()!='0':
        raise RuntimeError('Training exited unsuccessfully; final analysis was not started')
    status=json.loads((training/'status.json').read_text())
    if not status['complete'] or status['epoch']!=50:
        raise RuntimeError('Training has not completed its planned 50 epochs')
    final=ROOT/'runs/validation-shapes-full50-best'
    subprocess.run([sys.executable,'-u',str(ROOT/'code/evaluate_paired_shapes.py'),
        '--data',str(ROOT.parent/'supernisp_nir_y_f115w_v1'),
        '--checkpoint',str(training/'best.pt'),'--output',str(final),'--device','cuda:1'],check=True,cwd=ROOT)
    subprocess.run([sys.executable,'-u',str(ROOT/'code/compare_shape_runs.py'),
        '--before',str(ROOT/'runs/validation-shapes-full50-snapshot'),'--after',str(final),
        '--output',str(ROOT/'runs/full50-shape-comparison')],check=True,cwd=ROOT)
    print('Best-checkpoint validation and identical-population comparison completed.',flush=True)


if __name__=='__main__':
    main()
