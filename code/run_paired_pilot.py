"""Gated overfit/resume check followed by a short train/validation pilot."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    from train_paired import persistent_session_required
    persistent_session_required()
    checks = json.loads((ROOT/'runs/check/checks.json').read_text())
    if len(checks)!=4 or any(value!='passed' for value in checks.values()):
        raise RuntimeError('Integration checks must pass before training')
    common = [sys.executable,'-u',str(ROOT/'code/train_paired.py'),
              '--data',str(ROOT.parent/'supernisp_nir_y_f115w_v1'),
              '--project','superNISP-corrected','--head','legacy','--representation','linear']
    output = ROOT/'runs/residual-overfit-s20261016'
    overfit = common+['--output',str(output),'--mode','overfit','--train-limit','32',
                     '--epochs','40','--batch-size','8','--workers','0']
    subprocess.run(overfit+['--stop-after-epochs','2'],check=True,cwd=ROOT)
    paused = json.loads((output/'status.json').read_text())
    assert paused['epoch']==2 and not paused['complete']
    subprocess.run(overfit+['--resume'],check=True,cwd=ROOT)
    initial = json.loads((output/'initial_metrics.json').read_text())
    final = json.loads((output/'status.json').read_text())
    records = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in records] == list(range(1,41))
    assert final['updates']==160 and final['complete']
    # A debugging gate, not a scientific improvement threshold.
    if final['best'] >= .9*initial['mae']:
        raise RuntimeError('32-pair loss did not decrease by 10%; inspect before pilot')
    subprocess.run(common+['--output',str(ROOT/'runs/residual-pilot-s20261016'),
        '--mode','pilot','--train-limit','1024','--validation-limit','256',
        '--epochs','3','--batch-size','8','--workers','2'],check=True,cwd=ROOT)


if __name__=='__main__':
    main()
