"""Gated diffusion learning/resume check, followed by a fresh small pilot."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from train_paired import persistent_session_required,sha256
from train_diffusion_corrected import SOURCES

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())


def main():
    persistent_session_required()
    checks=read(ROOT/'runs/diffusion-corrected-check/checks.json')
    if not checks['complete'] or checks['regression_tests']!=5:
        raise RuntimeError('Diffusion integration checks required')
    if checks['source_sha256']!=sha256(ROOT/'code/diffusion_corrected.py'):
        raise RuntimeError('Model source changed since integration checks')
    if (ROOT/'runs/diffusion-check-console/exit_status').read_text().strip()!='0':
        raise RuntimeError('Integration-check process did not finish successfully')
    output=ROOT/'runs/diffusion-pilot-pipeline'
    overfit=ROOT/'runs/diffusion-corrected-overfit'
    pilot=ROOT/'runs/diffusion-corrected-pilot'
    for directory in [output,overfit,pilot]:
        if directory.exists(): raise RuntimeError(f'Refusing to overwrite {directory}')
    output.mkdir(parents=True); (output/'source').mkdir()
    sources=SOURCES+['run_diffusion_pilot.py']
    hashes={name:sha256(ROOT/'code'/name) for name in sources}
    for name in sources:
        path=output/'source'/name; path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'code'/name,path)
    common=[sys.executable,'-u',str(ROOT/'code/train_diffusion_corrected.py'),
            '--data',str(ROOT.parent/'supernisp_nir_y_f115w_v1'),'--device','cuda:1']
    command=common+['--output',str(overfit),'--mode','overfit','--train-limit','32',
        '--epochs','40','--evaluate-every','10','--sampling-steps','20','--samples','1']
    stages=[('overfit_start',command+['--stop-after-epochs','2']),('overfit_resume',command+['--resume']),
            ('pilot',common+['--output',str(pilot)])]
    (output/'protocol.json').write_text(json.dumps(dict(source_hashes=hashes,checks=checks,
        stages=stages,gate='Completed 40-epoch/320-update overfit, generated physical MAE reduced >=10%; then fresh 512-train/64-validation 5-epoch/640-update pilot.',
        selection='Fixed-seed NISP-only generation MAE in physical units',
        sampling='Overfit 20 DDIM steps; pilot 50; pilot best checkpoint sampled with 3 predeclared seeds.',
        budget='Bounded debugging experiment. No full-dataset diffusion training is queued.'),indent=2))
    for name,command in stages:
        for source,expected in hashes.items():
            if sha256(ROOT/'code'/source)!=expected: raise RuntimeError(f'Queued source changed: {source}')
        (output/'status.json').write_text(json.dumps(dict(complete=False,stage=name),indent=2))
        print('Starting stage: '+name,flush=True)
        subprocess.run(command,check=True,cwd=ROOT)
        if name=='overfit_start':
            status=read(overfit/'status.json')
            if status['epoch']!=2 or status['complete']: raise RuntimeError('Controlled interruption failed')
        else:
            directory=overfit if name=='overfit_resume' else pilot
            status=read(directory/'status.json')
            epochs,updates=(40,320) if name=='overfit_resume' else (5,640)
            records=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            if not status['complete'] or status['updates']!=updates or [r['epoch'] for r in records]!=list(range(1,epochs+1)):
                raise RuntimeError('Incomplete or duplicated training history')
            if name=='overfit_resume' and status['best']>=.9*read(overfit/'initial_metrics.json')['mae_mjy_sr']:
                raise RuntimeError('Generated 32-pair MAE did not improve by 10%; inspect before pilot')
        print('Completed stage: '+name,flush=True)
    (output/'status.json').write_text(json.dumps(dict(complete=True,stage='pilot_completed',
        pilot=read(pilot/'status.json')),indent=2))


if __name__=='__main__': main()
