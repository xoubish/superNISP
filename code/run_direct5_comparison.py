"""Gated direct-5x experiment with the frozen baseline's 50-epoch budget."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'runs/residual-full50-s20261016'
OUTPUT = ROOT / 'runs/direct5-comparison'
DIRECT = ROOT / 'runs/direct5-full50-s20261016'
EVALUATION = ROOT / 'runs/validation-shapes-direct5-best'
ROBUSTNESS = ROOT / 'runs/shape-robustness-direct5-best'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_command(config):
    command = [sys.executable, '-u', str(ROOT / 'code/train_paired.py')]
    for key, value in config.items():
        if value is not None:
            command.extend(['--' + key.replace('_', '-'), str(value)])
    return command


def completed_training(directory, epochs, updates):
    status = read(directory / 'status.json')
    records = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    if not status['complete'] or status['epoch'] != epochs or status['updates'] != updates:
        raise RuntimeError(f'Incomplete training: {directory}')
    if [r['epoch'] for r in records] != list(range(1, epochs + 1)):
        raise RuntimeError(f'Missing or duplicate epochs: {directory}')
    if not math.isfinite(status['best']):
        raise RuntimeError(f'Nonfinite validation metric: {directory}')
    return status


def prepare():
    baseline = read(BASELINE / 'provenance.json')
    completed_training(BASELINE, 50, 111350)
    checks = read(ROOT / 'runs/check/checks.json')
    if len(checks) != 4 or any(value != 'passed' for value in checks.values()):
        raise RuntimeError('The four integration checks must have passed')
    for name, expected in baseline['source_hashes'].items():
        if sha(ROOT / 'code' / name) != expected:
            raise RuntimeError(f'Frozen training source changed: {name}')
    for name, expected in baseline['data_hashes'].items():
        if sha(Path(baseline['config']['data']) / name) != expected:
            raise RuntimeError(f'Frozen data changed: {name}')
    full = dict(baseline['config'], head='direct5', output=str(DIRECT))
    overfit_dir = ROOT / 'runs/direct5-overfit-s20261016'
    pilot_dir = ROOT / 'runs/direct5-pilot-s20261016'
    stages = [
        ('overfit', training_command(dict(full, output=str(overfit_dir), mode='overfit',
            train_limit=32, epochs=40, workers=0, gallery_count=4, image_every=10)), overfit_dir),
        ('pilot', training_command(dict(full, output=str(pilot_dir), mode='pilot',
            train_limit=1024, validation_limit=256, epochs=3)), pilot_dir),
        ('full50', training_command(full), DIRECT),
        ('fixed_shapes', [sys.executable, '-u', str(ROOT / 'code/evaluate_paired_shapes.py'),
            '--data', full['data'], '--checkpoint', str(DIRECT / 'best.pt'),
            '--output', str(EVALUATION), '--device', 'cuda:1'], EVALUATION),
        ('fixed_comparison', [sys.executable, '-u', str(ROOT / 'code/compare_shape_runs.py'),
            '--before', str(ROOT / 'runs/validation-shapes-full50-best'),
            '--after', str(EVALUATION), '--output', str(ROOT / 'runs/direct5-fixed-comparison')],
            ROOT / 'runs/direct5-fixed-comparison'),
        ('robustness', [sys.executable, '-u', str(ROOT / 'code/run_shape_robustness.py'),
            '--data', full['data'], '--training', str(DIRECT), '--output', str(ROBUSTNESS),
            '--nominal-reference', str(EVALUATION), '--device', 'cuda:1'], ROBUSTNESS),
        ('robust_comparison', [sys.executable, '-u', str(ROOT / 'code/compare_robustness_runs.py'),
            '--before', str(ROOT / 'runs/shape-robustness-best49'), '--after', str(ROBUSTNESS),
            '--output', str(ROOT / 'runs/direct5-robust-comparison')], ROOT / 'runs/direct5-robust-comparison'),
    ]
    for directory in [OUTPUT, *[stage[2] for stage in stages]]:
        if directory.exists():
            raise RuntimeError(f'Refusing to overwrite existing experiment: {directory}')
    sources = set(baseline['source_hashes']) | {
        'run_direct5_comparison.py', 'evaluate_paired_shapes.py', 'shape_diagnostics.py',
        'test_shape_diagnostics.py', 'compare_shape_runs.py', 'run_shape_robustness.py',
        'robust_shape_measurements.py', 'test_shape_robustness.py', 'compare_robustness_runs.py'}
    protocol = dict(baseline=str(BASELINE), baseline_best_sha256=sha(BASELINE / 'best.pt'),
        full_config=full, baseline_source_hashes=baseline['source_hashes'],
        source_hashes={name: sha(ROOT / 'code' / name) for name in sorted(sources)},
        data_hashes=baseline['data_hashes'], stages=[dict(name=n, command=c, output=str(d)) for n,c,d in stages],
        selection='Best validation masked L1, unchanged from baseline; final test untouched.',
        evaluation='Fixed and adaptive observed-shape errors, paired common IDs and spatial blocks, with failure counts.',
        budget='Fresh 50 epochs / 111350 updates; pilot and overfit weights are discarded.',
        gates='Overfit: 40 epochs/160 updates and >=10% MAE decrease. Pilot: 3 epochs/384 updates, finite loss and improvement over initialization.',
        caveats=['Same trunk architecture and seed; the larger head changes parameter count and compute.',
                 'Trunk initializes identically, but final reconstruction layers consume a different Torch RNG position after the new head.',
                 'Single-seed validation architecture comparison; not calibrated shear or a final test result.'])
    return protocol, stages


def main(check_only=False):
    protocol, stages = prepare()
    if check_only:
        print(json.dumps(protocol, indent=2))
        return
    if not (os.environ.get('TMUX') or os.environ.get('STY')):
        raise RuntimeError('Run inside tmux or screen')
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'source').mkdir()
    for name in protocol['source_hashes']:
        shutil.copy2(ROOT / 'code' / name, OUTPUT / 'source' / name)
    (OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    environment = dict(os.environ, WANDB_MODE='online', WANDB_ENTITY='AI-Astro')
    for name, command, directory in stages:
        for source, expected in protocol['source_hashes'].items():
            if sha(ROOT / 'code' / source) != expected:
                raise RuntimeError(f'Experiment source changed while queued: {source}')
        (OUTPUT / 'status.json').write_text(json.dumps(dict(complete=False, stage=name), indent=2))
        print(f'Starting stage: {name}', flush=True)
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
        if name in ('overfit', 'pilot', 'full50'):
            epochs, updates = {'overfit': (40,160), 'pilot': (3,384), 'full50': (50,111350)}[name]
            status = completed_training(directory, epochs, updates)
            if name != 'full50':
                initial = read(directory / 'initial_metrics.json')['mae']
                threshold = .9 if name == 'overfit' else 1.
                if status['best'] >= threshold * initial:
                    raise RuntimeError(f'{name} did not pass its learning gate')
            else:
                actual = read(directory / 'provenance.json')
                if actual['config'] != protocol['full_config']:
                    raise RuntimeError('Full-run configuration differs from predeclared protocol')
                if read(directory / 'selection.json') != read(BASELINE / 'selection.json'):
                    raise RuntimeError('Training/validation rows differ from the baseline')
        print(f'Completed stage: {name}', flush=True)
    (OUTPUT / 'status.json').write_text(json.dumps(dict(complete=True, stage='all_completed'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify files and print protocol without running models')
    main(parser.parse_args().check_only)
