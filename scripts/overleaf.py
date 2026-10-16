#!/usr/bin/env python3
"""Connect to Overleaf and compare manuscript files without overwriting edits.

Run `python scripts/overleaf.py connect` from a local interactive terminal.
Git handles authentication; this script never reads or stores a token.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
CHECKOUT = ROOT / '.overleaf/project'
URL = 'https://git@git.overleaf.com/6ad2883f967dc592692b8809'
EXTENSIONS = {'.tex', '.bib', '.cls', '.sty', '.bst', '.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg'}
ENTRYPOINT = '% Main document for the Overleaf project.\n\\input{survey_enhancements.tex}\n'
STATE = ROOT / '.overleaf/publish.json'


def manuscript_files(directory):
    """Include root manuscript resources and figs/, excluding experiment reports."""
    files = {}
    for path in directory.rglob('*'):
        relative = path.relative_to(directory)
        if any(part.startswith('.') for part in relative.parts):
            continue
        if len(relative.parts) > 1 and relative.parts[0] != 'figs':
            continue
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            files[relative.as_posix()] = path
    return files


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def bundle():
    files = manuscript_files(PAPER)
    if 'main.tex' in files:
        raise RuntimeError('A local main.tex now exists; review the generated entry point before exporting.')
    path = ROOT / '.overleaf/superNISP-paper.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
        archive.writestr('main.tex', ENTRYPOINT)
    print(f'Prepared {len(files)+1} manuscript files: {path}', flush=True)
    return files


def git(*args):
    return subprocess.check_output(['git', '-C', str(CHECKOUT), *args], text=True).strip()


def connect():
    created = not (CHECKOUT / '.git').is_dir()
    if created:
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', URL, str(CHECKOUT)], check=True)
    if git('remote', 'get-url', 'origin') != URL:
        raise RuntimeError('The checkout points to a different Overleaf project.')
    return created


def publish():
    """Push the manuscript; stop if Overleaf has new edits to reconcile."""
    files = bundle()
    created = connect()
    if git('status', '--porcelain'):
        raise RuntimeError('The isolated Overleaf checkout has local edits; reconcile them before publishing.')
    branch = git('branch', '--show-current')
    if not branch:
        raise RuntimeError('The Overleaf checkout is on a detached commit.')
    if not created:
        subprocess.run(['git', '-C', str(CHECKOUT), 'fetch', 'origin'], check=True)
    remote_head = git('rev-parse', f'origin/{branch}')
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state['url'] != URL:
            raise RuntimeError('Publish state belongs to another project.')
        if remote_head not in {state['remote_head'], state.get('pending_head')}:
            raise RuntimeError('Overleaf has new edits. Reconcile them with paper/ before publishing.')
    else:
        tracked = git('ls-tree', '-r', '--name-only', f'origin/{branch}').splitlines()
        if set(tracked) - {'main.tex'}:
            raise RuntimeError('This project is not blank. Compare existing files before the first publish.')
        if git('rev-parse', 'HEAD') != remote_head:
            raise RuntimeError('Unexpected local commits in the initial checkout.')
        state = dict(url=URL, remote_head=remote_head)
    # Reuse the user's existing Git identity, scoped to this isolated checkout.
    for setting in ['user.name', 'user.email']:
        value = subprocess.check_output(['git', '-C', str(ROOT), 'config', '--get', setting], text=True).strip()
        if not value:
            raise RuntimeError(f'Configure Git {setting} before publishing.')
        subprocess.run(['git', '-C', str(CHECKOUT), 'config', setting, value], check=True)
    for name, source in files.items():
        target = CHECKOUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (CHECKOUT / 'main.tex').write_text(ENTRYPOINT)
    subprocess.run(['git', '-C', str(CHECKOUT), 'add', '--', 'main.tex', *sorted(files)], check=True)
    if git('diff', '--cached', '--name-only'):
        subprocess.run(['git', '-C', str(CHECKOUT), 'commit', '-m',
                        'Sync superNISP manuscript and publication figures'], check=True)
    state['pending_head'] = git('rev-parse', 'HEAD')
    STATE.write_text(json.dumps(state, indent=2)+'\n')
    subprocess.run(['git', '-C', str(CHECKOUT), 'push', 'origin', f'HEAD:{branch}'], check=True)
    state['remote_head'] = state.pop('pending_head')
    STATE.write_text(json.dumps(state, indent=2)+'\n')
    print('Published manuscript and figures to Overleaf. Main document: main.tex')


def compare():
    if not (CHECKOUT / '.git').is_dir():
        raise RuntimeError('Not connected yet. Run: python scripts/overleaf.py connect')
    remote = subprocess.check_output(
        ['git', '-C', str(CHECKOUT), 'remote', 'get-url', 'origin'], text=True).strip()
    if remote != URL:
        raise RuntimeError('The existing checkout points to a different remote; inspect it before continuing.')
    local_files, remote_files = manuscript_files(PAPER), manuscript_files(CHECKOUT)
    rows = []
    for name in sorted(local_files.keys() | remote_files.keys()):
        local_sha = sha(local_files[name]) if name in local_files else None
        remote_sha = sha(remote_files[name]) if name in remote_files else None
        state = ('same' if local_sha == remote_sha else 'local only' if remote_sha is None
                 else 'Overleaf only' if local_sha is None else 'different')
        rows.append(dict(file=name, state=state, local_sha256=local_sha,
                         overleaf_checkout_sha256=remote_sha))
        if state != 'same':
            print(f'{state:14} {name}')
    report = ROOT / '.overleaf/comparison.json'
    report.write_text(json.dumps(dict(remote=URL, files=rows), indent=2) + '\n')
    print(f'{sum(row["state"] == "same" for row in rows)} identical files; report: {report}')
    print('Comparison uses the local Overleaf checkout. No manuscript files were copied or pushed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['connect', 'status', 'bundle', 'publish'])
    args = parser.parse_args()
    if args.command == 'connect':
        connect()
    elif args.command == 'bundle':
        bundle()
        return
    elif args.command == 'publish':
        publish()
        return
    compare()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
