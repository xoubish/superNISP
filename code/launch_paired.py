"""Start a named tmux session with a durable console log and exit status.

Example: python code/launch_paired.py supernisp-check runs/check-console -- \
python -u code/check_paired_training.py --data ../supernisp_nir_y_f115w_v1 --output runs/check
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session')
    p.add_argument('log_directory',type=Path)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command
    if command and command[0]=='--':
        command = command[1:]
    if not command:
        p.error('Missing command')
    log = args.log_directory.resolve()
    log.mkdir(parents=True,exist_ok=True)
    if (log/'launch.sh').exists():
        p.error('Use a fresh console directory to preserve prior logs')
    script = log/'launch.sh'
    script.write_text('#!/usr/bin/env bash\nset -uo pipefail\n'+
        'export WANDB_MODE=online\nexport PYTHONUNBUFFERED=1\n'+
        shlex.join(command)+' 2>&1 | tee '+shlex.quote(str(log/'console.log'))+'\n'+
        'status=${PIPESTATUS[0]}\nprintf "%s\\n" "$status" > '+shlex.quote(str(log/'exit_status'))+'\n'+
        'printf "Process finished with exit status %s. Logs: %s\\n" "$status" '+shlex.quote(str(log))+'\n'+
        'exec bash\n')
    subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',os.getcwd(),
                    'bash '+shlex.quote(str(script))],check=True)
    print(f'Attach: tmux attach -t {args.session}\nLog: {log / "console.log"}')


if __name__=='__main__':
    main()
