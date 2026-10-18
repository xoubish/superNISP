#!/usr/bin/env bash
set -euo pipefail
cd /home/shemmati/Work/Projects/superNISP
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/supernisp-mpl
mkdir -p runs/psf-redesign-comparison-v01-console
trap 'echo "$?" > runs/psf-redesign-comparison-v01-console/exit_status' EXIT
python -u code/compare_psf_redesign.py --wait \
  > runs/psf-redesign-comparison-v01-console/console.log 2>&1
