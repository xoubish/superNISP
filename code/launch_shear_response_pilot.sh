#!/usr/bin/env bash
set -euo pipefail
cd /home/shemmati/Work/Projects/superNISP
export CUDA_VISIBLE_DEVICES=GPU-7f523ed6-dcff-1d8d-5f3a-b4b8fcc4c3b0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/supernisp-mpl
mkdir -p runs/shear-response-pilot-v01-console
trap 'echo "$?" > runs/shear-response-pilot-v01-console/exit_status' EXIT
timeout 1800 python -u code/pilot_shear_response.py --device cuda:0 \
    > runs/shear-response-pilot-v01-console/console.log 2>&1
