#!/usr/bin/env bash
set -euo pipefail
cd /home/shemmati/Work/Projects/superNISP
branch="${1:?conditioned or deconv}"
case "$branch" in
  conditioned) export CUDA_VISIBLE_DEVICES=GPU-7f523ed6-dcff-1d8d-5f3a-b4b8fcc4c3b0 ;;
  deconv) export CUDA_VISIBLE_DEVICES=GPU-68974dfb-ad3e-4055-fe62-7e5d042d438b ;;
  *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/supernisp-mpl
psf_console="runs/psf-redesign-${branch}-v01-console"
mkdir -p "$psf_console"
trap 'echo "$?" > "$psf_console/exit_status"' EXIT
python -u code/train_psf_redesign.py \
  --data /home/shemmati/Work/Projects/supernisp_nir_y_f115w_v1 \
  --output "runs/psf-redesign-${branch}-v01" \
  --branch "$branch" --gpu-uuid "$CUDA_VISIBLE_DEVICES" \
  > "$psf_console/console.log" 2>&1
