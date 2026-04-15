#!/bin/bash
#SBATCH --job-name=resnet_training
#SBATCH --account=m2218
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/resnet_training_%j.out
#SBATCH --error=logs/resnet_training_%j.err

mkdir -p logs

# JAX/XLA CPU threading — match to cpus-per-task
# export XLA_FLAGS="--xla_force_host_platform_device_count=256"
# export OMP_NUM_THREADS=64


module load conda
conda activate astroconda

cd /global/cfs/cdirs/m2218/eramey16/SR_data
python /global/homes/e/eramey16/superNISP/code/claude_model_NIR_2.py --from_run tbra2akh --n_stage1 50 --n_stage2 50
