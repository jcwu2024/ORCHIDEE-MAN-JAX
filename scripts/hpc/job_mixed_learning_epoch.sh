#!/bin/bash
#SBATCH -J mixed-epoch5415
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/mixed-learning-epoch-v1/slurm-%j.log
#SBATCH --no-requeue
# Approval required: CPU8*24*.07=13.44 + GPU1*24*2.20=52.80 = CNY66.24 maximum.
# Create the log directory before sbatch. One job; no automatic paid resubmission.
set -euo pipefail
export CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-epoch-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_epoch.sh" train "$@"
