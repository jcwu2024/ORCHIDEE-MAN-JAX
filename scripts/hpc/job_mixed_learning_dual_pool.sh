#!/bin/bash
#SBATCH -J mixed-dual-pool
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/mixed-learning-dual-pool-v1/slurm-%j.log
#SBATCH --no-requeue
# Explicit approval required: (8*.07 + 2*2.20)*16 = CNY79.36 maximum.
# One fresh full-network run, no exploratory or integration checkpoint inherited.
# Create the log directory before sbatch. No automatic retry or paid extension.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT=$ROOT/runtime/checkouts/mixed-learning-dual-pool-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must declare allocated GPUs}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_TIMEOUT=950m
export OUTPUT=$ROOT/runtime/outputs/training/mixed-learning-dual-pool-v1/report.json
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_epoch.sh" train "$@" \
  --dual-device-training \
  --learning-schedule "$CODE_ROOT/manifests/coarse_graining/mixed_learning_joint_pool_v1.json"
