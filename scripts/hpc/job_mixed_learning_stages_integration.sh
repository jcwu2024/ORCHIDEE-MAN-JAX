#!/bin/bash
#SBATCH -J mixed-stage-check
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-stages-integration-paid-%j.log
#SBATCH --no-requeue
# User authorizes moving the staged integration to paid GPUs on2026-09-15.
# Cap: CPU8*1.5*.07=.84 + GPU1*1.5*2.20=3.30 = CNY4.14.
set -euo pipefail
export CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-stages-integration-paid-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_TIMEOUT=85m
export OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-stages-integration-paid-v1.json
cd "$CODE_ROOT"
# Use the Slurm entry; the free integration launcher rejects paid-node hosts.
# Python still runs only the declared two-update integration identity.
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_epoch.sh" train "$@" \
  --integration-updates 2 \
  --learning-schedule "$CODE_ROOT/manifests/coarse_graining/mixed_learning_stages_integration_v1.json"
