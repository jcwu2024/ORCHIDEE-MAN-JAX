#!/bin/bash
#SBATCH -J mixed-stages-pilot
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-1%2
#SBATCH --time=04:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/mixed-learning-stages-pilot-v1/slurm-%A_%a.log
#SBATCH --no-requeue
# Two independent single-GPU arms; no distributed gradient calculation.
# Per task: CPU8*4*.07=2.24 + GPU1*4*2.20=8.80 = CNY11.04.
# Both tasks: CNY22.08 maximum. Concrete paid approval before submission.
set -euo pipefail
export CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-stages-pilot-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"
case "${SLURM_ARRAY_TASK_ID:?declare pilot arm}" in
  0) ARM=joint ;;
  1) ARM=staged ;;
  *) exit 2 ;;
esac
export RUN_TIMEOUT=220m
echo "pilot_arm=$ARM array_job=${SLURM_ARRAY_JOB_ID:-dry-run} task=$SLURM_ARRAY_TASK_ID"
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_stages.sh" "$ARM" "$@"
