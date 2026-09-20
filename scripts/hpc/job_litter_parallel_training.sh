#!/bin/bash
#SBATCH -J litter-parallel
#SBATCH -p gnall
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/litter-parallel-v1/slurm-%A_%a.log
#SBATCH --no-requeue
# User approved four paid arms: 4*(8*.07+2.20)*3 = CNY33.12 maximum.
# One GPU per arm, no auto retry, extension, or opportunistic extra arms.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT=$ROOT/runtime/checkouts/litter-parallel-v1
ARMS=(A B C D)
TASK=${SLURM_ARRAY_TASK_ID:?Slurm array index is required}
case "$TASK" in 0|1|2|3) ;; *) exit 2;; esac
export ARM=${ARMS[$TASK]}
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must declare allocated GPU}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_TIMEOUT=175m
export OUTPUT=$ROOT/runtime/outputs/training/litter-parallel-v1/$ARM/report.json
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_litter_parallel_training.sh" "$@"
