#!/bin/bash
# B512-only 45-minute proposal; sealed B128/B256 remain historical origins.
# Submission requires separate resource, command, directory and cost approval.
#SBATCH -J orchjax-gpu-b512
#SBATCH -p gnall
# Node selected explicitly after a fresh occupancy/load review.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -o runtime/logs/global_gpu_capacity_512_%j.txt
#SBATCH -e runtime/logs/global_gpu_capacity_512_%j.txt
#SBATCH --no-requeue
set -Eeuo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-gpu-metadata-v1}
export RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-${SLURM_JOB_ID:-dry-run}}
export CONTINUE_512=1
export CONTINUE_256_512=0
export OPTIONAL_B64_BASELINE=0
# Slurm spools only this wrapper. Resolve all helpers from declared CODE_DIR.
exec bash "$CODE_DIR/scripts/hpc/slurm_global_gpu_capacity.sh" "$@"
