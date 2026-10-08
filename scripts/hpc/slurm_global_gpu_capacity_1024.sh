#!/bin/bash
# Proposed fixed1024: new B512x2 then resource-admitted B1024x1, same job/process.
# Preparation only; explicit resources, command, directory and cost approval required.
#SBATCH -J orchjax-gpu-cap1024
#SBATCH -p gnall
# Node selected explicitly after a fresh occupancy/load review.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o runtime/logs/global_gpu_capacity_1024_%j.txt
#SBATCH -e runtime/logs/global_gpu_capacity_1024_%j.txt
#SBATCH --no-requeue
set -Eeuo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-gpu-fixed1024-v1}
export RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1}
export FIXED_1024=1
export CONTINUE_512=0
export CONTINUE_256_512=0
export CONDITIONAL_512=0
export OPTIONAL_B64_BASELINE=0
# Slurm spools only this wrapper; all helpers come from declared immutable CODE_DIR.
exec bash "$CODE_DIR/scripts/hpc/slurm_global_gpu_capacity.sh" "$@"
