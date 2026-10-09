#!/bin/bash
# Proposed fixed2048: symmetric fresh B1024x2 and conditional B2048x1 processes.
# Preparation only; explicit resources, node, command, directory and cost approval required.
#SBATCH -J orchjax-gpu-cap2048
#SBATCH -p gnall
# Node selected explicitly after a fresh occupancy/load review.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o runtime/logs/global_gpu_capacity_2048_%j.txt
#SBATCH -e runtime/logs/global_gpu_capacity_2048_%j.txt
#SBATCH --no-requeue
set -Eeuo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-gpu-fixed2048-v1}
export RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1}
export FIXED_2048=1
export FIXED_1024=0
export CONTINUE_512=0
export CONTINUE_256_512=0
export CONDITIONAL_512=0
export OPTIONAL_B64_BASELINE=0
# Slurm spools this wrapper only; all helpers resolve from immutable CODE_DIR.
exec bash "$CODE_DIR/scripts/hpc/slurm_global_gpu_capacity.sh" "$@"
