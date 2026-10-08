#!/bin/bash
# B256/B512-only proposed continuation; no submission authorization implied.
#SBATCH -J orchjax-gpu-cont
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o runtime/logs/global_gpu_capacity_continue_%j.txt
#SBATCH -e runtime/logs/global_gpu_capacity_continue_%j.txt
#SBATCH --no-requeue
set -Eeuo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-gpu-sweetspot-v2}
export RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-256-512-${SLURM_JOB_ID:-dry-run}}
export CONTINUE_256_512=1
export OPTIONAL_B64_BASELINE=0
# Resolve the launcher from this declared code artifact, never an older checkout.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/slurm_global_gpu_capacity.sh" "$@"
