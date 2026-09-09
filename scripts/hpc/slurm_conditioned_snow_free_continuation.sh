#!/bin/bash
# Approved ceiling: 1 GPU * 6h * 2.20 + 8 Slurm CPUs * 6h * 0.07 = CNY 16.56.
#SBATCH --job-name=orcjax-corrected10
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/corrected10_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/corrected10_%j.txt
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set execution commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPU}"
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/conditioned-snow-free-continuation-${SLURM_JOB_ID}/report.json}
export RUN_TIMEOUT=350m PREFLIGHT_ONLY=0
exec bash "$WORKTREE/scripts/hpc/run_conditioned_snow_free_continuation.sh"
