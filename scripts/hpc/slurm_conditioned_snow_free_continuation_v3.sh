#!/bin/bash
# Proposed ceiling: GPU 26.40 + CPU 6.72 = CNY 33.12. Separate approval required.
#SBATCH --job-name=orcjax-corrected40
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/corrected40_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/corrected40_%j.txt
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set execution commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPU}"
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/conditioned-snow-free-continuation-v3-${SLURM_JOB_ID}/report.json}
export CONTINUATION_CONTRACT=$WORKTREE/manifests/coarse_graining/conditioned_snow_free_continuation_v3.json
export RUN_TIMEOUT=710m PREFLIGHT_ONLY=0
exec bash "$WORKTREE/scripts/hpc/run_conditioned_snow_free_continuation.sh"
