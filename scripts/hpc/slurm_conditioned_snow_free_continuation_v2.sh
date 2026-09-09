#!/bin/bash
# Proposed ceiling: GPU 17.60 + CPU 4.48 = CNY 22.08. Separate approval required.
#SBATCH --job-name=orcjax-corrected20
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/corrected20_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/corrected20_%j.txt
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set execution commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPU}"
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/conditioned-snow-free-continuation-v2-${SLURM_JOB_ID}/report.json}
export CONTINUATION_CONTRACT=$WORKTREE/manifests/coarse_graining/conditioned_snow_free_continuation_v2.json
export RUN_TIMEOUT=470m PREFLIGHT_ONLY=0
exec bash "$WORKTREE/scripts/hpc/run_conditioned_snow_free_continuation.sh"
