#!/bin/bash
# Frozen 250-update direct-GPP specialist warm start.
# Ceiling: 8 CPU cores * 1.25 h * CNY 0.07 + 1 GPU * 1.25 h * CNY 2.20
#          = CNY 0.70 + CNY 2.75 = CNY 3.45 worst case.
# Submission requires explicit approval of this exact envelope.
#SBATCH --job-name=orcjax-gpp-u250
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_warm_u250_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_warm_u250_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-structured-gpp-warm-u250-${SLURM_JOB_ID}}
RUN_TIMEOUT=70m

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT_DIR RUN_TIMEOUT

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_structured_gpp_warm_start.sh"
