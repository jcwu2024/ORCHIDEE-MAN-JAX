#!/bin/bash
# Proposal only: formal bounded 96-window successor 7/30-day screen.
# Ceiling: 4 CPU cores * 0.75 h * CNY 0.07 + 1 GPU * 0.75 h * CNY 2.20
#          = CNY 0.21 + CNY 1.65 = CNY 1.86 worst case.
# Submission requires a fresh explicit approval block.
#SBATCH --job-name=orcjax-successor-screen
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/successor_screen_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/successor_screen_%j.txt
#SBATCH --no-requeue

set -euo pipefail

WORKTREE=${WORKTREE:?set WORKTREE}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set PREPARED_MANIFEST}
PREPARED_MANIFEST_SHA256=${PREPARED_MANIFEST_SHA256:?set PREPARED_MANIFEST_SHA256}
WINDOW_PLAN=${WINDOW_PLAN:?set WINDOW_PLAN}
WINDOW_PLAN_FILE_SHA256=${WINDOW_PLAN_FILE_SHA256:?set WINDOW_PLAN_FILE_SHA256}
FIT_REPORT=${FIT_REPORT:?set FIT_REPORT}
FIT_REPORT_SHA256=${FIT_REPORT_SHA256:?set FIT_REPORT_SHA256}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
CHECKPOINT_MANIFEST_SHA256=${CHECKPOINT_MANIFEST_SHA256:?set CHECKPOINT_MANIFEST_SHA256}
OUTPUT=${OUTPUT:?set OUTPUT}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE PREPARED_MANIFEST \
  PREPARED_MANIFEST_SHA256 WINDOW_PLAN WINDOW_PLAN_FILE_SHA256 \
  FIT_REPORT FIT_REPORT_SHA256 CHECKPOINT_DIR CHECKPOINT_MANIFEST_SHA256 OUTPUT

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "gpu_device=$GPU_DEVICE prepared_manifest=$PREPARED_MANIFEST output=$OUTPUT"
exec bash "$WORKTREE/scripts/hpc/run_successor_rollout_screen.sh"
