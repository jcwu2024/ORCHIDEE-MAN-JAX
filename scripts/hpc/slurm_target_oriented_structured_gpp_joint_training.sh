#!/bin/bash
# Frozen batch-four structured-GPP joint training.
# Ceiling: 8 CPU cores * 10 h * CNY 0.07 + 1 GPU * 10 h * CNY 2.20
#          = CNY 5.60 + CNY 22.00 = CNY 27.60 worst case.
# Submission requires explicit approval of this exact envelope.
# Set OUTPUT_DIR to an existing run directory to resume its recovery.pkl.
#SBATCH --job-name=orcjax-gpp-joint
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_joint_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_joint_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-structured-gpp-joint-${SLURM_JOB_ID}}
RUN_TIMEOUT=590m

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT_DIR RUN_TIMEOUT

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_structured_gpp_joint_training.sh"
