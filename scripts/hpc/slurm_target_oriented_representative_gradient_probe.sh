#!/bin/bash
# Proposal only: one zero-update representative target-oriented gradient probe.
# Ceiling: 4 CPU cores * 1 h * CNY 0.07 + 1 GPU * 1 h * CNY 2.20
#          = CNY 0.28 + CNY 2.20 = CNY 2.48 worst case.
# Submission requires fresh explicit approval.
#SBATCH --job-name=orcjax-target-grad
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/target_gradient_probe_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/target_gradient_probe_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_representative_gradient_probe.sh"
