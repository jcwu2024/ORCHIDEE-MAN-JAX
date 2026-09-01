#!/bin/bash
# First bounded coupled-GPP training stage.
# Ceiling: 8 CPU cores * 5 h * CNY 0.07 + 1 GPU * 5 h * CNY 2.20
#          = CNY 2.80 + CNY 11.00 = CNY 13.80 worst case.
# Submission requires explicit approval of this exact envelope.
# Set OUTPUT_DIR to the existing run directory to resume its recovery.pkl.
#SBATCH --job-name=orcjax-gpp-coupled
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_coupled_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_coupled_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-bounded-${SLURM_JOB_ID}}
RUN_TIMEOUT=290m
ADMISSION_ONLY=0

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT_DIR RUN_TIMEOUT ADMISSION_ONLY
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT_DIR"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_bounded_training.sh"
