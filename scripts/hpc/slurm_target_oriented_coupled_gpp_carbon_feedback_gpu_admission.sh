#!/bin/bash
# Zero-update real-v6 admission for the coupled GPP/carbon-feedback candidate.
# Ceiling: 4 CPU cores * 2 h * CNY 0.07 + 1 GPU * 2 h * CNY 2.20
#          = CNY 0.56 + CNY 4.40 = CNY 4.96 worst case.
# Submission requires explicit approval of this exact envelope.
# This job creates no optimizer and writes no checkpoint.
#SBATCH --job-name=orcjax-gpp-carbon-admit
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_carbon_admission_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_carbon_admission_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-carbon-feedback-real-v6-${SLURM_JOB_ID}.json}
RUN_TIMEOUT=110m

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT RUN_TIMEOUT
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_carbon_feedback_gpu_admission.sh"
