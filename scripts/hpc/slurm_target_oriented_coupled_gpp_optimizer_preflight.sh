#!/bin/bash
# One discarded macro-batch optimizer proposal for the coupled GPP candidate.
# Ceiling: 4 CPU cores * 4 h * CNY 0.07 + 1 GPU * 4 h * CNY 2.20
#          = CNY 1.12 + CNY 8.80 = CNY 9.92 worst case.
# Submission requires explicit approval of this exact envelope.
# The proposed parameters and optimizer are not persisted.
#SBATCH --job-name=orcjax-gpp-opt-pre
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_optimizer_preflight_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_optimizer_preflight_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-optimizer-preflight-${SLURM_JOB_ID}.json}
RUN_TIMEOUT=230m

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT RUN_TIMEOUT
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_optimizer_preflight.sh"
