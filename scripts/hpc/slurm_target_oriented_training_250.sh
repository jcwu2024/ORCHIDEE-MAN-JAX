#!/bin/bash
# Formal target-oriented updates 1-250 plus initialization/update-250 validation.
# Ceiling: 4 CPU cores * 5 h * CNY 0.07 + 1 GPU * 5 h * CNY 2.20
#          = CNY 1.40 + CNY 11.00 = CNY 12.40 worst case.
# Submission requires explicit approval of this exact envelope.
#SBATCH --job-name=orcjax-target-u250
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/target_training_250_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/target_training_250_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
PREPARED_MANIFEST=$ROOT/runtime/outputs/training/target-oriented-prepared-250-9b82b5a/prepared_manifest.json
VALIDATION_PREPARED_MANIFEST=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/prepared_manifest.json
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
MODE=formal

export WORKTREE EXPECTED_GIT_HEAD PREPARED_MANIFEST VALIDATION_PREPARED_MANIFEST GPU_DEVICE MODE

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_training.sh"
