#!/bin/bash
# Continue target-oriented training from update 250 through update 1000.
# Ceiling: 4 CPU cores * 10 h * CNY 0.07 + 1 GPU * 10 h * CNY 2.20
#          = CNY 2.80 + CNY 22.00 = CNY 24.80 worst case.
# Submission requires explicit approval of this exact envelope.
#SBATCH --job-name=orcjax-target-u1000
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/target_training_1000_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/target_training_1000_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
PREPARED_MANIFEST=$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json
VALIDATION_PREPARED_MANIFEST=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/prepared_manifest.json
STAGE_CHECKPOINT=$ROOT/runtime/outputs/training/target-oriented-training-formal-e363482/stage-update-0250.pkl
STAGE_CHECKPOINT_SHA256=099d48c76277cafb04daf868c492200d0271fd6a4a2c6fa29ca6e4bdbe7989a5
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
MODE=formal
STAGE_START=250
STAGE_END=1000
PREPARED_UPDATES=1000
RUN_TIMEOUT=580m
OUTPUT_DIR=$ROOT/runtime/outputs/training/target-oriented-training-formal-u1000-${EXPECTED_GIT_HEAD:0:7}

export WORKTREE EXPECTED_GIT_HEAD PREPARED_MANIFEST VALIDATION_PREPARED_MANIFEST
export STAGE_CHECKPOINT STAGE_CHECKPOINT_SHA256 GPU_DEVICE MODE
export STAGE_START STAGE_END PREPARED_UPDATES RUN_TIMEOUT OUTPUT_DIR

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_training.sh"
