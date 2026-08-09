#!/bin/bash
# Submit only after the exact resources, wall time, command, and cost are approved.
# This is one no-checkpoint, 500-update run containing both matched arms.
#SBATCH --job-name=orcjax-successor-fit
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/successor_small_fit_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/successor_small_fit_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=$ROOT/runtime/worktrees/fast-owner-gpu-probe-4b2cdd6
EXPECTED_GIT_HEAD=${SUCCESSOR_SMALL_FIT_EXPECTED_GIT_HEAD:?set the explicitly approved Git HEAD}
UPDATES=500
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT=$ROOT/runtime/outputs/training/successor-small-fit-500u-$HEAD_SHORT/runner_report.json
CACHE_ROOT=$ROOT/runtime/cache/jax/orcjax_gpu/successor-small-fit-runner-6710b35
SLURM_BIN=/rmprog/slurm/v22.05.7/bin

: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
test -x "$SLURM_BIN/srun"
test -d "$ROOT/runtime/logs"
test ! -e "$OUTPUT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(git status --porcelain --untracked-files=all)"

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE UPDATES OUTPUT CACHE_ROOT
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES gpu_device=$GPU_DEVICE updates=$UPDATES output=$OUTPUT"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

"$SLURM_BIN/srun" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v bash scripts/hpc/run_successor_small_fit.sh

test -f "$OUTPUT"
sha256sum "$OUTPUT"
