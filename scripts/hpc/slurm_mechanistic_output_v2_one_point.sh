#!/bin/bash
# Formal 100-epoch one-point memorization gate for mechanistic output v2.
# Ceiling: 4 CPU cores * 2 h * CNY 0.07 + 1 GPU * 2 h * CNY 2.20
#          = CNY 0.56 + CNY 4.40 = CNY 4.96.
# Submission requires explicit approval of this exact resource envelope.
#SBATCH --job-name=orcjax-mech-v2-1py
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/mechanistic_output_v2_one_point_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/mechanistic_output_v2_one_point_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/mechanistic-output-v2-one-point-$HEAD_SHORT/report.json}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/mechanistic-output-one-epoch-smoke-ba7f5d9}
SRUN=/rmprog/slurm/v22.05.7/bin/srun

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must be an isolated project runtime checkout" >&2; exit 2 ;;
esac
case "$OUTPUT" in
  "$ROOT/runtime/outputs/"*) ;;
  *) echo "OUTPUT escaped the project runtime outputs" >&2; exit 2 ;;
esac
case "$CACHE_ROOT" in
  "$ROOT/runtime/cache/"*) ;;
  *) echo "CACHE_ROOT escaped the project runtime cache" >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
test -x "$SRUN"
test "${SLURM_JOB_PARTITION:?missing partition allocation}" = gnall
test "${SLURM_CPUS_PER_TASK:?missing CPU allocation}" = 4
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
mkdir -p "$ROOT/runtime/logs"

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT CACHE_ROOT
export MECHANISTIC_OUTPUT=1 TRAINING_SMOKE=0 MAXIMUM_EPOCHS=100 EVALUATION_INTERVAL=5
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "gpu_device=$GPU_DEVICE output=$OUTPUT cache_root=$CACHE_ROOT"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

exec "$SRUN" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v bash \
    "$WORKTREE/scripts/hpc/run_strong_baseline_one_point_qualification.sh"
