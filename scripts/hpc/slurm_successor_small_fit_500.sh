#!/bin/bash
# Proposal only: one checkpoint-producing 500-update run containing both matched arms.
# Ceiling: 16 CPU cores * 2 h * CNY 0.07 + 1 GPU * 2 h * CNY 2.20
#          = CNY 2.24 + CNY 4.40 = CNY 6.64 worst case.
# Submission requires a fresh explicit approval block.
#SBATCH --job-name=orcjax-successor-fit
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/successor_fit_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/successor_fit_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set PREPARED_MANIFEST to the accepted 500-entry manifest}
UPDATES=500
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/successor-small-fit-500u-$HEAD_SHORT/runner_report.json}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$(dirname "$OUTPUT")/checkpoints}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/successor-small-fit-500u-$HEAD_SHORT}
SRUN=/rmprog/slurm/v22.05.7/bin/srun

case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
for path in "$PREPARED_MANIFEST" "$OUTPUT" "$CHECKPOINT_DIR" "$CACHE_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime path escaped the project root: $path" >&2; exit 2 ;;
  esac
done
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
test -x "$SRUN"
test -f "$PREPARED_MANIFEST"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
test ! -e "$CHECKPOINT_DIR"
mkdir -p "$ROOT/runtime/logs"

export WORKTREE EXPECTED_GIT_HEAD PREPARED_MANIFEST GPU_DEVICE UPDATES OUTPUT CHECKPOINT_DIR CACHE_ROOT
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES gpu_device=$GPU_DEVICE updates=$UPDATES output=$OUTPUT"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

"$SRUN" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v bash "$WORKTREE/scripts/hpc/run_successor_small_fit.sh"

test -f "$OUTPUT"
test -f "$CHECKPOINT_DIR/checkpoint_manifest.json"
sha256sum "$OUTPUT"
sha256sum "$CHECKPOINT_DIR/checkpoint_manifest.json"
