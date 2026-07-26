#!/bin/bash
# Submit with --dependency=afterok:<worker-array-job-id> and the same exports.
#SBATCH --job-name=orcjax-teacher-aggregate
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_aggregate_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_aggregate_%j.txt
#SBATCH --no-requeue

set -euo pipefail

REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$REPO/runtime
WORKTREE=${WORKTREE:-$REPO}
PYTHON=$REPO/.venvs/orcjax_cpu/bin/python
: "${TEACHER_PLAN:?TEACHER_PLAN must be an absolute generation-plan path}"
: "${TEACHER_WORKER_COUNT:?TEACHER_WORKER_COUNT must be set}"

case "$TEACHER_PLAN" in
  "$RUNTIME_ROOT"/*) ;;
  *) echo "TEACHER_PLAN must stay under $RUNTIME_ROOT" >&2; exit 2 ;;
esac
case "$WORKTREE" in
  "$REPO"|"$RUNTIME_ROOT/worktrees/"*) ;;
  *) echo "WORKTREE must be the repository root or stay under runtime/worktrees" >&2; exit 2 ;;
esac

export ORCHIDEE_REPO_ROOT=$WORKTREE
export ORCHIDEE_RUNTIME_ROOT=$RUNTIME_ROOT
export ORCHIDEE_DATA_ROOT=$RUNTIME_ROOT/data
export ORCHIDEE_REFERENCE_ROOT=$RUNTIME_ROOT/assets
export ORCHIDEE_OUTPUT_ROOT=$RUNTIME_ROOT/outputs
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True

test -x "$PYTHON"
test -f "$TEACHER_PLAN"
mkdir -p "$RUNTIME_ROOT/logs"
cd "$WORKTREE"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON" -m research.daily_coarse_graining.teacher_shards aggregate \
  --plan "$TEACHER_PLAN" \
  --worker-count "$TEACHER_WORKER_COUNT"
