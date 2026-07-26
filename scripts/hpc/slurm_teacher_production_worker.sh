#!/bin/bash
# Submit with an explicit array, for example:
# sbatch --array=0-3%2 --export=ALL,TEACHER_PLAN=/abs/plan.json,TEACHER_WORKER_COUNT=4 \
#   scripts/hpc/slurm_teacher_production_worker.sh
#SBATCH --job-name=orcjax-teacher
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_%A_%a.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_%A_%a.txt
#SBATCH --no-requeue

set -euo pipefail

REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$REPO/runtime
WORKTREE=${WORKTREE:-$REPO}
PYTHON=$REPO/.venvs/orcjax_cpu/bin/python
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
: "${TEACHER_PLAN:?TEACHER_PLAN must be an absolute generation-plan path}"
: "${TEACHER_WORKER_COUNT:?TEACHER_WORKER_COUNT must be set}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

case "$TEACHER_PLAN" in
  "$RUNTIME_ROOT"/*) ;;
  *) echo "TEACHER_PLAN must stay under $RUNTIME_ROOT" >&2; exit 2 ;;
esac
case "$WORKTREE" in
  "$REPO"|"$RUNTIME_ROOT/worktrees/"*) ;;
  *) echo "WORKTREE must be the repository root or stay under runtime/worktrees" >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TEACHER_WORKER_COUNT )); then
  echo "array index is outside TEACHER_WORKER_COUNT" >&2
  exit 2
fi

export ORCHIDEE_REPO_ROOT=$WORKTREE
export ORCHIDEE_RUNTIME_ROOT=$RUNTIME_ROOT
export ORCHIDEE_DATA_ROOT=$RUNTIME_ROOT/data
export ORCHIDEE_REFERENCE_ROOT=$RUNTIME_ROOT/assets
export ORCHIDEE_OUTPUT_ROOT=$RUNTIME_ROOT/outputs
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg
CACHE_ROOT=${TEACHER_XLA_CACHE:-$RUNTIME_ROOT/outputs/xla_cache/teacher_production}
export JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/worker-$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

test -x "$PYTHON"
test -x "$SLURM_BIN/srun"
test -f "$TEACHER_PLAN"
mkdir -p "$RUNTIME_ROOT/logs" "$JAX_COMPILATION_CACHE_DIR"
cd "$WORKTREE"
test -z "$(git status --porcelain --untracked-files=all)"

echo "git_head=$(git rev-parse HEAD)"
echo "job_id=$SLURM_JOB_ID array_index=$SLURM_ARRAY_TASK_ID worker_count=$TEACHER_WORKER_COUNT host=$(hostname)"
echo "plan=$TEACHER_PLAN cache=$JAX_COMPILATION_CACHE_DIR"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

COMMAND=(
  "$PYTHON" -m research.daily_coarse_graining.teacher_shards generate
  --plan "$TEACHER_PLAN"
  --worker-index "$SLURM_ARRAY_TASK_ID"
  --worker-count "$TEACHER_WORKER_COUNT"
)
if [[ -n "${TEACHER_MAX_NEW_ENTRIES:-}" ]]; then
  if ! [[ "$TEACHER_MAX_NEW_ENTRIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEACHER_MAX_NEW_ENTRIES must be a positive integer" >&2
    exit 2
  fi
  COMMAND+=(--max-new-entries "$TEACHER_MAX_NEW_ENTRIES")
fi

"$SLURM_BIN/srun" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v "${COMMAND[@]}"
