#!/bin/bash

set -euo pipefail

REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$REPO/runtime
WORKTREE=${WORKTREE:-$REPO}
PYTHON=$REPO/.venvs/orcjax_cpu/bin/python

: "${TEACHER_PLAN:?TEACHER_PLAN must be an absolute generation-plan path}"
: "${TEACHER_WORKER_COUNT:?TEACHER_WORKER_COUNT must be set}"
: "${TEACHER_EXPECTED_GIT_HEAD:?TEACHER_EXPECTED_GIT_HEAD must be set}"
: "${SLURM_PROCID:?run this script through a multi-task srun}"
: "${SLURM_LOCALID:?SLURM_LOCALID is required for per-node staggering}"

TEACHER_WORKER_OFFSET=${TEACHER_WORKER_OFFSET:-0}
TEACHER_STARTUP_STAGGER_SECONDS=${TEACHER_STARTUP_STAGGER_SECONDS:-0}
if [[ -n "${TEACHER_WORKER_INDICES:-}" ]]; then
  IFS=',' read -r -a WORKER_INDICES <<< "$TEACHER_WORKER_INDICES"
  if (( SLURM_PROCID >= ${#WORKER_INDICES[@]} )); then
    echo "SLURM_PROCID is outside TEACHER_WORKER_INDICES" >&2
    exit 2
  fi
  WORKER_INDEX=${WORKER_INDICES[$SLURM_PROCID]}
else
  WORKER_INDEX=$((TEACHER_WORKER_OFFSET + SLURM_PROCID))
fi

case "$TEACHER_PLAN" in
  "$RUNTIME_ROOT"/*) ;;
  *) echo "TEACHER_PLAN must stay under $RUNTIME_ROOT" >&2; exit 2 ;;
esac
case "$WORKTREE" in
  "$REPO"|"$RUNTIME_ROOT/worktrees/"*) ;;
  *) echo "WORKTREE must be the repository root or stay under runtime/worktrees" >&2; exit 2 ;;
esac
if ! [[ "$TEACHER_WORKER_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "TEACHER_WORKER_COUNT must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TEACHER_WORKER_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "TEACHER_WORKER_OFFSET must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WORKER_INDEX" =~ ^[0-9]+$ ]]; then
  echo "resolved worker index must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$TEACHER_STARTUP_STAGGER_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "TEACHER_STARTUP_STAGGER_SECONDS must be a non-negative integer" >&2
  exit 2
fi
if (( WORKER_INDEX >= TEACHER_WORKER_COUNT )); then
  echo "worker index is outside TEACHER_WORKER_COUNT" >&2
  exit 2
fi

export ORCHIDEE_REPO_ROOT=$WORKTREE
export ORCHIDEE_RUNTIME_ROOT=$RUNTIME_ROOT
export ORCHIDEE_DATA_ROOT=$RUNTIME_ROOT/data
export ORCHIDEE_REFERENCE_ROOT=$RUNTIME_ROOT/assets
export ORCHIDEE_OUTPUT_ROOT=$RUNTIME_ROOT/outputs
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg
CACHE_ROOT=${TEACHER_XLA_CACHE:-$RUNTIME_ROOT/outputs/xla_cache/teacher_production}
export JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/worker-$WORKER_INDEX
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

test -x "$PYTHON"
test -f "$TEACHER_PLAN"
mkdir -p "$RUNTIME_ROOT/logs" "$JAX_COMPILATION_CACHE_DIR"
cd "$WORKTREE"
test -z "$(git status --porcelain --untracked-files=all)"
ACTUAL_GIT_HEAD=$(git rev-parse HEAD)
if [[ "$ACTUAL_GIT_HEAD" != "$TEACHER_EXPECTED_GIT_HEAD" ]]; then
  echo "Teacher worktree HEAD mismatch: $ACTUAL_GIT_HEAD" >&2
  exit 2
fi

STAGGER_SECONDS=$((SLURM_LOCALID * TEACHER_STARTUP_STAGGER_SECONDS))
echo "git_head=$ACTUAL_GIT_HEAD"
echo "job_id=$SLURM_JOB_ID proc_id=$SLURM_PROCID local_id=$SLURM_LOCALID worker_index=$WORKER_INDEX worker_count=$TEACHER_WORKER_COUNT host=$(hostname)"
echo "plan=$TEACHER_PLAN cache=$JAX_COMPILATION_CACHE_DIR stagger_seconds=$STAGGER_SECONDS"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status
if (( STAGGER_SECONDS > 0 )); then
  sleep "$STAGGER_SECONDS"
fi

COMMAND=(
  "$PYTHON" -m research.daily_coarse_graining.teacher_shards generate
  --plan "$TEACHER_PLAN"
  --worker-index "$WORKER_INDEX"
  --worker-count "$TEACHER_WORKER_COUNT"
)
if [[ -n "${TEACHER_MAX_NEW_ENTRIES:-}" ]]; then
  if ! [[ "$TEACHER_MAX_NEW_ENTRIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEACHER_MAX_NEW_ENTRIES must be a positive integer" >&2
    exit 2
  fi
  COMMAND+=(--max-new-entries "$TEACHER_MAX_NEW_ENTRIES")
fi

/usr/bin/time -v "${COMMAND[@]}"
