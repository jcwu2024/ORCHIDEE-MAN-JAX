#!/bin/bash
# Submit only after reviewing the exact paths and cost, for example:
# First submit array index 0, then indices 1-5 only after task 0 passes.
# Required exports: GATE_E2_PARENT_MANIFEST, GATE_E2_TEACHER_PLAN, and
# GATE_E2_EXPECTED_GIT_HEAD.
#SBATCH --job-name=orcjax-e2-sidecar
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gate_e2_sidecar_%A_%a.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gate_e2_sidecar_%A_%a.txt
#SBATCH --no-requeue

set -euo pipefail

REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$REPO/runtime
WORKTREE=${WORKTREE:-$REPO}
PYTHON=$REPO/.venvs/orcjax_cpu/bin/python
PILOT_PLAN=$WORKTREE/manifests/coarse_graining/gate_e2_typed_sidecar_pilot_v1.json
OUTPUT_ROOT=$RUNTIME_ROOT/outputs/data/gate-e2-typed-sidecar-v1-pilot
: "${GATE_E2_PARENT_MANIFEST:?GATE_E2_PARENT_MANIFEST must be set}"
: "${GATE_E2_TEACHER_PLAN:?GATE_E2_TEACHER_PLAN must be set}"
: "${GATE_E2_EXPECTED_GIT_HEAD:?GATE_E2_EXPECTED_GIT_HEAD must be set}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

for path in "$GATE_E2_PARENT_MANIFEST" "$GATE_E2_TEACHER_PLAN" "$OUTPUT_ROOT"; do
  case "$path" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Gate-E2 runtime paths must stay under $RUNTIME_ROOT: $path" >&2; exit 2 ;;
  esac
done
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "Gate-E2 pilot array index must be 0 through 5" >&2
  exit 2
fi

export ORCHIDEE_REPO_ROOT=$WORKTREE
export ORCHIDEE_RUNTIME_ROOT=$RUNTIME_ROOT
export ORCHIDEE_DATA_ROOT=$RUNTIME_ROOT/data
export ORCHIDEE_REFERENCE_ROOT=$RUNTIME_ROOT/assets
export ORCHIDEE_OUTPUT_ROOT=$RUNTIME_ROOT/outputs
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg
export JAX_COMPILATION_CACHE_DIR=$RUNTIME_ROOT/outputs/xla_cache/gate_e2_sidecar/worker-$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

test -x "$PYTHON"
test -f "$PILOT_PLAN"
test -f "$GATE_E2_PARENT_MANIFEST"
test -f "$GATE_E2_TEACHER_PLAN"
mkdir -p "$RUNTIME_ROOT/logs" "$OUTPUT_ROOT" "$JAX_COMPILATION_CACHE_DIR"
cd "$WORKTREE"
test -z "$(git status --porcelain --untracked-files=all)"
ACTUAL_GIT_HEAD=$(git rev-parse HEAD)
if [[ "$ACTUAL_GIT_HEAD" != "$GATE_E2_EXPECTED_GIT_HEAD" ]]; then
  echo "Gate-E2 worktree HEAD mismatch: $ACTUAL_GIT_HEAD" >&2
  exit 2
fi

echo "git_head=$ACTUAL_GIT_HEAD job_id=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "parent=$GATE_E2_PARENT_MANIFEST teacher_plan=$GATE_E2_TEACHER_PLAN output=$OUTPUT_ROOT"
echo "diagnostic_max_days=${GATE_E2_DIAGNOSTIC_MAX_DAYS:-full_generation}"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

"$PYTHON" -m research.daily_coarse_graining.typed_sidecar_production preflight \
  --pilot-plan "$PILOT_PLAN" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --parent-manifest "$GATE_E2_PARENT_MANIFEST" \
  --teacher-plan "$GATE_E2_TEACHER_PLAN" \
  --expected-git-head "$GATE_E2_EXPECTED_GIT_HEAD"

MODE=generate
EXTRA_ARGS=()
if [[ -n "${GATE_E2_DIAGNOSTIC_MAX_DAYS:-}" ]]; then
  MODE=diagnose-state
  EXTRA_ARGS=(--max-days "$GATE_E2_DIAGNOSTIC_MAX_DAYS")
fi
/usr/bin/time -v "$PYTHON" -m research.daily_coarse_graining.typed_sidecar_production "$MODE" \
  --pilot-plan "$PILOT_PLAN" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --parent-manifest "$GATE_E2_PARENT_MANIFEST" \
  --teacher-plan "$GATE_E2_TEACHER_PLAN" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"
