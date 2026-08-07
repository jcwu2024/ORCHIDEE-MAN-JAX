#!/bin/bash
# Finalize only after every worker in the frozen 669-point plan has exited.
# Resource directives are conservative defaults; no job is submitted by this file.
#SBATCH -J orcjax-teacher-final
#SBATCH -p cnmix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_finalize_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_finalize_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$ROOT/runtime
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved research snapshot}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
TEACHER_PLAN=${TEACHER_PLAN:?set TEACHER_PLAN to the frozen generation plan}
TEACHER_WORKER_COUNT=${TEACHER_WORKER_COUNT:?set TEACHER_WORKER_COUNT}
DATASET=${DATASET:?set DATASET to the plan output dataset_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new finalization directory}
POLICY=${POLICY:-$WORKTREE/manifests/coarse_graining/daily_teacher_669_data_product_policy_v2.json}
FINALIZE_FROM_AGGREGATE=${FINALIZE_FROM_AGGREGATE:-0}

case "$WORKTREE" in
  "$ROOT") ;;
  "$RUNTIME_ROOT/worktrees/"*) ;;
  *) echo "WORKTREE must be the repository root or stay under $RUNTIME_ROOT/worktrees" >&2; exit 2 ;;
esac
for path in "$TEACHER_PLAN" "$DATASET" "$OUTPUT_DIR"; do
  case "$path" in
    "$RUNTIME_ROOT/"*) ;;
    *) echo "runtime inputs and outputs must stay under $RUNTIME_ROOT" >&2; exit 2 ;;
  esac
done
if ! [[ "$TEACHER_WORKER_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "TEACHER_WORKER_COUNT must be a positive integer" >&2
  exit 2
fi
if [[ "$FINALIZE_FROM_AGGREGATE" != 0 && "$FINALIZE_FROM_AGGREGATE" != 1 ]]; then
  echo "FINALIZE_FROM_AGGREGATE must be 0 or 1" >&2
  exit 2
fi

test -x "$PYTHON"
test -f "$TEACHER_PLAN"
test -f "$POLICY"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/cache/xdg"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg

cd "$WORKTREE"
"$PYTHON" -m research.daily_coarse_graining.teacher_shards progress \
  --plan "$TEACHER_PLAN" \
  --worker-count "$TEACHER_WORKER_COUNT" \
  --require-complete \
  --report "$OUTPUT_DIR/worker_progress.json"

if [[ "$FINALIZE_FROM_AGGREGATE" == 1 ]]; then
  : "${EXPECTED_DATASET_MANIFEST_SHA256:?required when resuming after aggregation}"
  test -f "$DATASET"
  OBSERVED_DATASET_MANIFEST_SHA256=$(sha256sum "$DATASET" | awk '{print $1}')
  if [[ "$OBSERVED_DATASET_MANIFEST_SHA256" != "$EXPECTED_DATASET_MANIFEST_SHA256" ]]; then
    echo "dataset manifest hash mismatch while resuming finalization" >&2
    exit 2
  fi
else
  "$PYTHON" -m research.daily_coarse_graining.teacher_shards aggregate \
    --plan "$TEACHER_PLAN" \
    --worker-count "$TEACHER_WORKER_COUNT"
fi
test -f "$DATASET"

"$PYTHON" -m research.daily_coarse_graining.teacher_data_product_admission \
  --policy "$POLICY" \
  --contract-manifest "$DATASET" \
  --require-production-dataset \
  --output "$OUTPUT_DIR/production_admission.json"

"$PYTHON" -m research.daily_coarse_graining.canonical_training_run \
  accept-dataset \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --chunk-rows 64 \
  --batch-size 8 \
  --seed 20260727

test -f "$OUTPUT_DIR/worker_progress.json"
test -f "$OUTPUT_DIR/production_admission.json"
test -f "$OUTPUT_DIR/training_statistics.json"
test -f "$OUTPUT_DIR/training_statistics.npz"
test -f "$OUTPUT_DIR/acceptance_report.json"
