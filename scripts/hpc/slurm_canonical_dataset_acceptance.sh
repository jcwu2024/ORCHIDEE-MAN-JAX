#!/bin/bash
# Submit only after teacher_shards aggregate has produced dataset_manifest.json.
#SBATCH -J orcjax-accept
#SBATCH -p cnmix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_acceptance_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_acceptance_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved code snapshot}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
DATASET=${DATASET:?set DATASET to the complete dataset_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to an isolated acceptance directory}

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
case "$DATASET" in
  "$ROOT/runtime/"*) ;;
  *) echo "DATASET must stay under $ROOT/runtime" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_DIR must stay under $ROOT/runtime" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test -f "$DATASET"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$ROOT/runtime/logs" "$ROOT/runtime/cache/xdg"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg

cd "$WORKTREE"
"$PYTHON" -m research.daily_coarse_graining.canonical_training_run \
  accept-dataset \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --chunk-rows 64 \
  --batch-size 8 \
  --seed 20260724

test -f "$OUTPUT_DIR/training_statistics.json"
test -f "$OUTPUT_DIR/training_statistics.npz"
test -f "$OUTPUT_DIR/acceptance_report.json"
