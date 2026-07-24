#!/bin/bash
# Losslessly append fast-owned leaf_ci from stored S[d+1]; no Teacher rerun.
#SBATCH -J orcjax-v5-migrate
#SBATCH -p cnmix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/markov_v5_migrate_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/markov_v5_migrate_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT/runtime/worktrees/neural-provisional-141b2f3}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
SOURCE_MANIFEST=${SOURCE_MANIFEST:?set SOURCE_MANIFEST to the v4 dataset manifest}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new v5 dataset directory}
DATASET_ID=${DATASET_ID:?set DATASET_ID to the v5 dataset identity}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved code snapshot}

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
case "$SOURCE_MANIFEST" in
  "$ROOT/runtime/"*) ;;
  *) echo "SOURCE_MANIFEST must stay under $ROOT/runtime" >&2; exit 2 ;;
esac
case "$OUTPUT_ROOT" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_ROOT must stay under $ROOT/runtime" >&2; exit 2 ;;
esac
if [[ "$SOURCE_MANIFEST" == "$OUTPUT_ROOT"/* ]]; then
  echo "v5 output must not contain the v4 source manifest" >&2
  exit 2
fi

test -x "$PYTHON"
test -f "$SOURCE_MANIFEST"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
mkdir -p "$OUTPUT_ROOT" "$ROOT/runtime/logs" "$ROOT/runtime/cache/xdg"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg

cd "$WORKTREE"
"$PYTHON" -m research.daily_coarse_graining.migrate_markov_v4_to_v5 \
  --source-manifest "$SOURCE_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --dataset-id "$DATASET_ID"

test -f "$OUTPUT_ROOT/dataset_manifest.json"
