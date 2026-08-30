#!/bin/bash
# Audit the target-blind broad-GPP schedule on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
OUTPUT=${OUTPUT:?set OUTPUT below the project runtime diagnostics tree}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

case "$(hostname -s)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "broad-GPP schedule audit requires a free CPU test node" >&2; exit 2 ;;
esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT/runtime/outputs/diagnostics/"*.json) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/runtime/cache/xdg"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs PYTHONPATH=$WORKTREE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$WORKTREE"

"$PYTHON" -m scripts.hpc.audit_target_oriented_structured_gpp_broad_schedule \
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
  --training-contract "$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json" \
  --warm-start-contract "$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_warm_start_v1.json" \
  --joint-training-contract "$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_joint_training_v1.json" \
  --broad-preflight-contract "$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_broad_preflight_v1.json" \
  --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json" \
  --coverage-report "$ROOT/runtime/outputs/diagnostics/target-oriented-structured-gpp-data-coverage-db9763b.json" \
  --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"
