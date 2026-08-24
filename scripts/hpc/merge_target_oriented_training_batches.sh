#!/bin/bash
# Merge disjoint target-oriented preparation chunks on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
UPDATES=${UPDATES:-250}
EXPECTED_CHUNKS=${EXPECTED_CHUNKS:-8}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-prepared-${UPDATES}-${HEAD_SHORT}}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

case "$(hostname)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "target-oriented preparation merge requires a free CPU test node" >&2; exit 2 ;;
esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT_DIR" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR/prepared_manifest.json"

mapfile -t CHUNKS < <(find "$OUTPUT_DIR/chunks" -mindepth 2 -maxdepth 2 -name prepared_manifest.json -type f | sort)
test "${#CHUNKS[@]}" = "$EXPECTED_CHUNKS"

export PYTHONPATH=$WORKTREE
cd "$WORKTREE"
"$PYTHON" -m scripts.hpc.merge_target_oriented_training_batches \
  --updates "$UPDATES" \
  --output-dir "$OUTPUT_DIR" \
  "${CHUNKS[@]}"

sha256sum "$OUTPUT_DIR/prepared_manifest.json"
