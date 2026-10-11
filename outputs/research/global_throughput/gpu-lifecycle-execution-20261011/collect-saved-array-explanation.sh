#!/bin/bash
# Transfer metadata and scalar differences only; no scientific-array loading.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
TRANSFER="$PROJECT/runtime/transfer/gpu-lifecycle-20261011-3b4ae35"
OUT="$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-15769234-saved-array-explanation"
ARCHIVE="$TRANSFER/saved-array-explanation-15769234.tar.gz"
for DIRECTORY in "$TRANSFER" "$OUT"; do
  RESOLVED=$(realpath -e -- "$DIRECTORY")
  case "$RESOLVED" in /WORK/liwei_work/jcwu/*) ;; *) exit 91 ;; esac
done
test "$(cat "$OUT/exit-status.txt")" = 0
test -f "$OUT/comparison/summary.json"
test -f "$OUT/comparison/all-failing-coordinates.jsonl.gz"
test ! -e "$ARCHIVE"
test -z "$(find "$OUT" -type f \( -name '*.pkl' -o -name '*.npy' -o -name '*.npz' \) -print)"
tar -czf "$ARCHIVE" -C "$OUT" . -C "$TRANSFER" saved-array-explanation-inventory.tsv saved-array-explanation-sha256.txt
hostname
date -u +%FT%TZ
stat -c '%s %n' "$ARCHIVE"
sha256sum "$ARCHIVE"
cat "$TRANSFER/saved-array-explanation-inventory.tsv"
