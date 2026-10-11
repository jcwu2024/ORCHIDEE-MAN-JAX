#!/bin/bash
# One authorized saved-array CPU explanation; never a model or scheduler entry.
set -euo pipefail
ssh -T -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 test03 bash -s <<'CPU'
set -euo pipefail
umask 022
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
TRANSFER="$PROJECT/runtime/transfer/gpu-lifecycle-20261011-3b4ae35"
CODE="$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1"
ORIGINAL="$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-15769234/phased_resident_control/capacity/B1024"
PARENT="$PROJECT/runtime/outputs/research/global_throughput/server-benchmark"
OUT="$PARENT/gpu-lifecycle-20261011-15769234-saved-array-explanation"
SCRIPT="$TRANSFER/explain_saved_failure.py"
LAYOUT="$TRANSFER/state-layout.json"
PYTHON="$PROJECT/.venvs/orcjax_cpu/bin/python"
test "$(hostname)" = ibtest03
for EXISTING in "$TRANSFER" "$CODE" "$ORIGINAL" "$PARENT"; do
  RESOLVED=$(realpath -e -- "$EXISTING")
  case "$RESOLVED" in /WORK/liwei_work/jcwu/*) ;; *) exit 91 ;; esac
done
test -x "$PYTHON"
cd "$CODE"
test "$(git rev-parse HEAD)" = 3b4ae35e695f85480ecb72461e4ba93f4534bbd8
test -z "$(git status --porcelain --untracked-files=all)"
printf '%s  %s\n' \
  a4d14fe02c7718e48ac9a7acd6ed6c81b84841013f4c7a40eb91d8739ec9ec80 "$SCRIPT" \
  8fcdaa71441ac5e9c193f54801bd2e583290ff544c90220aab620e5c2f944a89 "$LAYOUT" | sha256sum -c -
test ! -e "$OUT"
mkdir -- "$OUT"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -v 8388608
ulimit -c 0
CPU_COMMAND=(
  timeout --kill-after=10s 600s taskset -c 0 "$PYTHON" -B "$SCRIPT"
  --code-root "$CODE"
  --failed-input "$ORIGINAL/failed_initial_arrays.pkl"
  --reference-dir "$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model"
  --report "$ORIGINAL/report.json"
  --state-layout-json "$LAYOUT"
  --output-dir "$OUT/comparison"
  --lane-chunk 64 --budget-seconds 540
)
{
  hostname
  date -u +%FT%TZ
  uptime
  taskset -pc $$
  free -m
  printf 'virtual_memory_limit_KiB=%s\n' "$(ulimit -v)"
  printf 'selected_CPU=0\nouter_wall_limit_seconds=600\ninner_budget_seconds=540\n'
  printf 'PYTHONDONTWRITEBYTECODE=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$PYTHONDONTWRITEBYTECODE" "$CUDA_VISIBLE_DEVICES"
  printf 'OPENBLAS_NUM_THREADS=%s\nOMP_NUM_THREADS=%s\nMKL_NUM_THREADS=%s\nNUMEXPR_NUM_THREADS=%s\n' "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS"
  printf 'execution_SHA=%s\n' "$(git rev-parse HEAD)"
  printf 'code_checkout_porcelain_begin\n'
  git status --porcelain --untracked-files=all
  printf 'code_checkout_porcelain_end\n'
} > "$OUT/environment.txt"
printf '%q ' /usr/bin/time -v -o "$OUT/gnu-time.txt" "${CPU_COMMAND[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
date -u +%FT%TZ > "$OUT/comparison-start-utc.txt"
set +e
/usr/bin/time -v -o "$OUT/gnu-time.txt" "${CPU_COMMAND[@]}" > "$OUT/cpu.stdout.txt" 2> "$OUT/cpu.stderr.txt"
STATUS=$?
set -e
date -u +%FT%TZ > "$OUT/comparison-end-utc.txt"
printf '%s\n' "$STATUS" > "$OUT/exit-status.txt"
test -z "$(git status --porcelain --untracked-files=all)"
find "$OUT" -type f -printf '%s\t%P\n' | sort > "$TRANSFER/saved-array-explanation-inventory.tsv"
find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$TRANSFER/saved-array-explanation-sha256.txt"
cat "$OUT/environment.txt" "$OUT/command.txt" "$OUT/exit-status.txt" "$OUT/cpu.stdout.txt" "$OUT/cpu.stderr.txt" "$OUT/gnu-time.txt"
printf 'explanation_output=%s\n' "$OUT"
exit "$STATUS"
CPU
