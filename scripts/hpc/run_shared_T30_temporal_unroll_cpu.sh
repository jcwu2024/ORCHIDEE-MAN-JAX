#!/bin/bash
set -uo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY=$ROOT/runtime/transfer/shared-T30-temporal-unroll-cpu-20261004
CSE_OVERLAY=$ROOT/runtime/transfer/shared-command-buffer-loops-20261001
ENV=$ROOT/.venvs/orcjax_cpu
OUT=$ROOT/runtime/outputs/diagnostics/shared-T30-temporal-unroll-cpu-20261004
PYTHON=$OVERLAY/probe_shared_T30_temporal_unroll_cpu.py
ROLLOUT_SOURCE=$OVERLAY/temporal_rollout_source
PAYLOAD=$ROOT/runtime/outputs/diagnostics/shared-normalized-keys-payload-20261001.pkl
WINDOW=$OUT/prescreen-window.tsv
WRAPPER_RESERVE_SECONDS=3
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" && "$#" -eq 1 ]]; then
  DRY_RUN=1
elif [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
invocation_start_ns=$(date +%s%N)

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUT"
  for file in "$OUT"/prescreen-attempt*.json; do
    [[ -e "$file" ]] || continue
    if grep -Eq '"status"[[:space:]]*:[[:space:]]*"completed"' "$file"; then
      echo "completed prescreen evidence already exists; preserving it" >&2
      exit 2
    fi
  done
fi

if [[ -f "$WINDOW" ]]; then
  IFS=$'\t' read -r window_started_ns window_deadline_ns < "$WINDOW"
else
  window_started_ns=$invocation_start_ns
  window_deadline_ns=$((window_started_ns + 120000000000))
  if [[ "$DRY_RUN" -eq 0 ]]; then
    printf '%s\t%s\n' "$window_started_ns" "$window_deadline_ns" > "$WINDOW"
  fi
fi

next_attempt=1
for file in "$OUT"/prescreen-attempt*.json "$OUT"/actual-exit-attempt*.json; do
  [[ -e "$file" ]] || continue
  if [[ "$file" =~ attempt([0-9]+)\.json$ ]] && (( BASH_REMATCH[1] >= next_attempt )); then
    next_attempt=$((BASH_REMATCH[1] + 1))
  fi
done
REPORT=$OUT/prescreen-attempt${next_attempt}.json
EXIT_RECORD=$OUT/actual-exit-attempt${next_attempt}.json
if [[ -e "$REPORT" || -e "$EXIT_RECORD" ]]; then
  echo "next attempt output already exists; preserving it" >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]] && [[ ! -x "$ENV/bin/python" || ! -f "$PYTHON" || ! -f "$PAYLOAD" || ! -f "$CSE_OVERLAY/probe_shared_command_buffer_loops.py" || ! -f "$ROLLOUT_SOURCE/rollout_loss.py" || ! -f "$ROLLOUT_SOURCE/streamed_adjoint.py" || ! -f "$ROLLOUT_SOURCE/shared_transformer.py" || ! -f "$ROLLOUT_SOURCE/strong_baseline_architecture.py" ]]; then
  echo "canonical runtime, overlay or saved expert5 payload is missing" >&2
  exit 2
fi

remaining_seconds() {
  local now_ns remaining_ns whole_seconds
  now_ns=$(date +%s%N)
  remaining_ns=$((window_deadline_ns - now_ns))
  whole_seconds=$((remaining_ns / 1000000000 - WRAPPER_RESERVE_SECONDS))
  printf '%s\n' "$whole_seconds"
}

remaining=$(remaining_seconds)
if (( remaining < 1 )); then
  echo "the persistent 120-second prescreen window is exhausted; preserving prior evidence" >&2
  exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=1s "${remaining}s" env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$OVERLAY:$CSE_OVERLAY:$ROOT" "$ENV/bin/python" -B "$PYTHON" --payload "$PAYLOAD" --rollout-source-dir "$ROLLOUT_SOURCE" --output "$REPORT" --budget-seconds "$remaining")
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'node=test01 attempt=%s budget_seconds=%s reserved_seconds=%s persistent_window=%s output=%s command=' "$next_attempt" "$remaining" "$WRAPPER_RESERVE_SECONDS" "$WINDOW" "$OUT"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

printf '{"status":"launching","started_unix_seconds":%s,"host":"%s","node":"test01","budget_seconds":%s,"window_started_unix_ns":%s,"window_deadline_unix_ns":%s,"attempt":%s}\n' \
  "$(( $(date +%s%N) / 1000000000 ))" "$(hostname)" "$remaining" "$window_started_ns" "$window_deadline_ns" "$next_attempt" \
  > "$OUT/launcher-start-attempt${next_attempt}.json"
# Recompute just before launch so wrapper setup time is deducted from the child limit.
remaining=$(remaining_seconds)
if (( remaining < 1 )); then
  echo "the persistent 120-second prescreen window is exhausted before launch" >&2
  exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=1s "${remaining}s" env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$OVERLAY:$CSE_OVERLAY:$ROOT" "$ENV/bin/python" -B "$PYTHON" --payload "$PAYLOAD" --rollout-source-dir "$ROLLOUT_SOURCE" --output "$REPORT" --budget-seconds "$remaining")
set +e
"${COMMAND[@]}"
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then status=completed; elif [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then status=timeout; else status=failed; fi
printf '{"status":"%s","exit_code":%s,"finished_unix_seconds":%s,"host":"%s","node":"test01","budget_seconds":%s,"window_started_unix_ns":%s,"window_deadline_unix_ns":%s,"attempt":%s}\n' \
  "$status" "$rc" "$(( $(date +%s%N) / 1000000000 ))" "$(hostname)" "$remaining" "$window_started_ns" "$window_deadline_ns" "$next_attempt" \
  > "$EXIT_RECORD"
exit "$rc"
