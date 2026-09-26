#!/bin/bash
# Free-node admission of complete unchanged-work updates with saved binaries.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-combined-full-update-20260926"}
INSPECT_ONLY=${INSPECT_ONLY:-0}
DRY_ARGS=()
if [[ ${1:-} == --dry-run ]]; then DRY_ARGS+=(--dry-run); shift; fi
case $INSPECT_ONLY in
  0)
    case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) [[ ${#DRY_ARGS[@]} == 1 ]] || exit 2 ;; esac
    RUN_TIMEOUT=30m
    DEFAULT_OUTPUT="$ROOT/runtime/outputs/diagnostics/shared-combined-full-update-20260926.json"
    ;;
  1)
    case $(hostname -s) in test0[1-4]|ibtest0[1-4]) ;; *) [[ ${#DRY_ARGS[@]} == 1 ]] || exit 2 ;; esac
    RUN_TIMEOUT=5m
    DEFAULT_OUTPUT="$ROOT/runtime/outputs/diagnostics/shared-combined-full-update-20260926.inspect.json"
    ;;
  *) exit 2 ;;
esac
export INSPECT_ONLY RUN_TIMEOUT
export ENTRY_FILE="$OVERLAY/probe_shared_serialized_dual.py"
export INPUT_BUNDLE="$ROOT/runtime/outputs/diagnostics/shared-split-dual-inputs-c599236.pkl"
export OUTPUT=${OUTPUT:-"$DEFAULT_OUTPUT"}
export GPU_DEVICE=0 WORKER_GPU_DEVICE=1 PREPARE_INPUT_BUNDLE=0
exec bash "$OVERLAY/run_shared_serialized_dual.sh" ${DRY_ARGS[@]+"${DRY_ARGS[@]}"} \
  --daily-admission-report "$ROOT/runtime/outputs/diagnostics/shared-combined-real-daily-20260925.json" \
  --seven-admission-report "$ROOT/runtime/outputs/diagnostics/shared-combined-recurrent-T7-20260926.json" \
  --thirty-admission-report "$ROOT/runtime/outputs/diagnostics/shared-combined-recurrent-T30-20260926.json" \
  --daily-counts 32 "$@"
