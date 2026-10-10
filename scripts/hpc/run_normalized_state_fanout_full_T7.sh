#!/bin/bash
# PREPARATION ONLY until user authorizes a new finite free GPU1 window.
set -euo pipefail
export NORMALIZED_FANOUT_ORIGIN_UTC_SECONDS="${NORMALIZED_FANOUT_ORIGIN_UTC_SECONDS:-$(date +%s.%N)}"
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/normalized-state-fanout-full-T7-20261003"
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$OVERLAY/run_normalized_state_fanout_full_T7.py" "$@"
