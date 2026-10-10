#!/bin/bash
# One bounded free GPU1 capture; parent method confirmation precedes execution.
set -euo pipefail
export PUBLIC_COTANGENT_ORIGIN_UTC_SECONDS="${PUBLIC_COTANGENT_ORIGIN_UTC_SECONDS:-$(date +%s.%N)}"
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/public-state-summary-cotangents-T7-20261003"
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$OVERLAY/run_public_state_summary_cotangents_T7.py" "$@"
