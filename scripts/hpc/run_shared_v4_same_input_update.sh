#!/bin/bash
set -euo pipefail
export V4_SAME_INPUT_ORIGIN_UTC_SECONDS="${V4_SAME_INPUT_ORIGIN_UTC_SECONDS:-$(date +%s.%N)}"
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$ROOT/runtime/transfer/shared-v4-same-input-update-20261003/run_shared_v4_same_input_update.py" "$@"
