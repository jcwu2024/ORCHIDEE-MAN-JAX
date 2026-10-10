#!/bin/bash
set -euo pipefail
export V4_LEARNING_SENSITIVITY_ORIGIN_UTC_SECONDS="${V4_LEARNING_SENSITIVITY_ORIGIN_UTC_SECONDS:-$(date +%s.%N)}"
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$ROOT/runtime/transfer/shared-v4-learning-sensitivity-20261004/run_shared_v4_learning_sensitivity.py" "$@"
