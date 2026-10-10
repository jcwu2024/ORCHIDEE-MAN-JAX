#!/bin/bash
# Proposed new 360s free-GPU window. --dry-run never initializes GPU/JAX.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/normalized-state-fanout-component-20261003"
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$OVERLAY/run_normalized_state_fanout_component.py" "$@"
