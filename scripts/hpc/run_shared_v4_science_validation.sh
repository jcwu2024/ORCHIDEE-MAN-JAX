#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
ENTRY="$ROOT/runtime/transfer/shared-v4-heldout-science-20261004/run_shared_v4_science_validation.py"
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$ENTRY" "$@"
