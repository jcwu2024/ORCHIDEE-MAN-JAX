#!/bin/bash
# Requires explicit additional480s free-GPU approval; --dry-run is CPU-only.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/shared-bank-adjoint-accumulation-20261002"
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$OVERLAY/run_shared_bank_saved_binary_T7.py" "$@"
