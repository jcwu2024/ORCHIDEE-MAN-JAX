#!/bin/bash
# Preparation is --dry-run; actual free execution requires precision approval.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/shared-bank-adjoint-accumulation-20261002"
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$OVERLAY/run_shared_bank_adjoint_T7.py" "$@"
