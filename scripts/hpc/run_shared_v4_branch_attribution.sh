#!/bin/bash
set -euo pipefail
export V4_BRANCH_ORIGIN_UTC_SECONDS="${V4_BRANCH_ORIGIN_UTC_SECONDS:-$(date +%s.%N)}"
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
exec "$ROOT/.venvs/orcjax_cpu/bin/python" -B "$ROOT/runtime/transfer/shared-v4-branch-attribution-20261003/run_shared_v4_branch_attribution.py" "$@"
