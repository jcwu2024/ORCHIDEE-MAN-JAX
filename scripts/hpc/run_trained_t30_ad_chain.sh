#!/bin/bash
# Two bounded free-GPU stages; the saved reverse survives a later JVP timeout.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:-$ROOT}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
cd "$CODE_ROOT"
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  bash "$CODE_ROOT/scripts/hpc/run_trained_evaluation.sh" ad-t30-reverse --dry-run
  bash "$CODE_ROOT/scripts/hpc/run_trained_evaluation.sh" ad-t30-verify --dry-run
  exit 0
fi
trap 'CHAIN_STATUS=$?; echo "t30_ad_chain_exit_status=$CHAIN_STATUS"' EXIT
bash "$CODE_ROOT/scripts/hpc/run_trained_evaluation.sh" ad-t30-reverse
"$ROOT/.venvs/orcjax_cpu/bin/python" -c 'import json,sys; report=json.load(open(sys.argv[1])); sys.exit(0 if report.get("status")=="passed" else 3)' \
  "$ROOT/runtime/outputs/diagnostics/trained-forward-ad-v1-ad-t30-reverse.json"
bash "$CODE_ROOT/scripts/hpc/run_trained_evaluation.sh" ad-t30-verify
