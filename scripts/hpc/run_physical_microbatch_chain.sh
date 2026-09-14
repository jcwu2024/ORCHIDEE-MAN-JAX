#!/bin/bash
# Sequential configurations on the same free GPU avoid simultaneous compile contention.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
GPU_DEVICE=${GPU_DEVICE:?select idle free GPU}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
cd "$CODE_ROOT"
for MICRO in 8 16; do
  env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
    bash scripts/hpc/run_physical_microbatch.sh gpu "$MICRO" "$PRINT_ONLY"
  if [[ "$PRINT_ONLY" != --dry-run ]]; then
    "$ROOT/.venvs/orcjax_cpu/bin/python" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="passed" else 1)' \
      "$ROOT/runtime/outputs/diagnostics/physical-microbatch-${MICRO}.json"
  fi
done
DIAG=$ROOT/runtime/outputs/diagnostics
COMPARE="cd $CODE_ROOT && env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 5m $ROOT/.venvs/orcjax_cpu/bin/python -m scripts.dev.compare_physical_microbatches --microbatch-8-report $DIAG/physical-microbatch-8.json --microbatch-16-report $DIAG/physical-microbatch-16.json --output $DIAG/physical-microbatch-comparison.json"
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' ssh test02 "$COMPARE"; printf '\n'
else
  ssh test02 "$COMPARE"
fi
