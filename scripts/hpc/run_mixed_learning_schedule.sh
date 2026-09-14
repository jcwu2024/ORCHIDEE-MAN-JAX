#!/bin/bash
# Free CPU metadata preparation; reads existing indexes, not Teacher payloads.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
if [[ "$PRINT_ONLY" != --dry-run ]]; then
  case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
fi
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
COMMAND=(timeout --signal=TERM 15m "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.prepare_mixed_learning_schedule --updates 200
  --catalogue "$ROOT/runtime/outputs/training/one-day-expansion-15061225/prepared_one_day_catalogue.json.gz"
  --development-selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --output "$ROOT/runtime/outputs/diagnostics/mixed-learning-200-recurrent-schedule-v1.json")
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"; printf '\n'
else
  "${COMMAND[@]}"
fi
