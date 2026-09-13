#!/bin/bash
# Actual-data shape tracing of the existing pool runner; GPU evidence is reused.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the reviewed training checkout}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
if [[ "$PRINT_ONLY" != --dry-run ]]; then
  case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
fi
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
COMMAND=(timeout --signal=TERM 15m "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.run_fresh_start_one_day_pool --shape-only
  --pool-protocol "$CODE_ROOT/manifests/coarse_graining/fresh_start_feedback_pool_v1.json"
  --catalogue-report "$ROOT/runtime/outputs/diagnostics/one-day-catalogue-v1.json"
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
  --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
  --output "$ROOT/runtime/outputs/diagnostics/feedback-pool-v1-cpu-shape.json")
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  "${COMMAND[@]}"
fi
