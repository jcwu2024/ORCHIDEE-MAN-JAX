#!/bin/bash
# Bounded data/shape checks on a free CPU test node; never train or run Teacher.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the diagnostic checkout}
MODE=${1:?shape or labels}
PRINT_ONLY=${2:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" PYTHONDONTWRITEBYTECODE=1
DIAG=$ROOT/runtime/outputs/diagnostics
case "$MODE" in
  shape)
    MODULE=scripts.hpc.diagnose_trained_gpp_state_interventions
    ARGS=(--shape-only
      --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
      --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
      --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
      --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
      --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
      --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
      --selection "$DIAG/trained-forward-selection-v1.json"
      --training-report "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.json"
      --checkpoint "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.checkpoint.pkl"
      --attribution-report "$DIAG/trained-forward-attribution-v1.json"
      --archive-directory "$DIAG"
      --output "$DIAG/trained-gpp-state-interventions-v1-cpu-shape.json")
    ;;
  labels)
    MODULE=scripts.dev.audit_trained_canopy_inventory
    ARGS=(--report "$DIAG/trained-forward-attribution-v1.json"
      --archive-dir "$DIAG"
      --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.npz"
      --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.npz"
      --prepared-manifest "$ROOT/runtime/outputs/training/conditioned-coverage-expanded-14992138/prepared_manifest.json"
      --output "$DIAG/trained-canopy-inventory-label-audit-v1.json")
    ;;
  *) exit 2;;
esac
CMD=(timeout --signal=TERM 15m "$ROOT/.venvs/orcjax_cpu/bin/python" -m "$MODULE" "${ARGS[@]}")
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
  "${CMD[@]}"
fi
