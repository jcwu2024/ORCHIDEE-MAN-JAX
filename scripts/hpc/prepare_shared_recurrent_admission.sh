#!/bin/bash
# Free CPU preparation only; does not launch GPU tests or optimizer updates.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set immutable checkout}
OUTPUT=${OUTPUT:?set report JSON under runtime/outputs}
RUN_TIMEOUT=${RUN_TIMEOUT:-20m}
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT"/runtime/outputs/*.json) ;; *) exit 2 ;; esac
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
COMMAND=(env
  ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
  ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
  ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" PYTHONPATH="$CODE_ROOT"
  JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
  timeout --signal=TERM "$RUN_TIMEOUT" "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.prepare_shared_recurrent_admission
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
  --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
  --calibration-report "$ROOT/runtime/outputs/diagnostics/shared-persistent-158192f-forward.json"
  --protocol "$CODE_ROOT/manifests/coarse_graining/shared_recurrent_admission_v1.json"
  --output "$OUTPUT" "$@")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
mkdir -p "$(dirname "$OUTPUT")"
cd "$CODE_ROOT"
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
echo "host=$(hostname -s) code=$(git rev-parse --short HEAD) output=$OUTPUT"
"${COMMAND[@]}"
