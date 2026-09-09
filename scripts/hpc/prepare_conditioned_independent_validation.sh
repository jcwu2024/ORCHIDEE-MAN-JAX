#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
case "$(hostname -s)" in test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;; *) exit 2 ;; esac
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg PYTHONPATH=$WORKTREE
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs
cd "$WORKTREE"
exec timeout --signal=TERM 60m "$ROOT/.venvs/orcjax_cpu/bin/python" -m scripts.hpc.prepare_conditioned_independent_validation \
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
  --training-contract "$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json" \
  --continuation-contract "$WORKTREE/manifests/coarse_graining/conditioned_snow_free_continuation_v2.json" \
  --output-dir "$ROOT/runtime/outputs/training/conditioned-independent-validation-20260909"
