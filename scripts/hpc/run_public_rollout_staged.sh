#!/bin/bash
# Bounded free-node AD completion; preserve stages and stop on unmet gates.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:-$ROOT}
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
BASE=$ROOT/runtime/outputs/diagnostics/public-rollout-T2-v2
case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2;; esac
cd "$CODE_ROOT"
env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
  OUTPUT="$BASE-runtime.json" bash scripts/hpc/run_fresh_start_one_day.sh --backend gpu
for STAGE in core derivatives checkpointed; do
  STAGE_TIMEOUT=90m
  if [[ "$STAGE" == derivatives ]]; then STAGE_TIMEOUT=60m; fi
  echo "begin_stage=$STAGE limit=$STAGE_TIMEOUT"
  env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
    PYTHON_MODULE=scripts.hpc.probe_public_rollout_staged RUN_TIMEOUT="$STAGE_TIMEOUT" \
    OUTPUT="$BASE-$STAGE.json" \
    bash scripts/hpc/run_fresh_start_one_day.sh \
      --protocol manifests/coarse_graining/fresh_start_one_day_probe_v3.json \
      --stage "$STAGE" --stage-directory "$BASE-stages" --horizon 2
  "$ROOT/.venvs/orcjax_cpu/bin/python" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="passed" and r.get("passed") else 1)' \
    "$BASE-$STAGE.json"
  echo "passed_stage=$STAGE"
done
echo completed_all_stages
