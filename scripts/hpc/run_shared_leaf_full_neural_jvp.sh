#!/bin/bash
# CPU-only independent public-loss directional derivative; never a login node.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
[[ -z ${SLURM_JOB_ID:-} ]] || exit 2
[[ $(hostname -s) == ibtest03 ]] || exit 2
[[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
COMMAND=(taskset -c 0-7 env -u LD_LIBRARY_PATH
  SINGULARITYENV_JAX_PLATFORMS=cpu SINGULARITYENV_CUDA_VISIBLE_DEVICES=
  SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python" -B
  "$OVERLAY/probe_shared_leaf_full_neural_jvp.py"
  --manifest "$OVERLAY/original-full-state-schedule-resume.manifest.json"
  --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
  --old-snapshot "$DIAG/shared-original-full-state-schedule-20261005.old-diagnostic.pkl"
  --new-snapshot "$DIAG/shared-original-full-state-schedule-20261005.new-diagnostic.pkl"
  --field-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
  --backbone-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --turnover-overlay "$OVERLAY/turnover_leaf_age_fall.guarded-source.py"
  --domains old:16 old:17 new:16 new:17 --neural-execution component_jit --wall-seconds 3600
  --output "$DIAG/leaf-full-neural-cpu-jvp-20261005.json")
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
[[ $# == 0 ]] || exit 2
date -Is
uptime
exec "${COMMAND[@]}"
