#!/bin/bash
# Free-node CPU-only preparation; this wrapper never calls sbatch or a GPU runtime.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-union-neural-precision-20260929-v3"}
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
OUTPUT="$DIAGNOSTICS/shared-union-cache-preparation-cpu-20260929.json"
DRY_RUN=0
FORWARDED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --output) [[ $# -ge 2 ]] || exit 2; OUTPUT=$2; shift 2 ;;
    --output=*) OUTPUT=${1#--output=}; shift ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
case $(readlink -m "$OUTPUT") in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
if [[ $DRY_RUN != 1 ]]; then
  case $(hostname -s) in test0[1-4]|ibtest0[1-4]) ;; *) exit 2 ;; esac
  [[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
  [[ ! -e "$OUTPUT" && ! -e "${OUTPUT%.json}.tiny-cache" ]] || exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=10s 320s env
  SINGULARITYENV_JAX_PLATFORMS=cpu SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=1
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  singularity exec --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro" --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
  "$ROOT/.venvs/orcjax_cpu/bin/python" "$OVERLAY/probe_shared_union_cache_preparation_cpu.py"
  --repo-root "$CODE_ROOT" --neural-probe-file "$OVERLAY/probe_shared_neural_precision.py"
  --real-daily-probe-file "$ROOT/runtime/transfer/shared-ffn-precision-20260926/probe_shared_expert_real_daily.py"
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --reference-result "$DIAGNOSTICS/shared-union-integrated-20260927.pkl"
  --reuse-stage-report "$DIAGNOSTICS/shared-union-integrated-20260927.json"
  --candidate-module-file "$ROOT/runtime/transfer/shared-union-integrated-20260927/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --output "$OUTPUT" --process-seconds 300)
COMMAND+=(${FORWARDED[@]+"${FORWARDED[@]}"})
if [[ $DRY_RUN == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
cd "$CODE_ROOT"
"${COMMAND[@]}"
