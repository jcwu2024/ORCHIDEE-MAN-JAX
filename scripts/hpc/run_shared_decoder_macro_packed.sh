#!/bin/bash
# Actual-layout decoder screen, synthetic payloads; not a model training job.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer"
MICROBATCH=${MICROBATCH:-1}
GPU_INDEX=${GPU_INDEX:-0}
case "$MICROBATCH" in 1|4) ;; *) exit 2 ;; esac
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
OUTPUT=${OUTPUT:-"$ROOT/runtime/outputs/diagnostics/shared-split-macro-packed-20260925-micro${MICROBATCH}.json"}
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(timeout --signal=TERM --kill-after=30s 20m env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT" SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=2
  singularity exec --nv --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_decoder_macro_packed.py"
  --candidate-module-file "$OVERLAY/shared_decoder_macro_packed.py"
  --layout-metadata-file "$OVERLAY/shared-decoder-macro-layout-20260925.json"
  --require-gpu --width 384 --repeats 10 --microbatch "$MICROBATCH"
  --max-compiler-temp-gib 20 --variants packed128 expert128
  --output "$OUTPUT")
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
taskset -pc $$
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
