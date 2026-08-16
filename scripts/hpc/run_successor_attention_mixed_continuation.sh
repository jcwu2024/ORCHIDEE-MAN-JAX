#!/bin/bash
# Train the Attention candidate on free gln01 with mixed point-years and annealing.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the accepted 1001-2000 manifest}
PARENT_RECOVERY=${PARENT_RECOVERY:?set the accepted update-1000 recovery}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new output directory under the project runtime root}
OUTPUT=${OUTPUT:-$OUTPUT_ROOT/report.json}
STATE_PATH=${STATE_PATH:-$OUTPUT_ROOT/attention_state.pkl}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
RUNNER_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_training_staged_v3.json
CONTINUATION_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_attention_mixed_annealed_continuation_v1.json
ASSETS=$ROOT/runtime/outputs/training/successor-staged-assets-b800db9
COEFFICIENT_SOURCE=$ROOT/runtime/outputs/training/successor-multibatch-coefficients-d3200df/probe_report.json
COEFFICIENT_PREPARED=$ROOT/runtime/outputs/training/successor-sufficiency-coeff-prepared-16-c7c6af6/prepared_manifest.json
VALIDATION_PREPARED=$ROOT/runtime/outputs/training/successor-staged-validation-a2c2638/prepared_manifest.json
BASELINE_REPORT=$ROOT/runtime/outputs/training/successor-staged-fit-1000-ea21f8a/stage_report.json

case "$(hostname -s)" in
  gln01|ibgln01|ibc13b02n04) ;;
  ibc13b02n*|ibc13b03n*)
    test "${SLURM_JOB_PARTITION:-}" = gnall
    test -n "${SLURM_JOB_ID:-}"
    ;;
  *) echo "Attention mixed continuation requires gln01 or a gnall allocation" >&2; exit 2 ;;
esac
case "$OUTPUT_ROOT" in "$ROOT/runtime/"*) ;; *) exit 2 ;; esac
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$RUNNER_CONTRACT" \
  "$CONTINUATION_CONTRACT" "$ASSETS/frozen_coefficients.json" \
  "$ASSETS/validation_panel.json" "$COEFFICIENT_SOURCE" \
  "$COEFFICIENT_PREPARED" "$VALIDATION_PREPARED" "$PREPARED_MANIFEST" \
  "$PARENT_RECOVERY" "$BASELINE_REPORT"; do
  test -f "$path"
done
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
nvidia-smi -i "$GPU_DEVICE" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
mkdir -p "$OUTPUT_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false \
  SINGULARITYENV_PYTHONHASHSEED=0 \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_successor_attention_mixed_continuation \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --runner-contract "$RUNNER_CONTRACT" \
      --continuation-contract "$CONTINUATION_CONTRACT" \
      --coefficient-contract "$ASSETS/frozen_coefficients.json" \
      --coefficient-report "$COEFFICIENT_SOURCE" \
      --coefficient-prepared-manifest "$COEFFICIENT_PREPARED" \
      --validation-panel "$ASSETS/validation_panel.json" \
      --validation-prepared-manifest "$VALIDATION_PREPARED" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --parent-recovery "$PARENT_RECOVERY" \
      --baseline-report "$BASELINE_REPORT" \
      --state-path "$STATE_PATH" \
      --output "$OUTPUT"

test -f "$OUTPUT"
test -f "$STATE_PATH"
sha256sum "$OUTPUT" "$STATE_PATH"
