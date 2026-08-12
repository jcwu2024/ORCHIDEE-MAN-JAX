#!/bin/bash
# Run the free-gln01 two-update staged-training smoke.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the clean staged-training commit}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the two-update v3 manifest}
VALIDATION_PREPARED_MANIFEST=${VALIDATION_PREPARED_MANIFEST:?set fixed validation prepared manifest}
OUTPUT=${OUTPUT:?set a new output report below the project runtime root}
ASSETS=$ROOT/runtime/outputs/training/successor-staged-assets-b800db9
COEFFICIENT_SOURCE=$ROOT/runtime/outputs/training/successor-multibatch-coefficients-d3200df/probe_report.json
COEFFICIENT_PREPARED=$ROOT/runtime/outputs/training/successor-sufficiency-coeff-prepared-16-c7c6af6/prepared_manifest.json
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$(hostname)" in gln01|ibgln01) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT/runtime/"*) ;; *) exit 2 ;; esac
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
for path in "$PREPARED_MANIFEST" "$VALIDATION_PREPARED_MANIFEST" \
  "$ASSETS/frozen_coefficients.json" "$ASSETS/validation_panel.json" \
  "$COEFFICIENT_SOURCE" "$COEFFICIENT_PREPARED"; do
  test -f "$path"
done
mkdir -p "$(dirname "$OUTPUT")"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.probe_successor_staged_training \
      --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
      --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
      --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
      --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
      --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
      --runner-contract "$WORKTREE/manifests/coarse_graining/successor_training_staged_v3.json" \
      --coefficient-contract "$ASSETS/frozen_coefficients.json" \
      --coefficient-report "$COEFFICIENT_SOURCE" \
      --coefficient-prepared-manifest "$COEFFICIENT_PREPARED" \
      --validation-panel "$ASSETS/validation_panel.json" \
      --validation-panel-sha256 1ac7490a2c7067bf3f66faea75836d80a9e6fdc62ebcdc8673caa3f445c70350 \
      --validation-prepared-manifest "$VALIDATION_PREPARED_MANIFEST" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --prepared-source-git-head "$EXPECTED_GIT_HEAD" \
      --output "$OUTPUT"

sha256sum "$OUTPUT"
