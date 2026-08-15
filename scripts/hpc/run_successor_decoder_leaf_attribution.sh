#!/bin/bash
# Run the free-gln01 four-leaf decoder attribution.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the accepted 1001-2000 manifest}
RECOVERY_PATH=${RECOVERY_PATH:?set the accepted update-1000 recovery}
OUTPUT=${OUTPUT:?set a new report path under the project runtime root}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
RUNNER_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_training_staged_v3.json
AUDIT_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_decoder_leaf_attribution_v1.json
PREDECESSOR_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_group_aware_step_extension_v1.json
PREDECESSOR_REPORT=$ROOT/runtime/outputs/training/successor-group-aware-step-extension-f0b2235/report.json
CUMULATIVE_REPORT=$ROOT/runtime/outputs/training/successor-cumulative-microstep-b3d2f5f/report.json
GEOMETRY_REPORT=$ROOT/runtime/outputs/training/successor-parameter-geometry-c51dd87/report.json
CANDIDATE_COEFFICIENTS=$WORKTREE/manifests/coarse_graining/successor_named_biomass_candidate_coefficients_v1.json
TENDENCY_ACCEPTANCE=$WORKTREE/manifests/coarse_graining/successor_named_biomass_tendency_scale_acceptance_v1.json
TENDENCY_SCALE_ASSET=$ROOT/runtime/outputs/training/successor-named-biomass-tendency-scales-5bd55c7/scale_asset.json
ASSETS=$ROOT/runtime/outputs/training/successor-staged-assets-b800db9
COEFFICIENT_SOURCE=$ROOT/runtime/outputs/training/successor-multibatch-coefficients-d3200df/probe_report.json
COEFFICIENT_PREPARED=$ROOT/runtime/outputs/training/successor-sufficiency-coeff-prepared-16-c7c6af6/prepared_manifest.json
VALIDATION_PREPARED=$ROOT/runtime/outputs/training/successor-staged-validation-a2c2638/prepared_manifest.json

case "$(hostname -s)" in
  gln01|ibgln01|ibc13b02n04) ;;
  *) echo "decoder-leaf attribution requires free gln01" >&2; exit 2 ;;
esac
case "$OUTPUT" in "$ROOT/runtime/"*) ;; *) exit 2 ;; esac
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$RUNNER_CONTRACT" \
  "$AUDIT_CONTRACT" "$PREDECESSOR_CONTRACT" "$PREDECESSOR_REPORT" \
  "$CUMULATIVE_REPORT" "$GEOMETRY_REPORT" "$CANDIDATE_COEFFICIENTS" \
  "$TENDENCY_ACCEPTANCE" "$TENDENCY_SCALE_ASSET" \
  "$ASSETS/frozen_coefficients.json" "$ASSETS/validation_panel.json" \
  "$COEFFICIENT_SOURCE" "$COEFFICIENT_PREPARED" \
  "$VALIDATION_PREPARED" "$PREPARED_MANIFEST" "$RECOVERY_PATH"; do
  test -f "$path"
done
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
nvidia-smi -i "$GPU_DEVICE" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
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
  SINGULARITYENV_PYTHONHASHSEED=0 \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.probe_successor_parameter_geometry \
      --decoder-leaf-attribution \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --runner-contract "$RUNNER_CONTRACT" \
      --audit-contract "$AUDIT_CONTRACT" \
      --predecessor-contract "$PREDECESSOR_CONTRACT" \
      --predecessor-report "$PREDECESSOR_REPORT" \
      --cumulative-report "$CUMULATIVE_REPORT" \
      --parameter-geometry-report "$GEOMETRY_REPORT" \
      --candidate-coefficient-contract "$CANDIDATE_COEFFICIENTS" \
      --tendency-scale-asset "$TENDENCY_SCALE_ASSET" \
      --tendency-scale-acceptance "$TENDENCY_ACCEPTANCE" \
      --coefficient-contract "$ASSETS/frozen_coefficients.json" \
      --coefficient-report "$COEFFICIENT_SOURCE" \
      --coefficient-prepared-manifest "$COEFFICIENT_PREPARED" \
      --validation-panel "$ASSETS/validation_panel.json" \
      --validation-prepared-manifest "$VALIDATION_PREPARED" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --recovery-path "$RECOVERY_PATH" \
      --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"
