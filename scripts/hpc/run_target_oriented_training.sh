#!/bin/bash
# Run target-oriented free admission or the formal update-1-to-250 stage.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
MODE=${MODE:?set MODE to admission or formal}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
PREPARED_MANIFEST=${PREPARED_MANIFEST:-$ROOT/runtime/outputs/training/target-oriented-prepared-250-${HEAD_SHORT}/prepared_manifest.json}
VALIDATION_PREPARED_MANIFEST=${VALIDATION_PREPARED_MANIFEST:-$ROOT/runtime/outputs/training/target-oriented-validation-${HEAD_SHORT}/prepared_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-training-${MODE}-${HEAD_SHORT}}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/target-oriented-full-step-0d7f936}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$MODE" in admission|formal) ;; *) exit 2 ;; esac
case "$(hostname -s)" in
  gln01|ibgln01|ibc13b02n01|ibc13b02n02|ibc13b02n03|ibc13b02n04|ibc13b03n*) ;;
  *)
    test -n "${SLURM_JOB_ID:-}" || {
      echo "target-oriented training requires free gln01 or a Slurm-allocated GPU" >&2
      exit 2
    }
    ;;
esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$PREPARED_MANIFEST" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
case "$VALIDATION_PREPARED_MANIFEST" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
case "$OUTPUT_DIR" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
case "$CACHE_ROOT" in "$ROOT/runtime/cache/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test -f "$PREPARED_MANIFEST"
test -f "$VALIDATION_PREPARED_MANIFEST"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR/report.json"

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$TRAINING_CONTRACT"; do
  test -f "$path"
done
mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) git_head=$EXPECTED_GIT_HEAD mode=$MODE gpu_device=$GPU_DEVICE output=$OUTPUT_DIR"
nvidia-smi -i "$GPU_DEVICE" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM 290m \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_target_oriented_training \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --training-contract "$TRAINING_CONTRACT" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --validation-prepared-manifest "$VALIDATION_PREPARED_MANIFEST" \
      --prepared-updates 250 \
      --clip-threshold 5323655.8449 \
      --mode "$MODE" \
      --output-dir "$OUTPUT_DIR"

test -f "$OUTPUT_DIR/report.json"
sha256sum "$OUTPUT_DIR/report.json"
