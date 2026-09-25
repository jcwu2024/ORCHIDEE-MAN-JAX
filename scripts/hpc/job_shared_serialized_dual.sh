#!/bin/bash
#SBATCH --job-name=shared-dual-replay
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dual-replay-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dual-replay-%j.log
#SBATCH --no-requeue

set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ $DRY_RUN == 1 ]]; then
  GPU_ALLOCATION=${CUDA_VISIBLE_DEVICES:-0,1}
  RUN_ID=${SLURM_JOB_ID:-dry-run}
else
  GPU_ALLOCATION=${CUDA_VISIBLE_DEVICES:?Slurm must expose the two allocated GPUs}
  RUN_ID=${SLURM_JOB_ID:?launch inside the approved Slurm allocation}
fi
IFS=, read -r GPU_DEVICE WORKER_GPU_DEVICE EXTRA_GPU <<< "$GPU_ALLOCATION"
[[ -n "$GPU_DEVICE" && -n "$WORKER_GPU_DEVICE" && -z ${EXTRA_GPU:-} ]] || exit 2
export GPU_DEVICE WORKER_GPU_DEVICE
export RUN_TIMEOUT=28m
export OUTPUT="$ROOT/runtime/outputs/diagnostics/shared-dual-replay-${RUN_ID}.json"
LAUNCHER="$ROOT/runtime/transfer/run_shared_serialized_dual_paid.sh"
if [[ $DRY_RUN == 1 ]]; then
  exec bash "$LAUNCHER" --dry-run "$@"
fi

echo "host=$(hostname -s) job=$RUN_ID allocated_cuda=$GPU_ALLOCATION"
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
# A short read check can reject an unhealthy node before model setup. It is
# not a dataset sweep and does not certify all subsequent filesystem IO.
for INPUT_FILE in \
  "$CODE_ROOT/scripts/hpc/run_shared_learning.py" \
  /WORK/liwei_work/jcwu/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/lib/python3.11/urllib/parse.py \
  "$ROOT/runtime/outputs/diagnostics/shared-split-profile-c599236.json"
do
  if ! timeout --signal=TERM --kill-after=3s 15s head -c 64 "$INPUT_FILE" > /dev/null; then
    echo "filesystem_read_failed=$INPUT_FILE"
    exit 75
  fi
done
exec bash "$LAUNCHER" "$@"
