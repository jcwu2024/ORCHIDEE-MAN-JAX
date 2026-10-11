set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
TRANSFER=$PROJECT/runtime/transfer/gpu-lifecycle-20261011-3b4ae35
export CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1
export EXPECTED_CODE_SHA=3b4ae35e695f85480ecb72461e4ba93f4534bbd8
export PYTHONDONTWRITEBYTECODE=1
unset RUN_ROOT
test -d "$PROJECT/runtime/logs" && test -w "$PROJECT/runtime/logs"
cd "$CODE_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_CODE_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
node=$(/rmprog/slurm/v22.05.7/bin/scontrol show node ibc13b03n02)
printf '%s\n' "$node"
printf '%s\n' "$node" | grep -q 'CPUAlloc=0 '
printf '%s\n' "$node" | grep -q 'State=IDLE '
mkdir "$TRANSFER/submission-lock"
date -u +%FT%TZ > "$TRANSFER/submission-lock/attempt-utc.txt"
command=(/rmprog/slurm/v22.05.7/bin/sbatch --partition=gnall --nodes=1 --ntasks=1
  --cpus-per-task=4 --gres=gpu:1 --time=02:00:00 --no-requeue
  --chdir="$CODE_DIR"
  --output="$PROJECT/runtime/logs/global_gpu_lifecycle_%j.txt"
  --error="$PROJECT/runtime/logs/global_gpu_lifecycle_%j.txt"
  --nodelist=ibc13b03n02 "$CODE_DIR/scripts/hpc/slurm_global_gpu_lifecycle.sh")
printf '%q ' "${command[@]}" > "$TRANSFER/submission-lock/command.txt"
printf '\n' >> "$TRANSFER/submission-lock/command.txt"
submitted=$("${command[@]}")
printf '%s\n' "$submitted" | tee "$TRANSFER/submitted-job-id.txt"
test "$submitted" != ''
job=${submitted##* }
case "$job" in *[!0-9]*|'') exit 3;; esac
/rmprog/slurm/v22.05.7/bin/scontrol show job "$job"
