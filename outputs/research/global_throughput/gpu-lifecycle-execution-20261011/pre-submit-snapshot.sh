set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1
hostname
date -u +%FT%TZ
/rmprog/slurm/v22.05.7/bin/sinfo -p gnall -N -o '%N|%T|%C|%O|%G'
/rmprog/slurm/v22.05.7/bin/scontrol show node ibc13b03n02
test -d "$PROJECT/runtime/logs" && test -w "$PROJECT/runtime/logs"
cd "$CODE"
test "$(git rev-parse HEAD)" = 3b4ae35e695f85480ecb72461e4ba93f4534bbd8
test -z "$(git status --porcelain --untracked-files=all)"
echo FINAL_CODE_AND_SHARED_LOG_ADMISSION_PASS
