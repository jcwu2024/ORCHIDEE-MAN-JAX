set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
DEPLOY=$PROJECT/runtime/deployment
hostname
date -Is
git --version
/rmprog/slurm/v22.05.7/bin/scontrol show partition cnall
/rmprog/slurm/v22.05.7/bin/sinfo -N -p cnall,cntest -o '%N %P %t %c %C'
/rmprog/slurm/v22.05.7/bin/squeue -p cnall,cntest -o '%.18i %.9P %.12j %.8u %.2t %.10M %.6D %R'
cat "$DEPLOY/README.md"
sha256sum "$DEPLOY/registry.json"
cat "$DEPLOY/registry.json"
git --git-dir="$DEPLOY/repository.git" cat-file -t 03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8
git --git-dir="$DEPLOY/repository.git" show-ref
if test -e "$PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v4"; then
  echo FUTURE_CHECKOUT=exists
else
  echo FUTURE_CHECKOUT=absent
fi
