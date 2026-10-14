#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
EVIDENCE=$PROJECT/runtime/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
CODE=$PROJECT/runtime/worktrees/global-throughput-cpu-scalar-pool-v3
[[ ! -e $EVIDENCE/free-verification.started ]]
date -u +%FT%TZ > "$EVIDENCE/free-verification.started"
while IFS= read -r option; do unset "$option"; done < <(compgen -A variable SBATCH_)
set +e
/rmprog/slurm/v22.05.7/bin/srun --partition=cntest --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:05:00 --cpu-bind=cores --chdir="$CODE" --export=ALL bash "$EVIDENCE/verify-target.sh" > "$EVIDENCE/free-verification.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$EVIDENCE/free-verification-exit-code.txt"
cat "$EVIDENCE/free-verification.log"
exit "$status"
