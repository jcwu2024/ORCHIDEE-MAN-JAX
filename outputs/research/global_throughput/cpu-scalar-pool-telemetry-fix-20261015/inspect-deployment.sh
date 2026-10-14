#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RT=$PROJECT/runtime
[[ $(realpath -m "$RT") == /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime ]]
date -u +%FT%TZ
hostname
git --version
cat "$RT/deployment/README.md"
cat "$RT/deployment/registry.json"
git --git-dir="$RT/deployment/repository.git" rev-parse refs/heads/codex/global-cpu-scalar-pool-v1
/rmprog/slurm/v22.05.7/bin/sinfo -N -p cntest -o '%N %t %c %C'
test ! -e "$RT/worktrees/global-throughput-cpu-scalar-pool-v2"
test ! -e "$RT/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015"
