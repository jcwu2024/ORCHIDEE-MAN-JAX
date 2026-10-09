#!/bin/bash
# Terminal saved-array verification only; no model invocation or GPU backend.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$PROJECT/runtime/worktrees/global-throughput-gpu-fixed2048-v1
RUN_ROOT=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1
PREP=$PROJECT/runtime/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export ORCHJAX_DISABLE_COMPILATION_CACHE=1
date -u +%FT%TZ
hostname
taskset -pc $$
cd "$CODE"
test "$(git rev-parse HEAD)" = 8aa2e05aa24b65dc4d30a27d4ee857bba6312221
test -z "$(git status --porcelain --untracked-files=all)"
set +e
/usr/bin/time -v taskset -c 0-3 "$PROJECT/.venvs/orcjax_cpu/bin/python" -B "$CODE/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/collect_fixed2048_result.py" \
 --code-dir "$CODE" --expected-code-sha 8aa2e05aa24b65dc4d30a27d4ee857bba6312221 \
 --stage-dir "$RUN_ROOT/capacity" --sacct-file "$PREP/job-15671732-sacct-raw9.txt" \
 --admission "$PROJECT/runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json" \
 --b1024-evidence-dir "$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity" \
 --job-id 15671732 --job-name orchjax-gpu-cap2048 --node ibc13b03n03 \
 --gpu-uuid GPU-38115eff-1b81-4dee-ae92-996c77969246 --slurm-global-gpu-id 0 --driver-version 525.85.12 \
 --approved-nodes 1 --approved-tasks 1 --approved-cpus 4 --approved-gpus 1 --approved-wall-seconds 7200 \
 --output "$PREP/collected-fixed2048.json"
collector_exit=$?
set -e
printf 'collector_exit=%s\n' "$collector_exit"
printf '%s\n' "$collector_exit" > "$PREP/collector-exit-code.txt"
test -z "$(git status --porcelain --untracked-files=all)"
printf 'execution_checkout_clean_after=true\n'
date -u +%FT%TZ
exit "$collector_exit"
