#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$PROJECT/runtime/worktrees/global-throughput-gpu-fixed2048-v1
RUN_ROOT=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1
PREP=$PROJECT/runtime/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009
S=/rmprog/slurm/v22.05.7/bin
NODE=ibc13b03n03
date -u +%FT%TZ
cd "$CODE"
test "$(git rev-parse HEAD)" = 8aa2e05aa24b65dc4d30a27d4ee857bba6312221
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e "$RUN_ROOT"
test -d "$PROJECT/runtime/logs"
existing=$("$S/squeue" -u "$(id -un)" -h -o '%i|%j|%T|%R')
printf 'existing_own_jobs=%s\n' "$existing"
[[ $existing != *orchjax-gpu-cap2048* ]]
node_detail=$("$S/scontrol" show node "$NODE" -o)
printf '%s\n' "$node_detail"
[[ $node_detail =~ CPUAlloc=([0-9]+) ]]; allocated_cpu=${BASH_REMATCH[1]}
[[ $node_detail =~ CPUTot=([0-9]+) ]]; total_cpu=${BASH_REMATCH[1]}
[[ $node_detail =~ FreeMem=([0-9]+) ]]; host_free_mb=${BASH_REMATCH[1]}
[[ $node_detail =~ State=(IDLE|MIXED)([[:space:]]|$) ]]
allocation=${node_detail#*AllocTRES=}; allocation=${allocation%% *}
allocated_gpu=0
if [[ $allocation =~ gres/gpu=([0-9]+) ]]; then allocated_gpu=${BASH_REMATCH[1]}; fi
(( total_cpu - allocated_cpu >= 4 && allocated_gpu < 8 && host_free_mb >= 65536 ))
printf 'preflight_exact_clean=true output_root_absent=true qualified_node=%s available_slurm_cpus=%s available_gpus=%s host_free_mb=%s\n' "$NODE" "$((total_cpu-allocated_cpu))" "$((8-allocated_gpu))" "$host_free_mb"
cd "$PROJECT"
printf 'actual_submission_command='; printf '%q ' "$S/sbatch" --nodelist="$NODE" --export=ALL,CODE_DIR="$CODE",RUN_ROOT="$RUN_ROOT" "$CODE/scripts/hpc/slurm_global_gpu_capacity_2048.sh"; printf '\n'
set +e
"$S/sbatch" --nodelist="$NODE" --export=ALL,CODE_DIR="$CODE",RUN_ROOT="$RUN_ROOT" "$CODE/scripts/hpc/slurm_global_gpu_capacity_2048.sh" > "$PREP/submission-sbatch-stdout.txt" 2> "$PREP/submission-sbatch-stderr.txt"
submission_exit=$?
set -e
printf 'actual_sbatch_exit=%s\n' "$submission_exit"
cat "$PREP/submission-sbatch-stdout.txt"
cat "$PREP/submission-sbatch-stderr.txt"
exit "$submission_exit"
