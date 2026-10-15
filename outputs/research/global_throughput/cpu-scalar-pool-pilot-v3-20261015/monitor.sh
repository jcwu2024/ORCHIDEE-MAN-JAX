#!/bin/bash
set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
R=$P/runtime/outputs/research/global_throughput/server-benchmark/cpu4-15869035
S=/rmprog/slurm/v22.05.7/bin
date -u +%FT%TZ
"$S/squeue" -j 15869035 -o '%i %P %j %T %M %l %C %R'
"$S/scontrol" show job --details 15869035
"$S/sacct" -j 15869035 --noheader --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,TotalCPU,UserCPU,SystemCPU,CPUTimeRAW,MaxRSS,AveRSS
if [[ -f $P/runtime/logs/global_scalar_pool_15869035.txt ]]; then
  printf '\n=== job log tail ===\n'
  tail -n 55 "$P/runtime/logs/global_scalar_pool_15869035.txt"
fi
if [[ -d $R ]]; then
  printf '\n=== generated files ===\n'
  find "$R" -maxdepth 3 -type f -printf '%P %s bytes\n'
fi
