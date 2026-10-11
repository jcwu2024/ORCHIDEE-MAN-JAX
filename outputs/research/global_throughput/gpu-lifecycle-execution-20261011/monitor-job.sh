set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
JOB=15769234
RUN=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-$JOB
hostname
date -u +%FT%TZ
/rmprog/slurm/v22.05.7/bin/scontrol show job -o "$JOB" || true
/rmprog/slurm/v22.05.7/bin/sacct -j "$JOB" --parsable2 --units=K \
  --format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocCPUS,AllocTRES,NodeList,MaxRSS
if test -d "$RUN"; then
  find "$RUN" -maxdepth 4 -type f -printf '%P %s bytes\n' | sort
  for file in "$RUN/orchestration-events.jsonl" "$RUN/launch-context.txt"; do
    if test -f "$file"; then printf 'tail=%s\n' "$file"; tail -n 8 "$file"; fi
  done
fi
if test -f "$PROJECT/runtime/logs/global_gpu_lifecycle_$JOB.txt"; then
  echo SLURM_LOG_TAIL
  tail -n 12 "$PROJECT/runtime/logs/global_gpu_lifecycle_$JOB.txt"
fi
