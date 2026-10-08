#!/bin/bash
# Read-only foreground follow-up for the single approved job; no scheduler writes.
set -euo pipefail
JOB=15643868
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
ROOT=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512
SLURM=/rmprog/slurm/v22.05.7/bin
LAST_STATE=
LAST_EVENT=
LAST_STAGES=
STOP_AT=$(date -u -d '2026-10-08 09:11:39' +%s)
while [[ $(date +%s) -le $STOP_AT ]]; do
  ROW=$("$SLURM/sacct" -j "$JOB" -X --parsable2 --noheader --format=JobIDRaw,JobName,Partition,State,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,ExitCode)
  STATE=$(printf '%s\n' "$ROW" | awk -F'|' -v job="$JOB" '$1==job {print $4}')
  EVENT=$(tail -n 1 "$ROOT/capacity/events.jsonl" 2>/dev/null || true)
  STAGES=$(find "$ROOT/capacity" -maxdepth 1 -name 'stage_*.pkl' -printf '%f %s\n' 2>/dev/null | sort)
  if [[ $STATE != "$LAST_STATE" || $EVENT != "$LAST_EVENT" || $STAGES != "$LAST_STAGES" ]]; then
    date -u +%FT%TZ
    printf '%s\n' "$ROW"
    printf 'latest_event=%s\n' "$EVENT"
    printf 'saved_checkpoints=%s\n' "$STAGES"
    LAST_STATE=$STATE
    LAST_EVENT=$EVENT
    LAST_STAGES=$STAGES
  fi
  case "$STATE" in
    COMPLETED|FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL)
      printf 'terminal_observed=true\n'
      exit 0 ;;
  esac
  sleep 60
done
printf 'bounded_followup_reached_accounting_grace_limit=true\n'
exit 3
