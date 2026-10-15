#!/bin/bash
set -euo pipefail
E=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
/rmprog/slurm/v22.05.7/bin/scontrol show config > "$E/scheduler-config.raw.txt"
awk '/^[[:space:]]*(SelectType|SelectTypeParameters|TaskPlugin|TaskPluginParam|ProctrackType|SlurmdParameters|SchedulerParameters|CpuFreqDef|CpuFreqGovernors)[[:space:]]*=/' "$E/scheduler-config.raw.txt" | tee "$E/scheduler-config-selected.txt"
sha256sum "$E/scheduler-config.raw.txt" "$E/scheduler-config-selected.txt"
