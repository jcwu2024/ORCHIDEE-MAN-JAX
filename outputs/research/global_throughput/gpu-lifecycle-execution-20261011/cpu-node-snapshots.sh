#!/bin/bash
set -euo pipefail
for NODE in test01 test03 test04; do
  printf 'cpu_test_node=%s\n' "$NODE"
  ssh -T -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 "$NODE" bash -s <<'NODE'
hostname
date -u +%FT%TZ
uptime
taskset -pc $$
free -m
NODE
done
