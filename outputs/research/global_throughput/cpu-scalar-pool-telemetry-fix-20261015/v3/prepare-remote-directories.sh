#!/bin/bash
set -euo pipefail
RT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
EVIDENCE=$RT/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
TRANSFER=$RT/transfer/global-cpu-scalar-pool-v3-20261015
for path in "$EVIDENCE" "$TRANSFER"; do
  case "$(realpath -m "$path")" in "$RT"/*) ;; *) exit 2 ;; esac
  [[ ! -e $path ]]
done
mkdir -p "$EVIDENCE" "$TRANSFER"
date -u +%FT%TZ
