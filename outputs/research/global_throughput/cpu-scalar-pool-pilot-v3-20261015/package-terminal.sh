#!/bin/bash
set -euo pipefail
E=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
cd "$E"
[[ ! -e terminal-evidence.tar.gz ]]
tar -czf terminal-evidence.tar.gz --exclude=terminal-evidence.tar.gz .
sha256sum terminal-evidence.tar.gz
