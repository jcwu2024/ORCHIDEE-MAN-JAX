#!/bin/bash
set -euo pipefail
EVIDENCE=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
[[ $(realpath "$EVIDENCE") == "$EVIDENCE" ]]
[[ -f $EVIDENCE/free-verification.log ]]
[[ -f $EVIDENCE/free-verification-exit-code.txt ]]
[[ -f $EVIDENCE/free-verification.started ]]
[[ ! -e $EVIDENCE/free-verification-dispatch-path-error.log ]]
mv "$EVIDENCE/free-verification.log" "$EVIDENCE/free-verification-dispatch-path-error.log"
mv "$EVIDENCE/free-verification-exit-code.txt" "$EVIDENCE/free-verification-dispatch-path-error-exit-code.txt"
mv "$EVIDENCE/free-verification.started" "$EVIDENCE/free-verification-dispatch-path-error.started"
