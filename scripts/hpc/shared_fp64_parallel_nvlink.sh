#!/bin/bash
# Transparent per-stage invocation evidence for the independent nvlink.
set -uo pipefail
REAL=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/toolchains/cuda-nvlink-12.6.85/bin/nvlink
LOG=${ORCHIDEE_NVLINK_LOG:-}
[[ $LOG == /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/* ]] || exit 2
[[ -x $REAL ]] || exit 2
(
  flock 9 || exit 2
  printf '%s\tstage=%s\tpid=%s\t' "$(date +%s.%N)" "${ORCHIDEE_NVLINK_STAGE:-unknown}" "$$" >&9
  printf '%q ' "$@" >&9
  printf '\n' >&9
) 9>>"$LOG" || exit 2
exec "$REAL" "$@"
