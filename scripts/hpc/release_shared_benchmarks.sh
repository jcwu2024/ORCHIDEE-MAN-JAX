#!/bin/bash
# Poll on a free CPU node; held Slurm jobs consume no GPU allocation.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
REPORT=${1:?reference report}
shift
[[ $# == 2 ]] || exit 2
case "$REPORT" in "$ROOT"/runtime/outputs/diagnostics/*.json) ;; *) exit 2 ;; esac
for JOB in "$@"; do [[ $JOB =~ ^[0-9]+$ ]] || exit 2; done
case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2 ;; esac
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
SCHEDULER=/rmprog/slurm/v22.05.7/bin/scontrol
DEADLINE=$((SECONDS+2700))
while (( SECONDS < DEADLINE )); do
  STATUS=$($PYTHON -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(json.loads(p.read_text()).get("status", "pending") if p.exists() else "pending")' "$REPORT")
  if [[ $STATUS == completed ]]; then
    for JOB in "$@"; do
      # Scheduler operations are sent through the authorized login endpoint.
      ssh cln01 "$SCHEDULER release $JOB"
      echo "released=$JOB"
    done
    exit 0
  fi
  case "$STATUS" in error|cache_miss|rejected|failed)
    echo "reference_status=$STATUS; jobs remain held: $*"
    exit 1 ;;
  esac
  sleep 20
done
echo "reference deadline reached; jobs remain held: $*"
exit 1
