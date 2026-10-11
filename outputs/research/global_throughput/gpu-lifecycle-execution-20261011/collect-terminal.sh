#!/bin/bash
# Administrative evidence collection only: no model/JAX/validator or job launch.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
JOB=15769234
PIN=3b4ae35e695f85480ecb72461e4ba93f4534bbd8
CODE=$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1
RUN=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-$JOB
TRANSFER=$PROJECT/runtime/transfer/gpu-lifecycle-20261011-3b4ae35
CAPTURE=$TRANSFER/terminal-evidence-$JOB
ARCHIVE=$TRANSFER/terminal-evidence-$JOB.tar.gz
SACCT=/rmprog/slurm/v22.05.7/bin/sacct
SCONTROL=/rmprog/slurm/v22.05.7/bin/scontrol

STATE=$($SACCT -X -j "$JOB" --noheader --parsable2 --format=State | awk 'NR==1 {print $1}')
case "$STATE" in
  COMPLETED*|FAILED*|TIMEOUT*|OUT_OF_MEMORY*|CANCELLED*|NODE_FAIL*|PREEMPTED*) ;;
  *) printf 'collection denied: parent state=%s\n' "$STATE"; exit 2 ;;
esac
test "$(realpath -m "$RUN")" = "$RUN"
test "$(realpath -m "$CAPTURE")" = "$CAPTURE"
test "$(realpath -m "$ARCHIVE")" = "$ARCHIVE"
test -d "$RUN"
test -d "$TRANSFER"
test ! -e "$CAPTURE"
test ! -e "$ARCHIVE"
mkdir "$CAPTURE"
mkdir "$CAPTURE/run"
hostname > "$CAPTURE/collection-host.txt"
date -u +%FT%TZ > "$CAPTURE/collection-server-utc.txt"
printf '%s\n' "$STATE" > "$CAPTURE/parent-state.txt"
$SACCT -j "$JOB" --parsable2 --units=K \
  --format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocCPUS,AllocTRES,NodeList,MaxRSS \
  > "$CAPTURE/terminal-sacct.txt"
$SCONTROL show job -o "$JOB" > "$CAPTURE/terminal-scontrol.txt" 2> "$CAPTURE/terminal-scontrol.stderr.txt" || true
(
  cd "$CODE"
  test "$(git rev-parse HEAD)" = "$PIN"
  git rev-parse HEAD
  git status --porcelain --untracked-files=all
) > "$CAPTURE/terminal-code-identity.txt"
cp "$PROJECT/runtime/logs/global_gpu_lifecycle_$JOB.txt" "$CAPTURE/slurm-log.txt"
cp "$PROJECT/runtime/deployment/registry.json" "$CAPTURE/deployment-registry-current.json"
(
  cd "$RUN"
  find . -type f -printf '%P\t%s bytes\n' | sort > "$CAPTURE/run-file-inventory.tsv"
  find . -type f -print0 | sort -z | xargs -0 -r sha256sum > "$CAPTURE/run-sha256.txt"
  while IFS= read -r -d '' FILE; do
    cp --parents "$FILE" "$CAPTURE/run"
  done < <(find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.log' -o -name '*.tsv' \) -print0 | sort -z)
)
tar -czf "$ARCHIVE" -C "$CAPTURE" .
sha256sum "$ARCHIVE"
stat -c '%s bytes' "$ARCHIVE"
printf 'compact_archive=%s\nfull_scientific_arrays_retained=%s\n' "$ARCHIVE" "$RUN"
