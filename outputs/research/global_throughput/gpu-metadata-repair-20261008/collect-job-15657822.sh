#!/bin/bash
set -uo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
EVIDENCE="$PROJECT/runtime/outputs/research/global_throughput/gpu-metadata-repair-20261008"
CODE_DIR="$PROJECT/runtime/worktrees/global-throughput-gpu-metadata-v1"
STAGE_DIR="$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-15657822/capacity"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u +%FT%TZ > "$EVIDENCE/job-15657822-collection-start-UTC.txt"
/usr/bin/time -v -o "$EVIDENCE/job-15657822-collection-time.txt" taskset -c 0-3 \
  "$PROJECT/.venvs/orcjax_cpu/bin/python" "$EVIDENCE/collect_b512_result.py" \
  --code-dir "$CODE_DIR" --stage-dir "$STAGE_DIR" \
  --admission "$PROJECT/runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json" \
  --sacct-file "$EVIDENCE/job-15657822-terminal-sacct-raw9.txt" \
  --job-id 15657822 --job-name orchjax-gpu-b512 --node ibc13b03n02 \
  --gpu-uuid GPU-949cb714-8c53-353c-1f6a-1832c629230e --slurm-global-gpu-id 4 --driver-version 525.85.12 \
  --approved-nodes 1 --approved-tasks 1 --approved-cpus 4 --approved-gpus 1 --approved-wall-seconds 2700 \
  --b128-baseline-dir "$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity" \
  --b256-baseline-dir "$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity" \
  --output "$EVIDENCE/job-15657822-collected-result.json" \
  > "$EVIDENCE/job-15657822-collection-stdout.txt" 2> "$EVIDENCE/job-15657822-collection-stderr.txt"
COLLECTION_EXIT=$?
printf '%s\n' "$COLLECTION_EXIT" > "$EVIDENCE/job-15657822-collection-exit.txt"
date -u +%FT%TZ > "$EVIDENCE/job-15657822-collection-end-UTC.txt"
cd "$CODE_DIR"
git rev-parse HEAD > "$EVIDENCE/job-15657822-code-after-collection.txt"
git status --porcelain --untracked-files=all > "$EVIDENCE/job-15657822-clean-after-collection.txt"
exit "$COLLECTION_EXIT"
