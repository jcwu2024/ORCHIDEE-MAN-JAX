#!/bin/bash
# Read-only free CPU-node admission; no model/JAX or scientific array load.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1
hostname
case "$(hostname)" in ibtest02|ibctest02|ibtest*|ibctest*) ;; *) echo unexpected_cpu_test_node; exit 2 ;; esac
date -u +%FT%TZ
uptime
taskset -pc $$
free -m
command -v timeout
command -v taskset
cd "$CODE"
test "$(git rev-parse HEAD)" = 3b4ae35e695f85480ecb72461e4ba93f4534bbd8
test -z "$(git status --porcelain --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PROJECT/.venvs/orcjax_cpu/bin/python" -B - <<'PY'
import json, os, sys
import numpy as np
assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'jax_orchidee'} for name in sys.modules)
print(json.dumps({'python': sys.version, 'numpy': np.__version__,
                  'executable': sys.executable, 'affinity': sorted(os.sched_getaffinity(0)),
                  'JAX_or_model_imported': False, 'scientific_arrays_loaded': False}))
PY
