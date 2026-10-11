set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1
export PYTHONDONTWRITEBYTECODE=1
hostname
date -u +%FT%TZ
cd "$CODE"
"$PROJECT/.venvs/orcjax_cpu/bin/python" -B - <<'PY'
import json, sys
from pathlib import Path
import jax_orchidee
from scripts.dev import global_lifecycle_science as science
pin='3b4ae35e695f85480ecb72461e4ba93f4534bbd8'
identity=science.source_identity(Path.cwd(),expected_code_sha=pin,require_clean=True)
binding=science.run_two_binding(Path.cwd())
assert 'jax' not in sys.modules
print(json.dumps({'module_path':jax_orchidee.__file__, 'execution_source_identity':identity,
                  'run_two_binding':binding,'JAX_imported':False,'model_execution':False}))
PY
test -z "$(git status --porcelain --untracked-files=all)"
