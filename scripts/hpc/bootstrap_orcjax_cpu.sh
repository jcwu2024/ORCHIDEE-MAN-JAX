#!/bin/bash
set -euo pipefail

JCWU_ROOT=${JCWU_ROOT:-/WORK/liwei_work/jcwu}
REPO_ROOT=${ORCHIDEE_REPO_ROOT:-$JCWU_ROOT/ORCHIDEE-MAN-JAX}
RUNTIME_ROOT=${ORCHIDEE_RUNTIME_ROOT:-$REPO_ROOT/runtime}
UV=${UV_BIN:-$JCWU_ROOT/.local/bin/uv}
ENV_DIR=${ORCJAX_CPU_ENV:-$REPO_ROOT/.venvs/orcjax_cpu}
UV_PYTHON=${UV_PYTHON:-$JCWU_ROOT/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}
REQUIREMENTS=$REPO_ROOT/scripts/hpc/requirements-orcjax-cpu.txt

export UV_CACHE_DIR=${UV_CACHE_DIR:-$RUNTIME_ROOT/cache/uv}

test -x "$UV"
test -x "$UV_PYTHON"
test -f "$REQUIREMENTS"
mkdir -p "$RUNTIME_ROOT/cache/uv" "$REPO_ROOT/.venvs"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV" venv --python "$UV_PYTHON" "$ENV_DIR"
fi

"$UV" pip sync --python "$ENV_DIR/bin/python" "$REQUIREMENTS"
JAX_PLATFORMS=cpu "$ENV_DIR/bin/python" -c \
  'import jax, netCDF4, numpy, xarray, yaml; print(jax.__version__, jax.default_backend(), numpy.__version__)'
