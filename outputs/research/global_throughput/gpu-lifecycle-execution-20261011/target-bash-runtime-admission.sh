set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1
EXPECTED_CODE_SHA=3b4ae35e695f85480ecb72461e4ba93f4534bbd8
TRANSFER=$PROJECT/runtime/transfer/gpu-lifecycle-20261011-3b4ae35
export CODE_DIR EXPECTED_CODE_SHA PYTHONDONTWRITEBYTECODE=1
export ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py
export SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py
hostname
date -u +%FT%TZ
cd "$CODE_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_CODE_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
bash --version | head -n 1
bash -n scripts/hpc/slurm_global_gpu_lifecycle.sh
direct=$(bash scripts/hpc/slurm_global_gpu_lifecycle.sh --dry-run)
printf '%s\n' "$direct"
test ! -e "$TRANSFER/launcher-spooled.sh"
cp scripts/hpc/slurm_global_gpu_lifecycle.sh "$TRANSFER/launcher-spooled.sh"
spooled=$(bash "$TRANSFER/launcher-spooled.sh" --dry-run)
test "$direct" = "$spooled"
echo TARGET_DIRECT_SPOOLED_ARGV_MATCH
planned=$(printf '%s\n' "$direct" | sed -n 's/^command=//p')
test -n "$planned"
eval "$planned"
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER" && test -f "$NVML_DRIVER"
BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
printf 'host_driver_binds=%s\n' "$BINDS"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_JAX_PLATFORMS=cpu SINGULARITYENV_JAX_ENABLE_X64=true
singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR" /apps/soft/sif/foundationpose \
  "$PROJECT/.venvs/orcjax_gpu/bin/python" -B - <<'PY'
import ctypes, json, sys
import jax, jaxlib
assert jax.__version__ == '0.4.38' and jaxlib.__version__ == '0.4.38'
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1')
print(json.dumps({'python':sys.executable, 'jax':jax.__version__, 'jaxlib':jaxlib.__version__,
                  'x64_enabled':bool(jax.config.jax_enable_x64), 'driver_libraries_loaded':True,
                  'scope':'imports/library bindings only; no device enumeration, GPU initialization, model or JIT execution'}))
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo TARGET_CHECKOUT_STILL_CLEAN
