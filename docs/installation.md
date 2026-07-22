# Installation

The supported deployment is a Git source checkout. This keeps the portable
configuration, manifests, HPC templates, and provenance evidence beside the
installed Python package.

## CPU

```bash
uv sync --frozen
uv run orchidee-jax --version
```

## CUDA 12

```bash
uv sync --frozen --extra cuda12
```

Verify that the installed JAX build sees the intended devices before a long
run. GPU execution is optional; CPU remains the reference deployment target.

## Offline Cluster Nodes

Create the environment and populate the uv cache on a login node. Compute
jobs should use `uv run --frozen` and must not resolve or download packages.
Place `UV_CACHE_DIR` on a readable project or node-local cache as appropriate
for the scheduler.

Set `ORCHIDEE_OUTPUT_ROOT` to writable scratch. JAX compilation artifacts are
stored under `$ORCHIDEE_OUTPUT_ROOT/xla_cache`; the checkout itself may be
read-only after `uv sync`.

## Explore1000

Explore1000 runs CentOS 7. The general `uv.lock` currently targets a newer JAX
stack and may select wheels requiring a newer glibc. Use the accepted pinned
CPU compatibility profile instead:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
bash scripts/hpc/bootstrap_orcjax_cpu.sh
```

This creates or reconciles
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu`. Run the bootstrap on `cln01`, which
has package-index access. Compute and test nodes consume the shared environment
read-only.

The canonical GPU environment will be
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu`, but it is not yet frozen. Existing
legacy GPU environments are validation assets and must not be treated as the
project-wide runtime. Build `orcjax_gpu` only after neural training
dependencies are fixed, then reproduce the accepted V100 compatibility gate.

See `deployment-explore1000.md` for node roles, Git-bundle transfer, and paid
job approval rules.
