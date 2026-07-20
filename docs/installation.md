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
