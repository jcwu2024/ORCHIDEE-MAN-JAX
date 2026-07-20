# ORCHIDEE-MAN JAX

English | [简体中文](README.zh-CN.md)

Source-backed JAX implementation of the ORCHIDEE-MAN paper-era model for the
PFT14 mangrove configuration. Scientific process logic is ported from the
Fortran source and carries file, procedure, and line-span provenance.

## Current Scope

- PFT14 single-landpoint driver, SECHIBA, HYDROL, DIFFUCO, ENERBIL,
  THERMOSOIL, CONDVEG, STOMATE, restart handoff, and annual modelout.
- Complete later-day transitions run in compiled seven-day blocks while
  preserving the Fortran half-hour and daily state order.
- Cold-start 14-day, restart 14-day, 365-day, and cross-landpoint executable
  reuse gates pass on the current production path.
- Seven previously unused landpoints have completed 1961-2010 validation;
  their largest 2010 AGB/BGB/GPP/NPP relative error is `8.5e-5`.
- The full 669-landpoint acceptance run has not yet been completed.

This scope statement is deliberately narrower than claiming equivalence for
other PFTs or unsupported ORCHIDEE configurations.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `jax_orchidee/` | Production JAX model, driver, process modules, and CLI runners |
| `configs/` | Portable PFT14 paper-case configuration |
| `manifests/` | External data contract and 669-landpoint metadata |
| `scripts/hpc/` | Generic isolated-landpoint scheduler templates |
| `scripts/dev/` | Oracle, audit, diagnosis, and performance tooling; not a runtime dependency |
| `tests/` | Unit, parity, Oracle-evidence, and integration tests |
| `docs/` | User documentation and source-driven equivalence evidence |
| `fortran_run_scripts/` | Immutable historical paper protocol provenance |
| `fortran_source/` | Local mount point for licensed Fortran source; source tree is Git-ignored |
| `data/`, `reference/` | Local/external scientific inputs and Fortran truth; contents are Git-ignored |
| `outputs/`, `traces/` | Generated results, caches, and raw diagnostics; contents are Git-ignored |

Root-level `AGENTS.md`, `SERVER_ACCESS.md`, `.agents/`, `.venv/`, and tool
caches are local workspace controls and are not part of the public Git
candidate.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
git clone <repository-url> orchidee-man-jax
cd orchidee-man-jax
uv sync --frozen
uv run orchidee-jax --help
```

For development tools:

```bash
uv sync --frozen --extra dev
```

CUDA is optional and must match the server driver:

```bash
uv sync --frozen --extra cuda12
```

## External Data

Large forcing, static, restart, and reference files are not stored in Git.
Set these roots before running:

```bash
export ORCHIDEE_DATA_ROOT=/shared/orchidee-data
export ORCHIDEE_REFERENCE_ROOT=/shared/orchidee-reference
export ORCHIDEE_OUTPUT_ROOT=/scratch/$USER/orchidee-jax
```

The required layout is documented in
`manifests/paper_pft14_data.yaml`. Verify it with:

```bash
uv run orchidee-jax inventory --config configs/orchidee_man_250919.yaml
```

When the environment variables are absent, a development checkout falls back
to `data/`, `reference/`, and `outputs/` under the repository root.

## Run One Landpoint

```bash
uv run orchidee-jax run \
  --landpoint-id 001.0-071.0 \
  --start-year 1961 \
  --years 50 \
  --initial-state cold-start \
  --year-checkpoint-dir "$ORCHIDEE_OUTPUT_ROOT/checkpoints/001.0-071.0" \
  --resume-checkpoints on \
  --output "$ORCHIDEE_OUTPUT_ROOT/001.0-071.0.json"
```

The user-facing runner enables the accepted seven-day complete-day block by
default. Production runners live in `jax_orchidee/runners`; development
strict/A-B diagnostics remain available in `scripts/dev`.

## Validate Landpoints

```bash
uv run orchidee-jax validate-landpoints \
  --selection manifests/landpoints_669.json \
  --start-year 1961 \
  --years 50 \
  --initial-state cold-start \
  --strict-on-failure off \
  --output-dir "$ORCHIDEE_OUTPUT_ROOT/acceptance"
```

For production clusters, run one landpoint per process or scheduler array
task. This isolates JAX/XLA memory, makes restart checkpoints independent, and
avoids the native Windows failure observed when one process compiled several
landpoints consecutively.

## Scientific Policy

- Fortran source is the process truth.
- Runtime references validate numerical and state behavior; they do not
  replace source-driven branch coverage.
- Missing processes are not approximated or hidden by calibration.
- Discrete states compare exactly. Floating-point gates use explicit,
  field-aware tolerances recorded under `docs/source_audits`.

## Source and Licensing

The JAX repository license and citation metadata must be selected by the
project owner before public release. The bundled Fortran source tree has no
verified top-level redistribution license in this checkout; exclude it from a
public release unless redistribution permission is confirmed.

See `docs/installation.md`, `docs/data-layout.md`, and `docs/running.md` for
deployment details. `docs/release-checklist.md` records the remaining license,
citation, cluster-review, and 669-point acceptance gates.
