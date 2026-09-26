# ORCHIDEE-MAN JAX

English | [简体中文](README.zh-CN.md)

This project has two connected stages: a source-backed, differentiable JAX
implementation of the ORCHIDEE-MAN paper-era PFT14 mangrove model, and a
Transformer/Attention-based daily neural operator trained from that model.
The JAX Teacher is implemented and validated within its declared scope; the
daily replacement remains under development.

## Stage 1: Differentiable JAX Teacher

The Teacher preserves the Fortran process logic, with file, procedure, and
line-span provenance. It still executes 48 half-hour transitions and the
daily vegetation/carbon processes each day.

- PFT14 single-landpoint driver, SECHIBA, HYDROL, DIFFUCO, ENERBIL,
  THERMOSOIL, CONDVEG, STOMATE, restart handoff, and annual modelout.
- Complete later-day transitions run in compiled seven-day blocks while
  preserving the Fortran half-hour and daily state order.
- Cold-start 14-day, restart 14-day, 365-day, and cross-landpoint executable
  reuse gates pass on the current production path.
- Seven previously unused landpoints have completed 1961-2010 validation;
  their largest 2010 AGB/BGB/GPP/NPP relative error is `8.5e-5`.
- Selected physical-parameter gradients pass the canonical Teacher Gate D1:
  forward and reverse automatic differentiation agree with finite differences
  across the declared process, multiday, and restart cases. See the
  [gradient acceptance scope](docs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md).
- Performance work includes compiled day/block execution, executable reuse,
  vectorization, and restartable production runs. It preserves the physical
  time steps; these optimizations do not make the Teacher a daily surrogate.

Scientific support is currently limited to PFT14 and the declared validation
matrix. Neither all-PFT equivalence nor every physical-parameter derivative
is claimed. Full 669-landpoint Fortran-equivalence release acceptance remains
a separate gate from the Teacher dataset acceptance below.

## Stage 2: Daily Neural Process Operator

The research goal is to replace the subdaily process sequence with one daily
transition while keeping long-term GPP, respiration, NPP, aboveground biomass,
and belowground biomass accurate. The current candidate uses a
Transformer/Attention network coupled to a conservative state updater and
retained exact daily processes:

```text
day-start state + native 6-hour forcing + relevant parameters/static conditions
  -> Transformer/Attention process head: daily process quantities
  -> one constrained water/carbon/energy state update
  -> retained exact daily processes
  -> next-day state
```

The neural head connects through the existing `bind_daily_operator` boundary
with the canonical input assembler and retained-tail adapter. Parameters are
routed to the processes that use them. Daily inference must not reconstruct
48 half-hour forcing inputs or execute a 48-step state scan. Retaining the
physical next-day state allows days to be linked into multiyear simulations.

The frozen **v6 Teacher dataset is complete and accepted for neural training**:
669 landpoints, 1961-2010, 33,450 point-year shards, and 12,208,581 daily
transitions. See the
[data acceptance report](docs/research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md).
Use this coherent release; do not regenerate it or mix historical v5 rows into
training. Dataset acceptance does not establish daily-model accuracy.

### Research status and development entry

The daily model, physical coupling, and training/rollout infrastructure are
implemented. Existing Transformer/Attention experiments use daily and 7/30-day
science objectives and native checkpointed JAX differentiation. A model that
passes the required long-term accuracy gates is **not yet available**.

The current work order, measured performance and unresolved acceptance gates
are maintained in [NEXT_STEPS](docs/NEXT_STEPS.md) and
[current status](docs/current-status.md). These compact pages replace the
accumulated experiment chronology as the development entry point. Follow the
[context maintenance workflow](docs/RESEARCH_WORKFLOW.md) to select only the
code and evidence needed for a task. Historical experiments are accessible
through the [history index](docs/research/daily_coarse_graining/HISTORY_INDEX.md).

365-day rollouts, long-term held-out-site validation, daily-model physical-
parameter gradient acceptance (Gate D2), and product acceptance remain later
milestones. Teacher Gate D1 does not validate the neural model's gradients.
GPU supports training; CPU/GPU inference performance and the final speedup
must be established by workload-level measurements.

## Branches and Research Status

- `main` is the frozen, user-facing half-hour PFT14 Teacher baseline.
- `research/daily-coarse-graining` contains the complete Teacher plus
  research-only data capture, dataset generation, and neural-surrogate code.
  It remains separate while the surrogate is experimental. Capture hooks
  default to off, and ordinary Teacher mode is required to stay behaviorally
  aligned with `main`; see `docs/branch-alignment-20260802.md`.

Start with [`docs/START_HERE.md`](docs/START_HERE.md), then use
[`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) for the current work order and
[`docs/CODE_MAP.md`](docs/CODE_MAP.md) for repository ownership.
[`docs/current-status.md`](docs/current-status.md) records detailed evidence;
[`docs/DOCUMENT_STATUS.md`](docs/DOCUMENT_STATUS.md) identifies active contracts
and historical reports. The long research `HANDOFF.md` is a chronology, not the
active roadmap. Update current facts in place instead of appending old work orders.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `jax_orchidee/` | Production JAX model, driver, process modules, and CLI runners |
| `research/daily_coarse_graining/` | Daily neural operators, Teacher-data interfaces, losses, training, and rollout research |
| `configs/` | Portable PFT14 paper-case configuration |
| `manifests/` | External data contract and 669-landpoint metadata |
| `scripts/hpc/` | Explore1000 Teacher, data preparation, neural training, and diagnostic launchers |
| `scripts/dev/` | Oracle, audit, diagnosis and performance tooling; some helpers are reused by research code |
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

Explore1000 uses a CentOS 7-compatible pinned environment rather than the
general lock file. See
[`docs/deployment-explore1000.md`](docs/deployment-explore1000.md).

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

The public CLI below runs the physical half-hour Teacher. It does not select
the experimental daily neural model.

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
- Missing Teacher processes are not approximated or hidden by calibration.
  The daily surrogate is an explicit learned approximation with separate
  accuracy, physical-budget, and gradient acceptance requirements.
- Discrete states compare exactly. Floating-point gates use explicit,
  field-aware tolerances recorded under `docs/source_audits`.

## Source and Licensing

The JAX repository license and citation metadata must be selected by the
project owner before public release. The bundled Fortran source tree has no
verified top-level redistribution license in this checkout; exclude it from a
public release unless redistribution permission is confirmed.

See `docs/installation.md`, `docs/data-layout.md`, `docs/running.md`, and
`docs/deployment-explore1000.md` for deployment details.
`docs/release-checklist.md` records the remaining license, citation,
cluster-review, and 669-point acceptance gates.
