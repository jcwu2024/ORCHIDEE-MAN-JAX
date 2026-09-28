# Code Map

Updated: 2026-09-26. A bounded ownership map, not an exhaustive import inventory.
Current operations are in [NEXT_STEPS.md](NEXT_STEPS.md).

## Source and Teacher

| Path | Responsibility |
| --- | --- |
| `fortran_source/ORCHIDEE/` | Fortran source truth; process/branch provenance |
| `fortran_run_scripts/paper_250919/` | Paper runtime protocol truth |
| `jax_orchidee/cli/`, `runners/`, `driver/` | Public Teacher, forcing/restart orchestration, compiled lifecycle |
| `jax_orchidee/sechiba/`, `stomate/` | Source-backed half-hour processes and retained daily physics |
| `jax_orchidee/parameters/` | Parameter/PFT catalog and stable-identity ownership |
| `jax_orchidee/coupled.py`, `ad_primitives.py` | Source-order composition and canonical derivatives |
| `reference/`, `traces/`, `docs/source_audits/` | Reference/provenance/branch evidence, not surrogate logic |

Teacher and research remain separate products. `main` carries the user-facing
Teacher; `research/daily-coarse-graining` adds research code. The accepted core
synchronization is recorded at `5910ba9` / `10030eb`; it is not a claim that
all later branch contents are identical. PFT14 is supported within declared
scope; PFT2-PFT13 are `structural_only` and fail closed if activated.

## Current daily model and training

Paths below are relative to `research/daily_coarse_graining/daily_operator/`.

| Module | Owns |
| --- | --- |
| `model.py::bind_daily_operator` | Canonical input -> process head -> conservative updater -> retained-tail composition |
| `shared_transformer.py` | Shared masked backbone and common coordinate decoder |
| `shared_process_head.py::SharedTransformerProcessHead` | Neural outputs bound to typed physical process boundary |
| `shared_persistent_outputs.py`, `shared_cimean.py` | Persistent endpoint coordinates and source-backed annual-Ca identity |
| `shared_trajectory_loss.py`, `shared_training.py` | Shared science terms and daily objective |
| `shared_recurrent_training.py`, `rollout_loss.py` | Joint daily/T7/T30 objective and public native checkpointed rollout AD |

Under `scripts/hpc/`, `shared_transformer_runtime.py` binds accepted prepared
inputs/calibration; `shared_learning_plan.py` owns seeded ordering/panels;
`run_shared_learning.py` composes full-network updates;
`shared_learning_monitor.py` evaluates fixed scientific/physical panels;
`shared_daily_microbatch.py` preserves sample-mean objective accumulation.
`canonical_training_runtime.py` and `retained_training_runtime.py` own data
bootstrap and retained templates; `training_artifact_io.py` and
`runtime_diagnostics.py` provide shared host-side helpers. Current consumers
use these modules rather than importing retired experimental runners.
These executable paths are not authorization to resume training.

Current candidate: 64,884,481 parameters, backbone 768/eight blocks, decoder
384/two blocks. Physical state supplies day-to-day memory. All candidate heads
enter the same canonical binding, updater and retained tail; no final daily
inference path may execute a 48-step state scan.

## Data and reusable dependencies

Modules in `research/daily_coarse_graining/` owning Teacher release, shards,
typed storage/masks, dataset/statistics, PFT interface, flux inventory/replay,
physical-parameter registry and target-oriented optimization remain reusable
infrastructure. A filename containing `canonical`, `target_oriented`, `v5`,
`successor` or an old architecture name does **not** prove dead code.
Check live imports and contract responsibilities before moving/removing it.
Use [WORKSPACE_INVENTORY.md](research/daily_coarse_graining/WORKSPACE_INVENTORY.md)
for the current dependency/classification inventory and its reproducible check.

Machine identities live in `manifests/coarse_graining/`. Formal training uses
the accepted coherent **v6 fixed** release, not historical v5 rows or an
immutable-parent sidecar join. Runtime/Teacher cache data are external assets;
do not materialize one file per day/update for the full data release.

## Execution diagnostics, not model defaults

`scripts/hpc/probe_shared_serialized_updates.py` and
`probe_shared_serialized_dual.py` compare saved-binary full proposals and
optimizer parity. `probe_shared_combined_recurrent.py` admits bounded T7/T30
execution changes. The `shared_*execution.py` adapters and saved daily/recurrent
profilers are opt-in function-preserving experiments, not a new model family.

`probe_shared_ffn_precision.py` captures actual FFN inputs;
`probe_shared_ffn_algorithm.py` tests native dot policies/forward/input and
parameter VJPs. These are isolated precision diagnostics, not production
mixed precision or custom AD. Latest evidence is linked from NEXT_STEPS.

`shared_decoder_expert_execution.py` owns opt-in union dynamic projections;
the default remains field execution. `probe_shared_dynamic_projection.py`
tests that component; `probe_shared_neural_precision.py` owns the staged native
daily AD diagnostic. `probe_shared_neural_transpose_lower.py`,
`probe_shared_union_integrated.py` and `probe_shared_union_replay.py` preserve
lowering, actual-loss parity and cache-replay experiments respectively. These
are diagnostic interfaces, not the pending production micro4 speed comparison.

## History, tests and generated files

- Retired neural experiment implementations are recoverable from the source
  snapshot linked in HISTORY_INDEX. Scientific reports and retained Oracle/replay
  evidence remain available. Read
  [failed architecture lessons](research/daily_coarse_graining/failed_architecture_lessons.md)
  before reopening a rejected architecture.
- `tests/unit/`: lightweight model/contract/research checks; `tests/parity/`:
  external-data and Fortran comparisons; `tests/fortran_oracles/`: Oracle assets.
- `scripts/dev/`: source/branch audits and local diagnostics; `scripts/hpc/`:
  current launchers, shared runtime helpers and retained diagnostics.
- `outputs/`: new generated reports/assets; remote runtime belongs under the
  declared project runtime root. Neither is source truth.
- [HISTORY_INDEX.md](research/daily_coarse_graining/HISTORY_INDEX.md) routes
  named historical questions without loading the whole chronology.

This bounded map does not establish exhaustive dependency closure or dead code.
Use graph coverage and exact imports/source for the particular change.
