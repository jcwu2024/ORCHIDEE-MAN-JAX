# Code Map

Last updated: 2026-08-03.

This page explains where code belongs and which paths are production,
research, evidence, or generated assets.

## Branches

| Branch | Role | Current restriction |
| --- | --- | --- |
| `main` | User-facing PFT14 half-hour Teacher and canonical Teacher core | Do not add experimental daily-model code |
| `research/daily-coarse-graining` | Identical Teacher core plus data and neural research | Keep `jax_orchidee/` identical to `main` |

The branches remain separate because research code is not a user-facing
product. Their production Teacher packages are synchronized: there is no
`jax_orchidee/` difference. The earlier six-file difference and its numerical
admission evidence are preserved in `branch-alignment-20260802.md` and
`teacher-branch-parity-acceptance.md`.

## Production Teacher

| Path | Responsibility |
| --- | --- |
| `jax_orchidee/cli/` | Public `orchidee-jax` command |
| `jax_orchidee/runners/` | Multiyear and landpoint acceptance runners |
| `jax_orchidee/driver/` | Forcing, static data, restart, orchestration, and compiled day/block boundaries |
| `jax_orchidee/sechiba/` | Surface exchange, energy, water, thermal, snow, routing, and SECHIBA lifecycle |
| `jax_orchidee/stomate/` | Daily vegetation/carbon processes, restart, and modelout |
| `jax_orchidee/parameters/` | Source-backed control, soil, and vertical parameter materialization |
| `jax_orchidee/coupled.py` | Source-order cross-module composition helpers |
| `jax_orchidee/ad_primitives.py` | Canonical derivative rules that preserve forward primals |

The Teacher now has a versioned PFT catalog and per-run stable-ID layout in
`jax_orchidee/parameters/pft_catalog.py`. Parameter rows are selected by
canonical Fortran PFT identity, and modelout/checkpoint/restart metadata retain
that identity. Restart readers parse stable metadata or require the explicit
legacy paper layout and remap declared PFT axes by ID. Production orchestration
resolves PFT-indexed parameters and PFT14 capability slots from that layout;
fixed-slot history handling is confined to an explicitly named legacy
Fortran-history boundary. The complete Teacher is still a PFT14 paper-case
product, and PFT2-PFT13 are deliberately `structural_only`. The catalog is an
extensibility boundary, not evidence that another PFT is scientifically
supported.

The normal user path is approximately:

```text
jax_orchidee.cli
  -> jax_orchidee.runners.multiyear
  -> jax_orchidee.driver.orchestration
  -> SECHIBA half-hour transitions
  -> retained daily STOMATE
  -> restart/modelout
```

No code under `research/` is imported by the ordinary Teacher path.

## Daily Research

`research/daily_coarse_graining/` currently mixes reusable infrastructure and
historical prototypes. Use the following classification.

### Reusable Teacher and data infrastructure

- `daily_markov_contract.py`: implemented v5 shard/state contract. It remains
  the reader contract for existing data, not the final surrogate architecture.
- `teacher_shards.py`, `teacher_production.py`, `teacher_pilot.py`: Teacher
  capture, restartable shard production, and production planning.
- `markov_dataset.py`, `production_training_protocol.py`: hash-bound dataset,
  split, statistics, and batching infrastructure.
- `teacher_data_product_admission.py`, `teacher_compatibility_gate.py`: dataset
  admission and compatibility checks.
- `carbon_budget_ownership.py`: source-backed carbon ownership audit useful for
  the new daily budget contract.
- `daily_process_axis_layout.py`: axis/process metadata that may inform the new
  interface, but does not itself implement PFT-generic computation.
- `daily_pft_interface.py`: active Gate B2 parser, named-channel packer, mask,
  aggregation, and PFT-axis validation boundary. Its machine contract is
  `manifests/coarse_graining/daily_neural_pft_interface_v1.json`.
- `physical_parameter_registry.py`: accepted source-hash and classification
  validator for sensitivity/inversion candidates. Its machine contract is
  `manifests/coarse_graining/physical_parameter_candidate_registry_v1.json`.
- `daily_flux_label_inventory.py`: active Gate C1 parser and v5 contract audit.
  Its machine contract is
  `manifests/coarse_graining/daily_flux_label_inventory_v1.json`.
- `daily_flux_capture.py`, `constrained_daily_replay.py`: accepted Gate C2
  capture serialization, bounded inventory/tendency updates, and budget
  audits. They define the non-neural admission boundary for the new operator.

### Diagnostic and Oracle assets

- `ok_leak_*`, `replay_ceiling.py`, `counterfactual_teacher_diagnostic.py`,
  and `canonical_teacher_reentry.py` diagnose or replay Teacher boundaries.
- Their 48-step paths may be used as Oracles. They must not become final daily
  inference dependencies.

### Historical or rejected neural prototypes

- `canonical_*`, `structured_canonical_daily_model.py`,
  `axis_process_coupled_daily_model.py`, `causal_carbon_*`,
  `rollout_stability_*`, `persistence_baseline.py`,
  `supervised_learnability_pilot.py`, and `synthetic_operator_cost.py` record
  earlier experiments.
- These files remain reproducibility evidence. None is the accepted final
  daily model, and several contain PFT14-specific compaction or indexing.
- Do not extend one of them as the new architecture without an explicit
  decision recorded in `NEXT_STEPS.md`.

### Active final daily implementation

There is no accepted neural implementation yet. The specification is
`docs/research/daily_coarse_graining/conservative_daily_process_operator_v1.md`.
The PFT interface, state/flux labels, and non-neural updater are frozen. New
neural implementation code should be placed in a clearly named package under
`research/daily_coarse_graining/`; it must compose with the accepted updater.
The bounded implementation sequence and acceptance tests are in
`docs/research/daily_coarse_graining/gate_e1_daily_operator_skeleton.md`.

## Scripts

| Path | Classification | Use |
| --- | --- | --- |
| `scripts/dev/` | Development/evidence | Fortran micro-oracles, source-ledger audits, diagnostics, parity probes, and performance tools |
| `scripts/hpc/` | Scheduler launchers | Explore1000 environment, Teacher production, dataset jobs, and historical experiment launchers |
| `scripts/validation/` | Reserved | User-oriented validation wrappers when promoted |
| `scripts/trace/` | Reserved | Trace tooling when promoted |

The hundreds of files under `scripts/dev/` are not model source and are not a
new contributor's reading list. Start from a named gate, then use the script
named by that gate's documentation.

Current reusable HPC families are the environment bootstrap scripts and the
`teacher_*` production/acceptance launchers. Launchers named for rejected
architectures remain historical until deliberately reactivated.

## Tests

| Path | Use |
| --- | --- |
| `tests/unit/` | Fast model, contract, audit, and research unit tests |
| `tests/parity/` | External-data integration and Fortran/reference parity |
| `tests/fortran_oracles/` | Fortran Oracle support assets |

Test names mirror their owner or evidence script. A passing historical neural
test means the old experiment is reproducible, not that its architecture is
accepted.

## Contracts and Evidence

| Path | Use |
| --- | --- |
| `configs/` | Portable user configuration |
| `configs/pft_catalogs/` | Versioned PFT identities, traits, parameter ownership, capabilities, and layouts |
| `manifests/` | Machine-readable data, split, landpoint, and experiment identities |
| `docs/source_audits/` | Fortran provenance and equivalence evidence |
| `docs/research/daily_coarse_graining/` | Architecture decisions and dated research evidence |
| `docs/porting/pft14_extensible_design.md` | Teacher-side PFT catalog/capability target and current limitations |
| `fortran_source/ORCHIDEE/` | Local Fortran source truth, excluded from public Git pending license review |
| `fortran_run_scripts/paper_250919/` | Historical paper runtime protocol truth |

## External and Generated Data

`data/`, `reference/`, `outputs/`, `traces/`, and `runtime/` contain external
or generated assets. They are not imported source packages. Large contents do
not belong in Git; manifests and small README files describe their identities.

Ignored root files such as `calendar.mod` and `extracted_oracle.mod` are local
Fortran compiler by-products, not project modules.
