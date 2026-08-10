# Code Map

Last updated: 2026-08-10.

This page explains where code belongs and which paths are production,
research, evidence, or generated assets.

## Branches

| Branch | Role | Current restriction |
| --- | --- | --- |
| `main` | User-facing PFT14 half-hour Teacher and canonical Teacher core | Do not add experimental daily-model code |
| `research/daily-coarse-graining` | Identical Teacher core plus data and neural research | Keep `jax_orchidee/` identical to `main` |

The branches remain separate because research code is not a user-facing
product. Their scientific Teacher code is synchronized. The research branch
currently adds one default-off compiled-output projection hook in
`jax_orchidee/driver/orchestration.py` for coherent data capture; its
capture-off/capture-on A/B is accepted, and it must be promoted to `main`
without modification before the next branch-parity claim. The earlier
six-file difference and its numerical admission evidence are preserved in
`branch-alignment-20260802.md` and `teacher-branch-parity-acceptance.md`.

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
- `teacher_data_release.py`: freezes and verifies the exact Teacher tree,
  configuration, PFT catalog, contracts, and producer source identity.
- `teacher_shards.py`, `teacher_production.py`, `teacher_pilot.py`: Teacher
  capture, restartable shard production, and production planning. Their v3
  plan/v5 manifest path co-generates base, typed, and checkpoint assets under
  one release and worker transaction.
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
- `typed_sidecar.py`: lossless typed storage, supplemental collation,
  train-only statistics, historical v1 reader, and coherent v2 single-manifest
  reader.
- `typed_capture_masks.py`: shared capability/defined-mask owner for coherent
  and historical typed storage.
- `typed_sidecar_production.py`: historical immutable-parent Gate E2 runner.
  It remains reproducibility tooling and must not produce the formal dataset.

### Diagnostic and Oracle assets

- `ok_leak_*`, `replay_ceiling.py`, `counterfactual_teacher_diagnostic.py`,
  and `canonical_teacher_reentry.py` diagnose or replay Teacher boundaries.
- Their 48-step paths may be used as Oracles. They must not become final daily
  inference dependencies.

### Historical or rejected neural prototypes

Read
`docs/research/daily_coarse_graining/failed_architecture_lessons.md` before
reusing any historical neural code. It records the evidence-backed rejection
reason, retained lesson, reusable infrastructure, and conditions for reopening
an idea.

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

The Gate E1 executable boundary is in
`research/daily_coarse_graining/daily_operator/`. It owns named PyTrees,
native-record encoding, grouped state/PFT features, constrained process-head
parameterizations, the pure conservative updater, unique canonical input
assembler, reusable canonical retained-tail adapter, composition root, and
graph/label audits. Its minimal linear head is a plumbing fixture, not a
selected neural architecture. Gate E1 is accepted. The specification is
`docs/research/daily_coarse_graining/conservative_daily_process_operator_v1.md`;
the bounded implementation and evidence are in
`docs/research/daily_coarse_graining/gate_e1_daily_operator_skeleton.md`.
The controlling review and 47-label readiness matrix are in
`docs/research/daily_coarse_graining/gate_e1_architecture_data_readiness_review.md`.
Gate E2 now uses one frozen coherent Teacher release. Its active contract is
`coherent_teacher_dataset_generation_v1.md`; its six-point production pilot
passed, and the accepted measurements are in
`coherent_teacher_pilot_20260806.md`. The formal data product is manifest v6,
typed contract v3, and canonical release v2. Full 669-landpoint, 1961-2010
production and admission are complete. Architecture selection and training
now have a frozen first-candidate contract in
`first_neural_process_head_candidate_v1.md` and
`manifests/coarse_graining/first_neural_process_head_candidate_v1.json`.
`daily_operator/training_data.py` owns the host-side coherent-v6 input and
named-target adapter; `daily_operator/training_loss.py` owns equal-label macro
supervision and local one-day hard metrics.
`daily_operator/training_statistics.py` owns vectorized train/train target
extraction, mergeable masked moments, PFT-shared output/input scales, and the
hash-bound statistics artifact. Its CLI and proposed paid CPU launcher are
`scripts/hpc/fit_first_neural_process_statistics.py` and
`scripts/hpc/slurm_first_neural_process_statistics.sh`.
`daily_operator/neural_architecture.py` and
`daily_operator/neural_process_head.py` own the first implemented Transformer
candidate and exactly parameter-matched control. Their local and bounded
real-v6 pre-training smoke is accepted in
`first_neural_process_head_smoke_acceptance_20260808.md`. Host-only sampling,
optimization, validation, and checkpoint ownership are in
`research/daily_coarse_graining/first_neural_process_training_run.py`; the
GPU entry points and paid array launcher are in `scripts/hpc/`. The final
no-checkpoint streaming/validation preflight is accepted in
`first_neural_process_training_preflight_20260808.md`. One formal matched seed
pair later completed and was stopped for practical equivalence without
promotion; its controlling decision is
`first_neural_process_head_first_pair_decision_20260808.md`. The full-v6 process statistics artifact is accepted from eight
retained train/train partitions and an independent `test01` merge/load audit;
its evidence and hashes are in `first_neural_process_head_candidate_v1.md`.
Both arms enter through the existing Gate E1 composition boundary. The old
immutable-parent packet and archived loose v5 workers are historical evidence.

`daily_operator/long_horizon_objective.py` owns the architecture-neutral
1/7/30-day cumulative productivity, terminal carbon-stock, signed-bias, AGB/BGB
derivation, and hard-gate foundation. Its source ledger and objective contract
are `long_horizon_biomass_productivity_objective_v1.md` and the matching
manifest. It does not select or implement the successor neural architecture.

`daily_operator/parameter_routing.py` owns the active ownership-aware split of
the full operator parameter carrier. It supplies named `vcmax25` and
maintenance-temperature factors to replaced fast processes and dynamically
routes `alloc_min`, `residence_time`, and the retained `vcmax25` consumer into
canonical STOMATE. Its contract is
`daily_operator_parameter_routing_v1.md`. The closed first Transformer A/B did
not consume this boundary and is not retroactively promoted.

The selected successor decision is
`successor_process_head_architecture_options_v1.md`, with frozen machine
identity in `manifests/coarse_graining/successor_process_head_candidate_v1.json`.
`daily_operator/successor_contract.py` owns host-side contract admission;
`successor_architecture.py` owns source tokens, masks, matched attention/graph
arms, decoders, and named fast parameter factors; `successor_process_head.py`
owns the Gate-E process binding; and `successor_training.py` owns free daily
rollout, predicted PFT-view refresh, the 5+7 trajectory adapter, combined
objective, and one train/train gradient-ratio preflight. The local acceptance
record is `successor_process_head_local_foundation_20260809.md`.
`daily_operator/neural_architecture.py` remains the closed first-pair
implementation and is not the successor default.

The blocking real-v6 boundary diagnosis is
`successor_fast_boundary_coverage_20260809.md`. Its successor owner schema is
`manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json`, with
the full review table in
`successor_fast_owner_reconstruction_contract_v1.md`.
`scripts/dev/build_successor_fast_owner_reconstruction_contract.py` owns
generation and validation of the 160-leaf/2,855-column registry. It reads only
accepted v6 manifest metadata, locks the v6 and source-contract identities,
and does not read Teacher rows. `daily_operator/fast_owner_reconstruction.py`
owns the identity-locked JAX materializer; `canonical_retained_tail.py`
consumes it only when explicitly bound; and
`scripts/hpc/run_successor_process_head_canonical_gate.py` owns the all-Teacher
`fast_owner` stage. Contract SHA256 `4db5f828...fdfc4c9` and free-`gln01`
report SHA256 `f3ae02d0...f360e084` establish cold/ordinary/restart `160/160`
acceptance. `daily_operator/successor_fast_owner_decoder.py` owns the
inference-safe 181-field schema, delta-to-physical transform, explicit
loss/domain/source metadata, and strict normalization composition.
`successor_fast_owner_training.py` is host-only and owns named single-day
targets plus the 48-label value/BCE loss. `successor_fast_owner_statistics.py`
is host-only and owns vectorized v6 target extraction and mergeable 79-value
statistics. `scripts/hpc/fit_successor_fast_owner_statistics.py` is their
fit/merge CLI; `slurm_successor_fast_owner_statistics.sh` produced the
accepted 8,591,565-row artifact under job `14515519`. Local decoder evidence
is in `successor_fast_owner_decoder_local_acceptance_20260809.md`; formal
statistics evidence is in
`successor_fast_owner_statistics_acceptance_20260809.md`.
`scripts/hpc/run_successor_fast_owner_gpu_probe.py` owns the accepted real-v6
two-arm forward/loss/full-gradient and canonical-transition probe;
`run_successor_fast_owner_gpu_probe.sh` binds the Explore1000 Singularity GPU
runtime and fail-closed identities. Its acceptance record is
`successor_fast_owner_gpu_probe_acceptance_20260809.md`.

`daily_operator/successor_small_fit.py` owns the derived bounded-fit contract,
deterministic train/train annual-shard schedule, coherent typed-v6 batch
assembly, 16-row one-day anchor, two-window 30-day objective, Teacher replay
conversion, hard gates and AdamW update. The machine contract is
`manifests/coarse_graining/successor_small_fit_runner_v1.json`.
`scripts/hpc/run_successor_small_fit_probe.py` and its shell wrapper execute
the accepted no-checkpoint, one-update-per-arm free-`gln01` resource preflight.
Its evidence is
`successor_small_fit_gpu_probe_acceptance_20260809.md`.
`scripts/hpc/run_successor_small_fit.py` owns the contract-bound matched
500-update runner and requires a prepared manifest plus the accepted
frozen-coefficient contract and its source report. Successful production fits
publish both arms atomically through
`daily_operator/successor_fit_checkpoint.py`; failed and submaximal runs write
no checkpoint. The producer/consumer identity amendment is
`manifests/coarse_graining/successor_small_fit_checkpoint_handoff_v1.json`,
documented in `successor_small_fit_checkpoint_handoff_v1.md`.
`scripts/hpc/prepare_successor_small_fit_batches.py` and its CPU wrapper own
resumable, hash-bound batch/day-input packaging; `successor_small_fit_prepared.py`
owns artifact serialization and admission. `run_successor_small_fit.sh` owns
the Explore1000 GPU container boundary. The former paid Slurm wrapper is
removed. Current prepared-batch/hot-reuse evidence and the rejected
cross-process-cache assumption are in
`successor_small_fit_prepared_batch_acceptance_20260810.md`; the earlier
runner evidence remains in `successor_small_fit_runner_reuse_acceptance_20260810.md`.
`daily_operator/successor_frozen_coefficients.py` admits the accepted
train/train coefficient contract and rejects runner, schedule, candidate,
owner, first-artifact or source-report drift. The current compile-reduction
evidence is `successor_small_fit_compile_reduction_acceptance_20260810.md`.
`slurm_prepare_successor_small_fit_500.sh` owns the proposed four-worker
`cnmix` preparation envelope; `slurm_successor_small_fit_500.sh` owns the
separately gated one-GPU same-process fit envelope. Their accepted partition
smoke, recovery boundary and cost contract are in
`successor_small_fit_production_resource_plan_20260810.md`.
The complete four-worker 500-entry artifact product and independent full-hash
admission are recorded in
`successor_small_fit_prepared_500_acceptance_20260810.md`; its server execution
identity remains bound to producer HEAD `d493699`. A later clean consumer
worktree may admit it only through the exact checkpoint-handoff contract.

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
