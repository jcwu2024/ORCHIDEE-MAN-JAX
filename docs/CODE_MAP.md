# Code Map

Last updated: 2026-08-21.

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

The current independently capable baseline is frozen by
`manifests/coarse_graining/strong_baseline_axis_aligned_process_expert_v1.json`.
`daily_operator/strong_baseline_contract.py` owns its seven-expert field,
source, and axis admission; `strong_baseline_architecture.py` owns the
49,750,681-parameter float32 forcing/factorized-axis/expert/output network;
`strong_baseline_process_head.py` owns the Gate-E binding; and
`strong_baseline_loss.py` owns equal-expert then equal-label process fidelity.
`daily_operator/output_transform.py` contains the shared float64 constrained
output transformation used by both the strong baseline and historical
successor; `daily_operator/parameter_routing.py` owns their shared explicit
fast-parameter factors. The local structural evidence is
`strong_baseline_local_acceptance_20260816.md`.
`daily_operator/strong_baseline_qualification.py` owns exact shuffled-epoch
padding, the no-clip AdamW schedule, decay exclusions, and reusable
real-sample-weighted exact-gradient accumulation for the one-point gate.
`scripts/hpc/run_strong_baseline_one_point_qualification.py` owns full v6
point-year assembly, training, detailed expert/label/field loss-floor evidence,
corrected one-day science metrics, diagnostic-state serialization, and hard
counts. The frozen exact-gradient tail is launched on free `gln01` by
`scripts/hpc/run_strong_baseline_one_point_loss_floor_diagnostic.sh` and is
specified in `strong_baseline_one_point_loss_floor_diagnostic_v1.md`. The
completed diagnostic and its dominant-label attribution are recorded in
`strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md`; the
earlier failed qualification is classified in
`strong_baseline_one_point_qualification_result_20260816.md`. The earlier
zero-update admission is implemented by
`scripts/hpc/run_strong_baseline_gpu_admission.py` and its free-`gln01`
launcher and has passed.

`research/daily_coarse_graining/fast_owner_delta_representation_audit.py`
owns the bounded base-shard audit of constrained endpoint reachability, global
normalization conditioning, and one-boundary source reconstruction. Its v2
logic reconstructs absolute endpoints before testing physical ranges. It reads
one accepted v6 base shard plus the existing train/train statistics without
reading typed values, regenerating data, expanding 48 forcing steps, or
running a subdaily state scan. Its unit tests are in
`tests/unit/test_fast_owner_delta_representation_audit.py`, and its scientific
classification is
`dominant_endpoint_source_representation_audit_20260817.md`.

`research/daily_coarse_graining/fast_owner_exact_reconstruction_panel.py`
expands the admitted source formulas to an explicit non-test point-year panel,
fails closed on spatial or temporal test selections, and aggregates exact
field errors without reconstructing subdaily forcing or state scans. Its unit
tests are in `tests/unit/test_fast_owner_exact_reconstruction_panel.py`.

`research/daily_coarse_graining/thermal_coefficient_exact_reconstruction_panel.py`
owns the complete daily-boundary reconstruction of the ten THERMOSOIL and
ENERBIL coefficient fields. It calls the existing source-backed hydrology,
thermal, final-state, and explicit-snow sequence and is tested by
`tests/unit/test_thermal_coefficient_exact_reconstruction_panel.py`.

`research/daily_coarse_graining/doc_diffusion_boundary_baseline_panel.py`
compares day-start, day-end, and endpoint-trapezoid source-kernel magnitudes
with the immutable v6 cumulative diffusion label on non-test point-years. It
does not claim exact reconstruction. Its unit tests are in
`tests/unit/test_doc_diffusion_boundary_baseline_panel.py`, and the accepted
result is `doc_diffusion_boundary_baseline_result_20260817.md`.

`research/daily_coarse_graining/hydrology_canopy_boundary_audit.py` owns the
non-test, no-scan daily-boundary audit of HYDROL stress diagnostics,
tile-to-grid bare-evaporation beta, `temp_hydro`, dynamic `nroot`, and signed
canopy storage/debt. Its focused tests are in
`tests/unit/test_hydrology_canopy_boundary_audit.py`; the source result and the
subsequently accepted v2 mechanism basis are in
`hydrology_canopy_boundary_audit_result_20260817.md`.

`manifests/coarse_graining/strong_baseline_mechanistic_output_review_v1.json`
is the historical proposal-only complete 181-field classification used by
`mechanistic_output_architecture_review_v1.md`. Its coverage/count regression is
`tests/unit/test_strong_baseline_mechanistic_output_review.py`.

`manifests/coarse_graining/strong_baseline_mechanistic_output_v1.json` is the
historical first executable output-mechanism contract.
`manifests/coarse_graining/strong_baseline_mechanistic_output_v2.json` is the
active hydrology/canopy contract. Under `daily_operator/`,
`mechanistic_output_contract.py` owns the 161/20 learned/exact schema split,
`mechanistic_output_reconstruction.py` owns pre/post-conservative-update exact
and baseline/residual materialization, including support-aware `nroot` and the
signed canopy endpoint-to-transfer conversion. `mechanistic_output_loss.py`
excludes exact fields from optimization, supervises canopy endpoint/status
coordinates, and retains exact-field diagnostics, while
`mechanistic_process_head.py` binds the mechanism to the strong Attention
backbone. `scripts/hpc/run_mechanistic_output_gpu_admission.py` owns bounded
real-v6 forward/loss/gradient/state/budget admission. The shared one-point
runner selects this path with `--mechanistic-output`; its explicit
`--training-smoke` mode distinguishes the one-epoch execution gate from the
full memorization gate. `daily_operator/hierarchical_coordinate_attention.py`
owns the v3 litter/POC/DOC parent: its four coordinate-query stages,
field-balanced process contexts, field-specific heads, candidate-local stable
bounded transforms, fresh parameter initialization, and 60-million-parameter
cap. `mechanistic_process_head.py` exposes its separate process-head class;
the shared one-point runner selects it only with
`--hierarchical-coordinate-attention`. For this candidate the runner preserves
logical batch size 8 while accumulating exact gradients through size-2
microbatches and derives coverage from the first and final actual updates;
other model paths retain direct-batch execution.
`daily_operator/coordinate_coupled_doc_attention.py` owns the rejected v4
successor evidence: exact full/compact-PFT coordinate gathers, the five-token
aligned DOC bank, the joint signed conservative endpoint materializer, fresh
initialization, and its process-output path. `mechanistic_process_head.py`
exposes the v4 class, while the shared runner and launcher select it only with
`--coordinate-coupled-doc-transition` /
`COORDINATE_COUPLED_DOC_TRANSITION=1` and reuse the size-2 exact gradient
microbatch boundary.
`scripts/hpc/slurm_coordinate_coupled_doc_transition_v4_one_point.sh` owns the
completed paid single-V100 formal v4 one-point resource envelope; the failed
candidate must not be rerun. The older
`slurm_mechanistic_output_v2_one_point.sh` remains historical.

`daily_operator/doc_status_branch_magnitude_attention.py` owns the rejected v5
DOC endpoint evidence. It reuses v4's aligned source bank and replaces only
the final signed scalar with three status logits and separate incoming and
outgoing magnitude heads. `daily_operator/doc_status_branch_magnitude_loss.py`
owns the class-macro status, branch-coordinate, physical endpoint, and revised
inventory-label composition. `mechanistic_process_head.py` exposes the V5
process head. The shared runner and launcher select it only with
`--doc-status-branch-magnitude` / `DOC_STATUS_BRANCH_MAGNITUDE=1`, enforce the
5/10-epoch no-checkpoint gate, and reject simultaneous v3/v4 selection.
`scripts/hpc/slurm_doc_status_branch_magnitude_v5_early_screen.sh` owns the
completed one-V100/two-hour envelope. The candidate failed its frozen gate and
must not be continued or rerun. Its result and physical-materialization
diagnosis are recorded in
`doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md`.

`daily_operator/doc_target_diagnostics.py` owns the zero-training DOC endpoint
partition, inverse-target, reconstruction, normalized-Jacobian, and loss-share
statistics. Its frozen contract is
`manifests/coarse_graining/doc_endpoint_target_loss_geometry_audit_v1.json`.
`scripts/hpc/audit_doc_endpoint_target_loss_geometry.py` assembles the accepted
v6 point-year and writes the report, while
`scripts/hpc/run_doc_endpoint_target_loss_geometry_audit.sh` restricts that
execution to free CPU test nodes. The accepted result is
`doc_endpoint_target_loss_geometry_audit_result_20260820.md`; it rejects the
joint head and opens mechanism discussion, not another training run.

`daily_operator/nonlinear_projection.py` owns the active host-side bounded
SLSQP subproblem, exact violation merit, and candidate acceptance for the
zero-retention low-dimensional projection audit. The JAX reconstruction,
maintenance microbatch gradients, frozen parameter basis, sequential JVPs,
and exact finite trials remain in
`scripts/hpc/probe_successor_parameter_geometry.py`; the free-GPU boundary is
`scripts/hpc/run_successor_low_dimensional_nonlinear_projection.sh`. These are
optimizer-geometry research tools and do not alter the Attention process head
or production inference path.

`daily_operator/attention_mixed_continuation.py` owns the completed
candidate-only contract, compressed cosine schedule, scientific
classification, and diagnostic recoverable single-Attention state. The executable and
single-GPU boundary are
`scripts/hpc/run_successor_attention_mixed_continuation.py` and the matching
`.sh` launcher; `scripts/hpc/slurm_successor_attention_mixed_continuation.sh`
owns the bounded one-GPU `gnall` resource envelope. They reuse the accepted
update-1000 Attention parameters and prepared 1001-2000 assets; they do not
alter the architecture or Teacher. The completed run is classified in
`successor_attention_mixed_annealed_continuation_result_20260816.md`; the
final state is not promotion eligible.

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
conversion, hard gates, AdamW update, staged-v3 contract admission and
deterministic milestone-validation panel selection. The original machine
contract is `manifests/coarse_graining/successor_small_fit_runner_v1.json`;
the current staged policy is
`manifests/coarse_graining/successor_training_staged_v3.json`.
`daily_operator/successor_training_assets.py` independently admits the frozen
v3 coefficient and validation-panel assets.
`scripts/dev/freeze_successor_training_assets.py` derives the lower-quantile
coefficients from the existing v2 report's raw measurements and freezes the
new non-test panel on a CPU node. Its acceptance is
`successor_staged_training_assets_acceptance_20260812.md`.
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
`scripts/hpc/probe_successor_checkpoint_handoff.py` owns the serialization-only
free-GPU preflight. It stages synthetic GPU-origin parameter and optimizer
trees in a temporary runtime directory, verifies exact reload, removes the
temporary checkpoint and writes only a JSON evidence report.
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
`scripts/hpc/probe_successor_production_admission.py` and its CPU-test-node
wrapper now own the required fail-fast admission of the exact production
schedule, manifest, first artifact, coefficient source and checkpoint handoff.
They execute zero training updates and publish no checkpoint. This gate was
added after job `14517771` exposed a source-versus-consuming schedule ownership
bug; its diagnosis is `successor_small_fit_paid_admission_failure_20260811.md`.
The repaired gate passed on `test01` at consumer commit `0ed9891`; the accepted
report SHA256 is `50d090e6...18ea089`, recorded in
`successor_production_fit_asset_admission_20260811.md`.

`daily_operator/successor_rollout_screen.py` owns the frozen post-fit 7/30-day
non-test inventory, deterministic matched-window plan, additive
biomass/productivity metrics, exact hard counts, dual-checkpoint admission and
candidate/control classification. Its machine contract is
`manifests/coarse_graining/successor_7_30_rollout_screen_v1.json`.
`scripts/hpc/prepare_successor_rollout_screen.py` creates hash-bound host
artifacts on free CPU test nodes; `successor_rollout_screen_prepared.py` owns
their atomic persistence and admission. `run_successor_rollout_screen.py`
consumes both admitted checkpoints at the canonical daily-operator boundary.
The shell wrappers bind free CPU/GPU execution, while
`slurm_successor_rollout_screen.sh` is an unsubmitted 96-window formal-screen
proposal. The frozen package is documented in
`successor_7_30_rollout_screen_package_20260810.md`.

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
