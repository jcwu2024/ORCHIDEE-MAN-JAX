# Code Map

Last updated: 2026-09-06.

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

`daily_operator/target_oriented_contract.py` owns the current architecture-
neutral output hierarchy, independent carbon-flux/stock identities, seven-
group one-day state anchor, 7/30-day independent-basis science objective,
low-weight necessary-process supervision, and explicit total-loss composition.
Its frozen machine contract is
`manifests/coarse_graining/target_oriented_daily_operator_v1.json`; the
scientific decision and absolute 7/30/365-day gates are in
`target_oriented_daily_operator_contract_v1.md`. Its lightweight regression is
`tests/unit/test_daily_operator_target_oriented_contract.py`. It supersedes
the older equal-field promotion policy.

`daily_operator/target_oriented_attention.py` owns the selected 83,755,173-
parameter source-routed/domain-refined Attention backbone, five science-token
interactions, and category-specific coordinate decoders.
`daily_operator/target_oriented_process_head.py` binds those 161 learned
fields to the existing mechanistic reconstruction, conservative updater, and
retained exact tail. Their frozen mechanism/capacity record is
`manifests/coarse_graining/target_oriented_attention_v1.json` and
`target_oriented_attention_v1.md`; focused local qualification is in
`tests/unit/test_daily_operator_target_oriented_attention.py`. The local gate
and zero-update real-v6 GPU admission are accepted; training and rollout remain
conditional. The accepted admission runner and free-`gln01` launcher are
`scripts/hpc/run_target_oriented_attention_gpu_admission.py` and
`scripts/hpc/run_target_oriented_attention_gpu_admission.sh`; its evidence is
`target_oriented_attention_gpu_admission_20260821.md` and
`outputs/target-oriented-attention-gpu-admission-7c6371c-report.json`.

`daily_operator/target_oriented_training.py` owns the frozen bounded-training
contract loader, deterministic same-year 28-landpoint/point-year schedule,
prepared batch construction, target scales, clipped AdamW proposal, and exact
microbatch sufficient-statistics reductions for one-day state, 7/30-day
science, and necessary process supervision. Its machine contract is
`manifests/coarse_graining/target_oriented_bounded_training_v1.json`, its
decision record is `target_oriented_bounded_training_v1.md`, and its focused
regressions are
`tests/unit/test_daily_operator_target_oriented_training.py`,
`tests/unit/test_daily_operator_target_oriented_training_statistics.py`, and
`tests/unit/test_daily_operator_target_oriented_prepared_batch.py`.
The post-preflight state-loss amendment is
`manifests/coarse_graining/target_oriented_state_anchor_amendment_v1.json` and
`target_oriented_state_anchor_amendment_20260821.md`; it owns the residual
`other` eligibility mask, low-variance native-unit scale fallback, and
declared-dynamic `rveget` reporting policy without changing the architecture
or prepared-batch selection.
`scripts/hpc/target_oriented_training_prepared.py` owns serialization,
`prepare_target_oriented_training_batches.py` owns free-CPU real-v6
preparation, and `run_target_oriented_full_step_preflight.py` owns the
zero-retained-update free-GPU measurement. Their free-node launchers are the
same basenames with `.sh` extensions. The first amended representative probe
is recorded in `target_oriented_representative_gradient_probe_20260823.md`.
The intervening zero-variance repair in
`strong_baseline_architecture.py::_masked_summary` is valid but does not lie
on the network-parameter derivative path. The actual nonfinite derivative was
the zero-stock division in
`mechanistic_output_reconstruction.py::signed_endpoint_to_transfers`.
`scripts/hpc/run_target_oriented_output_boundary_vjp.py` and its free-`gln01`
shell wrapper own the learned-output-coordinate localization and complete
Attention VJP. The repaired boundary is finite through all 83,755,173
parameters. Job `14652590` passed the complete repaired single-batch real-v6
gradient probe; its runner emits host-only completion timing after each
objective-gradient source so a scheduler timeout cannot hide the last
completed stage.
`scripts/hpc/slurm_target_oriented_full_step_preflight.sh` owns the completed
one-V100 eight-batch execution boundary; the shared runner reused the accepted
prepared manifest and the compilation cache populated by the representative
probe. Job `14659152` completed it in 27 minutes; its accepted report and the
corrected hot-step resource measurement are recorded in
`target_oriented_full_step_preflight_acceptance_20260824.md`.

`scripts/hpc/run_target_oriented_training.py` owns the completed formal
update-1-to-250 retained-update, recovery, stage-checkpoint, conditioning, and
96-window validation lifecycle. Its paid boundary is
`scripts/hpc/slurm_target_oriented_training_250.sh`, and checkpoint schemas
are owned by
`research/daily_coarse_graining/daily_operator/target_oriented_checkpoint.py`.
Job `14664525` completed that stage; its formal evidence is
`target_oriented_formal_training_250_result_20260825.md`. The same runner now
owns the explicit update-251-to-1000 checkpoint-handoff and absolute-index
continuation path. `target_oriented_checkpoint.py` validates and restores the
parent parameters plus AdamW state, while `target_oriented_training_prepared.py`
and `merge_target_oriented_training_batches.py` admit a partial absolute-update
range without copying the first 250 artifacts. The stage checkpoint is not
passed through the recovery reader or treated as a new initialization.

`scripts/hpc/audit_target_oriented_flux_gradient_variance.py` owns the completed
zero-update 30-day flux-gradient batch-size diagnosis at the update-1000
checkpoint. It reuses optional acceptance-statistic hooks in
`run_target_oriented_full_step_preflight.py`, forms exact family and global
Gram matrices for 4/8/16/32-window aggregates, and never invokes AdamW or
writes a checkpoint. Its free-node launcher is
`scripts/hpc/run_target_oriented_flux_gradient_variance_audit.sh`, its focused
regression is
`tests/unit/test_target_oriented_flux_gradient_variance_audit.py`, and its
frozen decision record is
`target_oriented_flux_gradient_variance_audit_v1.md`. The accepted free-GPU
result and the controlling component-resolved next gate are in
`target_oriented_flux_gradient_variance_audit_result_20260826.md`.

`daily_operator/target_oriented_flux_basis.py` and
`manifests/coarse_graining/target_oriented_flux_basis_amendment_v1.json` own
the explicit component-resolved flux basis without mutating the historical v1
objective. `target_oriented_contract.py` and `target_oriented_training.py`
accept that explicit basis for scale projection, science statistics, exact
microbatch accumulation, and separate flux/stock linearized gradients.
`scripts/hpc/audit_target_oriented_component_resolved_geometry.py` owns the
completed zero-update five-source/eight-group geometry audit, and
`scripts/hpc/run_target_oriented_component_resolved_geometry_audit.sh` owns
its free-`gln01` execution boundary. The frozen decision record is
`target_oriented_component_resolved_flux_basis_v1.md`.

`daily_operator/target_oriented_flux_virtual_step.py` owns the completed
protected direct-step contract and classifier.
`scripts/hpc/audit_target_oriented_flux_protected_virtual_step.py` owns the
16-gradient common direction, independently realized Gram verification, and
seven exact discarded parameter-step evaluations; its free-node launcher is
`scripts/hpc/run_target_oriented_flux_protected_virtual_step.sh`. The result
closes this raw-loss realization and opens a separate residual-tail and
sampling-unit audit; it does not authorize another direct-step grid.

`daily_operator/target_oriented_flux_residual_tail.py` owns the frozen
window-loss concentration, bounded influence, exact Gram-direction, and
tail-mechanism classification logic. The existing full-step factory exposes
standalone 7/30-day window-flux gradients for this audit without changing the
training objective. `scripts/hpc/audit_target_oriented_flux_residual_tail.py`
owns the 96-window train/train execution and group-local streamed Grams; its
free-node launcher is
`scripts/hpc/run_target_oriented_flux_residual_tail_audit.sh`. This path
takes no optimizer or parameter step and reads no validation artifact value.
The accepted result is
`target_oriented_flux_residual_tail_result_20260826.md`; it identifies an
extreme maintenance-respiration Jacobian amplification and makes a focused
free-rollout-versus-Teacher-state gradient-path attribution the next gate.

`daily_operator/target_oriented_flux_gradient_path.py` owns the frozen four-
window contract, complete fine-family partition, Gram summary, and bounded
path classifier. The full-step factory exposes weighted per-field free and
Teacher-state gradient functions without changing its retained training
steps. `scripts/hpc/audit_target_oriented_flux_gradient_path.py` executes the
24 GPP/maintenance/growth gradients, reconstructs the aggregate gradient, and
reports coarse/fine/head attribution. Its free-`gln01` launcher is
`scripts/hpc/run_target_oriented_flux_gradient_path_attribution.sh`. It takes
no optimizer or parameter step and cannot edit the architecture. The accepted
result is `target_oriented_flux_gradient_path_result_20260826.md`: it locates
the extreme-gradient path in the predicted snowheat state carried between
rollout days. The next code boundary is a separate zero-training snowheat-only
carry intervention; it must resolve the leaf from
`DailyMarkovContract.state_leaves` and must not alter the general rollout or
final inference path.

`daily_operator/target_oriented_snowheat_recurrence.py` owns the frozen
intervention contract, exact Markov-leaf resolution, diagnostic one-slice
Teacher carry, and bounded causal classifier. The full-step factory exposes
the optional intervention gradients only when its diagnostic state key is
explicitly supplied; the ordinary free and Teacher-state paths are unchanged.
`scripts/hpc/audit_target_oriented_snowheat_recurrence.py` owns the eight
maintenance gradients and complete family/head attribution, and
`scripts/hpc/run_target_oriented_snowheat_recurrence_intervention.sh` owns its
free-`gln01` execution boundary. The intervention is forbidden in training
and final inference.

The accepted snowheat-only result is
`target_oriented_snowheat_recurrence_result_20260827.md`. It confirms the
7-day causal path and locates the unresolved 30-day gradient in the snowtemp
state head. The next code boundary reuses the same diagnostic rollout with
two exact Markov leaves and evaluates only the two 30-day windows.

`daily_operator/target_oriented_snow_thermal_recurrence.py` owns the frozen
two-window coupled contract and causal classifier.
`scripts/hpc/audit_target_oriented_snow_thermal_recurrence.py` reuses the
diagnostic multi-leaf rollout to compute exactly two complete maintenance
gradients against the hash-bound snowheat-only parent. Its free-node launcher
is `scripts/hpc/run_target_oriented_snow_thermal_recurrence_intervention.sh`.

The accepted coupled result is
`target_oriented_snow_thermal_recurrence_result_20260827.md`. Its post-result
owner audit found that the two `hydrol` leaves have prognostic `diffuco`
mirrors. `daily_operator/target_oriented_snow_thermal_alias_recurrence.py`
owns the four-slice contract, fast-owner mirror verification, and bounded
classifier. `scripts/hpc/audit_target_oriented_snow_thermal_alias_recurrence.py`
owns the two complete mirror-aware gradients, and
`scripts/hpc/run_target_oriented_snow_thermal_alias_recurrence_intervention.sh`
is its free-`gln01` boundary. This diagnostic cannot enter training or final
inference.

The accepted mirror-complete result is
`target_oriented_snow_thermal_alias_recurrence_result_20260827.md`. It proves
the omitted `diffuco` snowheat path and locates the remaining gradient in the
direct maintenance head. Source tracing excludes the reset
`resp_maint_part` accumulator and identifies persisted `resp_maint` as the
only bounded maintenance-state recurrence question.
`daily_operator/target_oriented_maintenance_state_recurrence.py` owns the
frozen five-slice contract and causal classifier.
`scripts/hpc/audit_target_oriented_maintenance_state_recurrence.py` owns the
two complete gradients, and
`scripts/hpc/run_target_oriented_maintenance_state_recurrence_intervention.sh`
is its free-`gln01` boundary. It does not edit the maintenance source mask or
architecture and cannot enter training or final inference.

The accepted result is
`target_oriented_maintenance_state_recurrence_result_20260827.md`. It rejects
persisted `resp_maint` recurrence as an explanation and closes further
state-by-state Teacher carry. The next code boundary must operate on the
existing independent maintenance final head and exact 7/30-day losses without
editing the Attention architecture, input mask, or source routing.

`daily_operator/target_oriented_maintenance_head_conditioning.py` owns the
frozen 769-parameter final-head boundary, loss-scaled step calibration, and
science/state/stock classifier. The shared full-step builder exposes two
head-only 7/30-day reverse functions; all other parameters are primal parent
values, not differentiated variables.
`scripts/hpc/audit_target_oriented_maintenance_head_conditioning.py` owns the
16-gradient geometry and exact virtual evaluations.
`scripts/hpc/run_target_oriented_maintenance_head_conditioning_audit.sh` is
the free-`gln01`, zero-training launcher. It cannot retain a proposal or enter
final inference.

The accepted result is
`target_oriented_maintenance_head_conditioning_result_20260827.md`. It finds a
common direction across all 16 maintenance objectives but no finite step that
passes every frozen protection. The current implementation boundary therefore
stops before output-architecture edits. The existing coordinate-aligned
Attention body remains open; the next design must discuss a structured
maintenance materialization instead of silently continuing the shared
`768 -> 1` final head.

`daily_operator/target_oriented_source_shaped_maintenance.py` owns the current
source-shaped maintenance materialization and its two bounded correction
branches. `daily_operator/target_oriented_source_shaped_screen.py` owns the
frozen train/train data separation and 20%/1% science classifier.
`scripts/hpc/run_target_oriented_source_shaped_train_train_zero_panel.py`
evaluates the eight-group zero-update panel, and its same-named shell launcher
is the free-`gln01` zero-panel boundary.
`scripts/hpc/run_target_oriented_source_shaped_one_update_admission.py` owns
the accepted p90 branch-gradient calibration, one ephemeral branch-only AdamW
proposal, frozen-parent identity check, and post-update forward/hard gate. Its
shell launcher is free-`gln01` only and cannot write a checkpoint or recovery.
`scripts/hpc/run_target_oriented_source_shaped_branch_autodiff_admission.py`
owns the accepted direct branch-subtree versus extracted-full-gradient
equivalence, timing, live-memory, optimizer-proposal, and post-proposal hard
gate. Its shell launcher is free-`gln01` only and cannot write a checkpoint or
recovery. The shared full-step builder now exposes the unchanged complete
objective with an optional differentiated parameter subtree; existing
full-gradient callers retain their previous behavior.

`daily_operator/target_oriented_source_shaped_checkpoint.py` owns the compact
branch-only atomic recovery and update-100 checkpoint. It persists no retained
parent coordinate and binds reconstruction to the exact parent hash.
`scripts/hpc/run_target_oriented_source_shaped_branch_training.py` owns the
frozen artifacts `252..351`, 100 direct-branch updates, ten-update recovery,
and final eight-group train/train screen. Its same-named shell wrapper owns the
GPU container boundary, while
`scripts/hpc/slurm_target_oriented_source_shaped_branch_training_100.sh` owns
the completed one-V100 paid resource envelope used by job `14701638`.

`daily_operator/target_oriented_source_shaped_coupling.py` owns the completed
update-0/update-100 free-versus-full-Teacher comparison, raw carbon-bias
identity, and causal classifier. Its accepted result is
`target_oriented_source_shaped_coupling_attribution_result_20260828.md`.
`daily_operator/target_oriented_source_owner_state_bridge.py` owns the active
four-owner contract and strict ten-cell classifier. The shared full-step
factory exposes Teacher-carry acceptance forwards only when explicit
diagnostic state keys are supplied. The zero-training executable and free-
`gln01` boundary are
`scripts/hpc/audit_target_oriented_source_owner_state_bridge.py` and
`scripts/hpc/run_target_oriented_source_owner_state_bridge.sh`. They reuse the
accepted free and full-Teacher report values, execute only the bridge, create
no optimizer, and cannot write a checkpoint or enter final inference.
The accepted result is
`target_oriented_source_owner_state_bridge_result_20260828.md`: four of ten
protected cells reverse, so broader recurrent state or Attention context
remains in scope.

`daily_operator/target_oriented_distribution_aware_maintenance.py` owns the
accepted 24-coordinate, two-block maintenance Attention adapter and exact
source-backed output materialization. `daily_operator/target_oriented_distribution_aware_training.py`
owns its completed Teacher-guard calibration, frozen 250-update adapter-only
contract, combined-gradient rule, and relative plus aggregate-absolute science
classifier. `scripts/hpc/calibrate_target_oriented_distribution_aware_guard.py`
owns the accepted zero-update calibration, while
`scripts/hpc/run_target_oriented_distribution_aware_one_update_admission.py`
owns the accepted ephemeral free-`gln01` AdamW admission. Neither free runner
can write a checkpoint, read validation, or authorize paid training.
`daily_operator/target_oriented_distribution_aware_checkpoint.py` owns compact
adapter-only atomic recovery and update-250 persistence.
`scripts/hpc/run_target_oriented_distribution_aware_training.py` owns the
exact 250-update combined-gradient fit plus update-0/update-250 train/train
screens; its shell wrapper owns the container boundary, and
`scripts/hpc/slurm_target_oriented_distribution_aware_training_250.sh` owns
the completed one-V100 paid envelope. None makes the retained Attention parent
trainable. Job `14856147` produced the diagnostic update-250 checkpoint.

`daily_operator/target_oriented_distribution_aware_gpp_bridge.py` owns the
active zero-training Teacher-GPP contract loader, the single learned-coordinate
substitution before the retained exact tail, the matched-GPP free-state scan,
and the frozen three-way classifier. Optional forward hooks live in
`scripts/hpc/run_target_oriented_full_step_preflight.py`.
`scripts/hpc/audit_target_oriented_distribution_aware_gpp_bridge.py` owns the
two required update-250 train/train modes and reuses the accepted ordinary
free-rollout screen; its same-basename shell wrapper owns the completed
free-`gln01` container boundary. The accepted audit scopes the next
architecture discussion to GPP. These files cannot train, change architecture,
write a checkpoint, read validation values, or enter final inference.

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

The current structured-GPP successor keeps its endpoint-specific training
contracts in `daily_operator/target_oriented_structured_gpp_training.py` and
its compact specialist checkpoint format in
`daily_operator/target_oriented_structured_gpp_checkpoint.py`. The post-warm
train-only coefficient policy is owned by
`daily_operator/target_oriented_structured_gpp_joint_calibration.py` and
`manifests/coarse_graining/target_oriented_structured_gpp_joint_gradient_calibration_v1.json`.
`scripts/hpc/calibrate_target_oriented_structured_gpp_joint_gradients.py`
measures direct, 7-day, 30-day, and combined specialist gradients without an
optimizer or checkpoint write. Its free-`gln01` container boundary is
`scripts/hpc/run_target_oriented_structured_gpp_joint_gradient_calibration.sh`.
The accepted calibrated coefficients are consumed by
`daily_operator/target_oriented_structured_gpp_joint_update.py` and
`manifests/coarse_graining/target_oriented_structured_gpp_joint_update_preflight_v1.json`.
`scripts/hpc/run_target_oriented_structured_gpp_joint_update_preflight.py`
owns the completed inherited-AdamW discarded proposal and its realized-step
checks; `run_target_oriented_structured_gpp_joint_update_preflight.sh` owns
the free-`gln01` execution boundary. The subsequent single-artifact rejection
opens grouped gradient analysis in
`daily_operator/target_oriented_structured_gpp_joint_heterogeneity.py` and
`manifests/coarse_graining/target_oriented_structured_gpp_joint_gradient_heterogeneity_audit_v1.json`.
`scripts/hpc/audit_target_oriented_structured_gpp_joint_gradient_heterogeneity.py`
owns the zero-update `1/2/4/8` effective-batch audit, with its free-node
boundary in `run_target_oriented_structured_gpp_joint_gradient_heterogeneity.sh`.
The accepted audit selects batch size 4. Its finite-step gate is owned by
`daily_operator/target_oriented_structured_gpp_joint_batch4.py` and
`manifests/coarse_graining/target_oriented_structured_gpp_joint_batch4_preflight_v1.json`.
`scripts/hpc/run_target_oriented_structured_gpp_joint_batch4_preflight.py`
executes two independent discarded step-251 proposals from the same warm-start
parameter and AdamW state; its shell wrapper is the free-`gln01` boundary.
Neither proposal can write a checkpoint or authorize paid training. The
accepted result opens the bounded production owner
`daily_operator/target_oriented_structured_gpp_joint_training.py` with machine
contract
`manifests/coarse_graining/target_oriented_structured_gpp_joint_training_v1.json`.
`scripts/hpc/run_target_oriented_structured_gpp_joint_training.py` owns the
four-artifact gradient accumulation, inherited AdamW update, atomic recovery,
train-only epoch panel, early stop, and last-passing checkpoint selection.
`run_target_oriented_structured_gpp_joint_training.sh` is its container
boundary, and `slurm_target_oriented_structured_gpp_joint_training.sh` is the
unsubmitted paid `gnall` launcher. The frozen design is documented in
`target_oriented_structured_gpp_joint_training_v1.md`.

The post-broad-training independent screen is owned by
`daily_operator/target_oriented_structured_gpp_fresh_screen.py` and
`manifests/coarse_graining/target_oriented_structured_gpp_fresh_screen_v1.json`.
The selector excludes consumed point-years using metadata only.
`scripts/hpc/prepare_target_oriented_structured_gpp_fresh_validation.py`
materializes the resulting 96 host artifacts on free CPU test nodes, and
`run_target_oriented_structured_gpp_fresh_screen.py` evaluates Teacher-state
daily GPP plus canonical 7/30-day free rollouts from the broad update-1000
checkpoint. Their shell wrappers are direct free-node boundaries and contain
no `sbatch` submission.

The failed screen's read-only causal attribution is owned by
`daily_operator/target_oriented_structured_gpp_fresh_attribution.py` and
`manifests/coarse_graining/target_oriented_structured_gpp_fresh_attribution_v1.json`.
`scripts/hpc/audit_target_oriented_structured_gpp_fresh_attribution.py`
executes ordinary free rollout, Teacher-day-start local GPP, and the existing
diagnostic Teacher-GPP bridge on the same 96 windows. It reports exact GPP
residual decomposition and signed growth/NPP propagation without an optimizer,
parameter step, or checkpoint write. Its shell wrapper is a direct free-
`gln01` boundary and contains no `sbatch` submission.
`scripts/hpc/finalize_target_oriented_structured_gpp_fresh_attribution.py`
is the hash-bound CPU-only correction boundary for the completed raw numerical
report. It recomputes formal-report reproduction, attribution classification,
and integrity from existing JSON statistics without loading Teacher data or
executing the model.

The completed train-only feedback-scope audit is owned by
`daily_operator/target_oriented_structured_gpp_state_feedback_attribution.py`
and
`manifests/coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_v1.json`.
`scripts/hpc/audit_target_oriented_structured_gpp_state_feedback.py` evaluates
free rollout, full Teacher day-start state, and six source-component
Teacher-carry modes on prepared updates `301`, `601`, and `901`.
`daily_operator/target_oriented_snowheat_recurrence.py` and
`run_target_oriented_full_step_preflight.py` provide the shared runtime-mask
rollout so the components reuse one compiled graph per horizon. The shell
wrapper is a direct free-`gln01` boundary with no `sbatch` path. The accepted
result selects only `slowproc_stomate_previous_step_state` as material and
keeps water as a monitored secondary path. Its evidence is
`target_oriented_structured_gpp_state_feedback_attribution_result_20260831.md`.

The approved coupled owner is
`daily_operator/target_oriented_coupled_gpp_carbon_feedback.py`. It derives
the 2,386-coordinate/107-field state layout from the accepted Markov and
strong-baseline contracts, owns the shared two-block PFT-query cross-attention
adapter and three structured-GPP residual injections, and preserves GPP-only
output ownership. `target_oriented_process_head.py` exposes its stable
process-head builder. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_carbon_feedback_v1.json`.
The local implementation and real-v6 zero-update GPU admission are accepted.
`scripts/hpc/run_target_oriented_coupled_gpp_carbon_feedback_gpu_admission.py`
owns the machine-readable admission, while its direct and Slurm shell wrappers
own free-`gln01` and bounded `gnall` launch boundaries. No persistent optimizer
or training owner exists for the candidate yet.

`daily_operator/target_oriented_gpp_state_reachability.py` owns the frozen
field attachment and exact-nonzero Jacobian union. Its machine contract is
`manifests/coarse_graining/target_oriented_gpp_state_reachability_v1.json`.
`scripts/hpc/audit_target_oriented_gpp_state_reachability.py` loads the exact
accepted candidate and computes each real-v6 forward-mode Jacobian through the
conservative update and canonical retained tail;
`run_target_oriented_gpp_state_reachability.sh` owns its free-`gln01` or
approved-`gnall` execution boundary, and
`slurm_target_oriented_gpp_state_reachability.sh` owns the bounded paid
envelope. The accepted result is
`target_oriented_gpp_state_reachability_result_20260901.md`. None of these
owners creates an optimizer or checkpoint.

`daily_operator/target_oriented_coupled_gpp_objective.py` owns the accepted
41-coordinate/16-field state anchor, the two-family trainable boundary, and
the frozen four-source coefficient rule. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_objective_calibration_v1.json`.
`scripts/hpc/calibrate_target_oriented_coupled_gpp_objective.py` owns the
eight-artifact zero-update calibration report; its direct and Slurm wrappers
own free-`gln01` and bounded `gnall` launch boundaries. No optimizer or
checkpoint is created. The accepted result is
`target_oriented_coupled_gpp_objective_calibration_result_20260901.md`.

`daily_operator/target_oriented_coupled_gpp_optimizer_preflight.py` owns the
frozen macro-objective aggregation and proposal policies; its machine
contracts are
`manifests/coarse_graining/target_oriented_coupled_gpp_optimizer_preflight_v1.json`
and the active exact-half-rate
`target_oriented_coupled_gpp_optimizer_backoff_v1.json`.
`scripts/hpc/run_target_oriented_coupled_gpp_optimizer_preflight.py` owns one
fresh-AdamW proposal and complete eight-artifact before/after evaluation. Its
direct and Slurm wrappers own free-`gln01` and bounded `gnall` execution. The
proposal and optimizer state are discarded; these files are not a training or
checkpoint owner. Job `14902356` rejected the `4e-6` finite step after complete
execution; `slurm_target_oriented_coupled_gpp_optimizer_backoff.sh` owns the
accepted `2e-6` execution envelope. Its result is
`target_oriented_coupled_gpp_optimizer_backoff_result_20260901.md`.
`daily_operator/target_oriented_coupled_gpp_bounded_training.py` now owns the
frozen target-blind eight-macro-update schedule, disjoint train-only monitor,
finite-step retention gate, milestones, and stop rules.
`daily_operator/target_oriented_coupled_gpp_checkpoint.py` owns atomic
two-family recovery and non-promotable stage checkpoints.
`scripts/hpc/run_target_oriented_coupled_gpp_bounded_training.py` owns both the
discarded production-equivalent runner admission and retained bounded stage;
its direct shell wrapper admits free `gln01` or allocated `gnall`, and its
Slurm wrapper freezes the proposed one-V100 paid envelope. The runner remains
conditional on local and free-GPU qualification.

`daily_operator/target_oriented_coupled_gpp_persistent_scale_training.py`
owns the frozen 100-update schedule, scale-selection policy, detailed monitor
gate, and checkpoint deadline. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_v1.json`.
`daily_operator/target_oriented_coupled_gpp_persistent_checkpoint.py` owns
atomic parameter-only recovery and milestone checkpoints without optimizer
state. `scripts/hpc/run_target_oriented_coupled_gpp_persistent_scale_training.py`
owns update-7 reconstruction, persistent common-direction updates, combined
7/30-day aggregate and per-key monitoring, recovery, and runner admission.
Its direct and Slurm wrappers own the free-`gln01` admission and the separately
authorized one-V100 `gnall` execution boundary.

`daily_operator/target_oriented_coupled_gpp_direct_generalization.py` owns the
frozen update-1-through-10 trajectory, first-adjacent-regression definition,
cosine normalization, and sign-based attribution policy. Its machine contract
is
`manifests/coarse_graining/target_oriented_coupled_gpp_direct_generalization_audit_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_direct_generalization.py` owns
the zero-retention trajectory replay and 8-training/8-monitor direct-gradient
geometry; its direct wrapper is free-`gln01` only. It does not read the
persistent recovery, repeat scale search, or own a checkpoint.

`daily_operator/target_oriented_coupled_gpp_direct_step_transfer.py` owns the
frozen update-2 scale ladder, four disjoint train-only GPP guard groups,
train-only selection, and post-selection monitor classification. Its machine
contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_audit_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_direct_step_transfer.py` owns
the one-direction finite-step evaluations; its free-`gln01` wrapper has no
paid submission path. It writes only the result report and never exposes the
fixed monitor before train-only scale selection.

`daily_operator/target_oriented_coupled_gpp_multi_group_direction.py` owns the
eight-constraint registry and post-selection classification for the current
same-state audit. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_multi_group_direction_audit_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_multi_group_direction.py`
accumulates five disjoint direct-GPP group means plus the scheduled state,
7-day, and 30-day gradients, solves the `8 x 8` common-direction problem, and
performs train-only finite-step selection before at most one monitor read. Its
direct wrapper is free-`gln01` only and retains no state.

`daily_operator/target_oriented_coupled_gpp_artifact_aware_direction.py` owns
the 43-constraint sample-level direction and train-only scale gate. Its machine
contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_artifact_aware_direction.py`
owns the accepted one-step audit over 40 individual direct-GPP artifacts and
three scheduled non-GPP objectives.

`daily_operator/target_oriented_coupled_gpp_artifact_aware_continuity.py` owns
the three disjoint train-only panels, consecutive-update classification, and
post-selection monitor-trajectory rule. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_artifact_aware_continuity.py`
recomputes the 43-gradient direction at each in-memory state and defers every
monitor read until all three train-only steps pass. Its free-`gln01` wrapper
writes one report and no parameters, optimizer state, recovery, or checkpoint.

`daily_operator/target_oriented_coupled_gpp_artifact_aware_persistent_training.py`
owns the six-update rotating-panel schedule and the update-4/update-7 monitor
gates. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.json`.
`daily_operator/target_oriented_coupled_gpp_artifact_aware_persistence.py`
owns immutable exact-start serialization, atomic parameter-only recovery, and
the final-gate-only diagnostic checkpoint.
`scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_persistent_training.py`
owns one-time start reconstruction, identity-exact no-replay resume, per-step
artifact-aware training, and detailed monitor execution. Its direct wrapper is
free-`gln01` only; no paid launcher is authorized.

`daily_operator/target_oriented_coupled_gpp_snowheat_causal.py` owns the
mirror-complete snowheat source resolution, bounded Day-5 causal classifier,
and existing GPP-factor log attribution. The completed executable
`scripts/hpc/audit_target_oriented_coupled_gpp_snowheat_causal.py` reconstructs
the rejected update-3 displacement in memory, runs control, treatment, and
matched-snowheat trajectories, and writes a report even when an engineering
subgate fails. Its wrapper is free-`gln01` only and never retains parameters.

`daily_operator/target_oriented_coupled_gpp_day5_crossed_attribution.py` owns
the four-cell parameter/state classifier, exact six-owner route masks,
owner-level materiality rule, and two-family effect decomposition.
`scripts/hpc/audit_target_oriented_coupled_gpp_day5_crossed_attribution.py`
owns the complete Day-5 crossed evaluations, matched owner and parameter
controls, and frozen interpolation paths. Its dedicated wrapper reuses the
shared free-`gln01` boundary and has no optimizer, checkpoint, or paid path.

`daily_operator/target_oriented_coupled_gpp_conditioned_attention.py` owns the
16 bounded per-block/head temperature parameters, conditioned trainable-tree
boundary, fixed-temperature mechanism reference, and continuity classifier.
The bounded cosine Q/K operation lives in
`daily_operator/target_oriented_structured_gpp.py`; the existing process-head
builder in `daily_operator/target_oriented_process_head.py` installs it only
for the two potential blocks. The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_conditioned_attention.py` owns
the real-v6 one-day and known-jump zero-update admission, and its dedicated
shell wrapper reuses the shared free-`gln01` boundary. It creates no optimizer,
writes no checkpoint, and has no paid launcher.

`daily_operator/target_oriented_coupled_gpp_conditioned_attention_discarded_update.py`
owns the active 43-constraint direction registry, architecture-aware step
calibration, nine-key science gates, fixed-monitor transfer gate, and exclusive
outcome classification. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1.json`.
`scripts/hpc/audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update.py`
loads the exact accepted update-2 recovery, constructs the conditioned
trainable tree, performs train-only direction and scale selection, and then
loads each fixed-monitor artifact once for paired initial/candidate evaluation.
Its dedicated wrapper is free-`gln01` only. It writes one report and no
parameters, optimizer state, recovery, or checkpoint.

`daily_operator/target_oriented_coupled_gpp_conditioned_attention_persistent_training.py`
owns the current three-update schedule, per-update architecture-aware
calibration, and terminal classification. Its machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.json`.
`daily_operator/target_oriented_coupled_gpp_conditioned_attention_persistence.py`
owns the immutable conditioned start, atomic parameter-only recovery, and
diagnostic update-5 checkpoint.
`scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.py`
recomputes 43 constraints at each state, performs train-only scale selection,
and admits retention only after both step-local and cumulative fixed-monitor
gates pass. Its direct wrapper is resumable and free-`gln01` only; no paid
launcher or product-promotion path is authorized.

`scripts/hpc/audit_conditioned_attention_update4_derivative.py` owns the
current zero-retention diagnostic after persistent update 4 failed. It
reconstructs only nonzero simplex support, records reverse/forward primal
agreement and gradient accumulation precision, and measures signed 30-day
finite changes on the scheduled training artifacts. Its wrapper reuses the
qualified free-node argument branch and never writes model parameters.

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
