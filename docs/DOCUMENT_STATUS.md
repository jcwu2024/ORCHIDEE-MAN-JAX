# Documentation Status

Last updated: 2026-08-11.

Use this page to decide what must be read and what is historical evidence.

## Authority Order

1. `START_HERE.md`: entry point and reading order.
2. `NEXT_STEPS.md`: current work order and acceptance gates.
3. `CODE_MAP.md`: code ownership and active/historical classification.
4. `current-status.md`: detailed accepted facts and experiment chronology.
5. Named contracts and architecture decisions: durable interface rules.
6. Dated reports: immutable evidence under their original commit and data.

When a dated report conflicts with `NEXT_STEPS.md`, the dated report describes
an earlier experiment and does not control current work.

## Active Documents

| Document | Status | Read when |
| --- | --- | --- |
| `NEXT_STEPS.md` | Current operation authority | Always |
| `CODE_MAP.md` | Current repository map | Always |
| `current-status.md` | Detailed status authority | Looking up accepted evidence |
| `branch-alignment-20260802.md` | Completed Gate A audit | Checking branch provenance |
| `teacher-branch-parity-acceptance.md` | Accepted Gate A decision | Checking canonical Teacher identity |
| `porting/pft14_extensible_design.md` | Accepted Gate B structural design; new PFT science remains unsupported | Extending Teacher PFT support |
| `research/daily_coarse_graining/conservative_daily_process_operator_v1.md` | Active architecture; non-neural boundary accepted | Designing the daily neural model |
| `research/daily_coarse_graining/daily_pft_interface_v1.md` | Frozen Gate B2 contract | Assembling PFT-axis inputs or adding parameter/trait channels |
| `research/daily_coarse_graining/daily_flux_label_inventory_v1.md` | Frozen Gate C1 contract | Reading daily label ownership |
| `research/daily_coarse_graining/gate_c2_constrained_replay_v1.md` | Accepted Gate C2 evidence | Implementing the conservative daily operator |
| `research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md` | Accepted canonical-Teacher gradient evidence | Adding physical parameters or planning inversion |
| `research/daily_coarse_graining/physical_parameter_candidate_registry_v1.md` | Accepted source-driven candidate inventory | Selecting sensitivity or inversion parameters |
| `research/daily_coarse_graining/gate_e1_daily_operator_skeleton.md` | Accepted Gate E1 implementation evidence | Using the true-daily operator boundary |
| `research/daily_coarse_graining/gate_e1_architecture_data_readiness_review.md` | Accepted Gate E1 architecture/data decision | Starting Gate E2 data preparation or candidate admission |
| `research/daily_coarse_graining/coherent_teacher_dataset_generation_v1.md` | Accepted Gate E2 v6 production contract | Reproducing or auditing formal Teacher training data |
| `research/daily_coarse_graining/coherent_teacher_pilot_20260806.md` | Accepted bounded production evidence | Sizing or reviewing coherent Teacher production |
| `research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md` | Accepted full v6 data evidence | Starting or auditing neural training |
| `research/daily_coarse_graining/storage_cleanup_20260807.md` | Current external-asset archive map | Locating or restoring historical v5 workers |
| `research/daily_coarse_graining/failed_architecture_lessons.md` | Active negative-design authority | Selecting or reviewing a neural architecture or objective |
| `research/daily_coarse_graining/first_neural_process_head_candidate_v1.md` | First matched seed completed; comparison closed without promotion | Reviewing the first formal neural process-head design and outcome |
| `research/daily_coarse_graining/first_neural_process_head_smoke_acceptance_20260808.md` | Accepted local and real-v6 pre-training smoke | Reviewing candidate admission or planning the next bounded gate |
| `research/daily_coarse_graining/first_neural_process_training_preflight_20260808.md` | Accepted historical preflight; first pair later completed and the remaining array was stopped | Reviewing timing, cost, or preflight provenance |
| `research/daily_coarse_graining/first_neural_process_head_first_pair_decision_20260808.md` | Controlling stop decision after one matched pair | Reviewing why neither attention nor factorized arm is being continued |
| `research/daily_coarse_graining/long_horizon_biomass_productivity_objective_v1.md` | Active source ledger and implemented 1/7/30-day objective contract | Reviewing successor rollout loss and scientific metrics |
| `research/daily_coarse_graining/daily_operator_parameter_routing_v1.md` | Implemented local parameter-ownership boundary used by the successor | Routing physical parameters or planning Gate D2 |
| `research/daily_coarse_graining/successor_process_head_architecture_options_v1.md` | Accepted decision package; attention/graph pair selected and frozen | Reviewing the selected successor mechanism and control |
| `research/daily_coarse_graining/successor_process_head_local_foundation_20260809.md` | Accepted local synthetic foundation; real-v6 lifecycle later exposed an incomplete fast boundary | Reviewing implementation evidence before the boundary diagnosis |
| `research/daily_coarse_graining/successor_fast_boundary_coverage_20260809.md` | Accepted blocking diagnosis; superseded for current action by the owner schema | Tracing why the original 101-output boundary failed |
| `research/daily_coarse_graining/successor_fast_owner_reconstruction_contract_v1.md` | Accepted 160/160 owner/materializer boundary; decoder extension implemented | Auditing the 79-value-head plus mask ownership contract |
| `research/daily_coarse_graining/successor_fast_owner_decoder_local_acceptance_20260809.md` | Accepted local 181-field decoder/statistics foundation | Reviewing decoder semantics or local gates |
| `research/daily_coarse_graining/successor_fast_owner_statistics_acceptance_20260809.md` | Accepted formal v6 79-value statistics artifact | Reviewing statistics identity, counts, or hashes |
| `research/daily_coarse_graining/successor_fast_owner_gpu_probe_acceptance_20260809.md` | Accepted real-v6 GPU execution/gradient gate; bounded-fit preparation is next | Reviewing GPU evidence, threshold policy, or the next gate |
| `research/daily_coarse_graining/successor_small_fit_runner_contract_v1.md` | Active bounded-small-fit contract; free single-update preflight accepted | Changing sampling, objective, optimizer, lifecycle scope, or paid-gate policy |
| `research/daily_coarse_graining/successor_small_fit_gpu_probe_acceptance_20260809.md` | Accepted free GPU replay, coefficient, gradient, and one-update gate | Reviewing bounded-fit evidence or implementing the paid production runner |
| `research/daily_coarse_graining/successor_small_fit_runner_reuse_acceptance_20260810.md` | Historical accepted same-process reuse evidence; its 20-hour paid projection is superseded | Reviewing why repeated host preparation was rejected |
| `research/daily_coarse_graining/successor_small_fit_prepared_batch_acceptance_20260810.md` | Accepted prepared-batch and hot-reuse evidence; cross-process cache and old paid envelope rejected | Reviewing the prepared boundary or rejected cache assumption |
| `research/daily_coarse_graining/successor_small_fit_compile_reduction_acceptance_20260810.md` | Accepted cold/hot timing evidence; avoidable compilation removed | Reviewing the timing basis for the bounded 500-update resource plan |
| `research/daily_coarse_graining/successor_small_fit_production_resource_plan_20260810.md` | Historical accepted staged resource contract; both stages complete | Reviewing resources, costs, or recovery policy |
| `research/daily_coarse_graining/successor_small_fit_prepared_500_acceptance_20260810.md` | Accepted immutable 500-entry CPU input to the completed fit | Reviewing prepared input identity |
| `research/daily_coarse_graining/successor_small_fit_checkpoint_handoff_v1.md` | Active checkpoint amendment; production dual checkpoint accepted | Consuming the checkpoints in rollout |
| `research/daily_coarse_graining/successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md` | Historical accepted free-GPU serialization evidence; no checkpoint was retained by that probe | Reviewing pre-fit persistence evidence |
| `research/daily_coarse_graining/successor_small_fit_paid_admission_failure_20260811.md` | Accepted failed-run diagnosis; schedule-ownership defect closed by the production admission | Reviewing job 14517771 or the reason for the admission repair |
| `research/daily_coarse_graining/successor_production_fit_asset_admission_20260811.md` | Historical accepted exact-production admission used by the completed fit | Reviewing production asset identity |
| `research/daily_coarse_graining/successor_small_fit_500_checkpoint_acceptance_20260811.md` | Active accepted 500-update fit and dual checkpoint product | Preparing or admitting the 7/30-day rollout screen |
| `research/daily_coarse_graining/successor_7_30_rollout_screen_package_20260810.md` | Active frozen post-fit acceptance package; dual checkpoints are now available | Preparing or reviewing the free smoke and formal bounded 7/30-day screen |
| `research/daily_coarse_graining/successor_7_30_free_smoke_acceptance_20260811.md` | Active accepted free-smoke execution evidence; not promotion eligible | Preparing or reviewing the formal 96-window bounded screen |
| `research/daily_coarse_graining/successor_7_30_formal_prepared_acceptance_20260811.md` | Active accepted formal prepared-panel input; paid screen awaits approval | Admitting or reviewing the formal 96-window bounded screen |
| `research/daily_coarse_graining/successor_7_30_formal_cache_timeout_20260811.md` | Active infrastructure-failure diagnosis; no model classification | Validating the no-persistent-cache repair before a replacement paid screen |
| `research/daily_coarse_graining/development_standard.md` | Active policy | Adding a research experiment |
| `deployment-explore1000.md` | Active platform instructions | Running on Explore1000 |
| `installation.md`, `data-layout.md`, `running.md` | Active user docs | Installing or running the Teacher |

## Conditional Contracts

- `research/daily_coarse_graining/daily_markov_contract_v5.md` is active for
  reading existing v5 Teacher shards. It is superseded as the final neural
  boundary by the conservative daily operator decision.
- `research/daily_coarse_graining/teacher_dataset_generation.md` remains the
  production and provenance contract for the historical v4/v5 Teacher data
  product. New formal data use the coherent generation contract above.
- `docs/source_audits/*.yaml`, `oracle_families/`, and
  `production_families/` are machine/evidence contracts used by their audit
  commands. They are not general onboarding material.
- `PROJECT_MANIFEST.md` owns repository and external-asset boundaries.

## Historical Documents

- `research/daily_coarse_graining/HANDOFF.md` is a chronological research log.
  It is preserved for provenance and is no longer the current-operation
  authority.
- Dated `*_202607*.md` and `*_202608*.md` research reports are immutable
  experiment snapshots.
- `research/daily_coarse_graining/gate_e2_typed_sidecar_data_preparation.md`
  records the rejected immutable-parent sidecar design and server attempts.
- `daily_boundary_v0_draft.yaml`, `daily_markov_contract_v3.md`, and
  `daily_markov_contract_v4.md` are superseded contracts retained for
  migration provenance.
- Stage-numbered plans and completion notes under `docs/source_audits/` record
  the PFT14 porting campaign. They do not define the daily-surrogate roadmap.
- Neural launchers and manifests named for rejected `canonical`, `structured`,
  `rollout_stability`, or `causal_carbon` experiments remain reproducibility
  assets, not active defaults.

Do not delete historical material merely to make the tree shorter. Navigate it
through this status index and `CODE_MAP.md`; move or prune only in a dedicated
archive change that preserves links and provenance hashes.
