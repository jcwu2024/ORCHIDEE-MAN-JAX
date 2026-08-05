# Documentation Status

Last updated: 2026-08-05.

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
| `research/daily_coarse_graining/gate_e2_typed_sidecar_data_preparation.md` | Active Gate E2 local data contract/evidence; production pilot pending | Reading or producing supplemental typed labels |
| `research/daily_coarse_graining/failed_architecture_lessons.md` | Active negative-design authority | Selecting or reviewing a neural architecture or objective |
| `research/daily_coarse_graining/development_standard.md` | Active policy | Adding a research experiment |
| `deployment-explore1000.md` | Active platform instructions | Running on Explore1000 |
| `installation.md`, `data-layout.md`, `running.md` | Active user docs | Installing or running the Teacher |

## Conditional Contracts

- `research/daily_coarse_graining/daily_markov_contract_v5.md` is active for
  reading existing v5 Teacher shards. It is superseded as the final neural
  boundary by the conservative daily operator decision.
- `research/daily_coarse_graining/teacher_dataset_generation.md` remains the
  production and provenance contract for the existing Teacher data product.
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
