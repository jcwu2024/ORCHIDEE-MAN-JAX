# Documentation Status

Last updated: 2026-08-02.

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
| `branch-alignment-20260802.md` | Open Gate A audit | Working on Teacher parity |
| `porting/pft14_extensible_design.md` | Active Teacher extensibility target, not yet implemented | Working on Gate B |
| `research/daily_coarse_graining/conservative_daily_process_operator_v1.md` | Active, not implemented | Designing the daily model |
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
