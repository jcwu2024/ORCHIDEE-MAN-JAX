# Research History: Read by Question

Updated: 2026-09-26. This is a small route into retained evidence, not a list of
every report or a current work queue. Read current entry pages first; historical
"running", "next" and paid-job proposals below describe their original dates.

## Select one evidence route

| Question / epoch | Start with | Scope / follow-up |
| --- | --- | --- |
| Why does Teacher branch parity count as accepted? | [teacher-branch-parity-acceptance.md](../../teacher-branch-parity-acceptance.md) | Scoped source correction/synchronization; not full 669-point release equivalence |
| Why must coherent v6 replace v5 joins? (August 6-7) | [coherent generation contract](coherent_teacher_dataset_generation_v1.md), [v6 acceptance](teacher_v6_full_production_acceptance_20260807.md) | Dataset release identities and accepted full production; no regeneration authority |
| What is the conservative daily boundary? (Gate E1) | [process operator](conservative_daily_process_operator_v1.md), [skeleton evidence](gate_e1_daily_operator_skeleton.md) | Canonical binding/updater/retained-tail and bounded replay/lifecycle |
| What is proven about physical gradients? | [Teacher D1](gate_d1_physical_parameter_gradients_v1.md) | Separate Teacher evidence from pending surrogate D2 |
| Why not revive an old architecture? | [failed architecture lessons](failed_architecture_lessons.md) | Rejection evidence, reusable lessons and explicit reopening conditions |
| Why prefer native checkpointed AD? (September 13) | [priority/metric decision](native_ad_priority_and_canopy_metric_20260913.md), [trained T30 evidence](trained_native_t30_ad_20260913.md) | Bounded native AD/science priorities; not every old custom-reverse closure |
| What did older fresh/joint and litter experiments show? (September 16-21) | [direct joint results](direct_joint_dual_pool_results_20260920.md), [four-arm litter results](litter_parallel_training_results_20260921.md) | Learning plus T30 tradeoffs; no scientifically accepted arm or automatic continuation |
| Why is final-method reproducibility different from checkpoint replay? | [fresh-method draft](fresh_method_verification_plan_20260920.md) | Historical rationale; current fresh-init/full-network requirements are in AGENTS/NEXT_STEPS |
| Why the shared architecture, then split decoder width? (September 21-23) | [shared design](shared_transformer_long_term_design_20260921.md), [split-width proposal](shared_split_width_proposal_20260923.md) | Accepted architecture/config roles; exact current parameter count in current-status |
| What admission was measured for the split-width candidate? | [daily results](shared_split_width_daily_results_20260923.md), [recurrent partial](shared_split_recurrent_partial_20260924.md) | Bounded technical evidence, not trained scientific acceptance |
| What exact-execution speedup survives complete updates? (September 26) | [combined full update](shared_combined_full_update_20260926.md) | Same-work gradients/AdamW/physical gates and complete timing |
| Why is daily64 not same-work 2x? | [daily64 balance](shared_combined_daily64_balance_20260926.md) | Changed sampling count, balanced dual branches, GPU-time accounting |
| What does mixed precision actually demonstrate? | [FFN results](shared_ffn_precision_results_20260926.md), [structural route](shared_next_structural_speed_route_20260926.md) | Native VJP/fragment gain and exact-gate failure; no production policy/full-update claim |
| Where are archived runtime checkout restore instructions? | [runtime archival](../../runtime-archival-20260922.md) | Verified archives and protected active/shared Git paths; no data-deletion authority |

For a specifically named experiment not covered here, search its name/date in
this directory or in the preserved old entry document. `HANDOFF.md` can answer
chronology questions, but should not be loaded as routine current context.

## Exact pre-cleanup document recovery

The later source-retirement snapshot is
`research/daily-before-source-retirement-20260926` (commit `5682064`). It preserves
the code after entry-document consolidation and before whole obsolete experiment
families were removed. Historical reports keep their original commands and paths;
run those experiments from their recorded revision or this recovery snapshot,
not by mixing individual old runners into the current training checkout.
List files retired since that snapshot with:

```powershell
git diff --name-only --diff-filter=D research/daily-before-source-retirement-20260926 HEAD
```

The annotated tag `research/daily-before-context-cleanup-20260926` preserves the
tracked repository at commit `273157c6bafea340d0be32d91fffaef929fc8a62`.
This avoids copying more than 13,000 obsolete entry-document lines into a new
archive folder. Dated evidence stays at its existing path.

Read any old entry page without modifying the current checkout:

```powershell
git show research/daily-before-context-cleanup-20260926:docs/START_HERE.md
git show research/daily-before-context-cleanup-20260926:docs/NEXT_STEPS.md
git show research/daily-before-context-cleanup-20260926:docs/CODE_MAP.md
git show research/daily-before-context-cleanup-20260926:docs/DOCUMENT_STATUS.md
git show research/daily-before-context-cleanup-20260926:docs/current-status.md
```

For a focused historical question, select matching lines instead of loading
the entire file, for example:

```powershell
git show research/daily-before-context-cleanup-20260926:docs/current-status.md | Select-String '15279139' -Context 3,8
```

If deliberately restoring a tracked file to that historical content, first
inspect/preserve its present changes, then use an explicit path:

```powershell
git diff -- docs/NEXT_STEPS.md
git restore --source=research/daily-before-context-cleanup-20260926 --worktree -- docs/NEXT_STEPS.md
```

The restore command overwrites that working file. It is documented for explicit
historical recovery, not a routine cleanup or permission to revert others'
changes. The tag preserves tracked content only, not ignored/untracked runtime
assets, unique scientific data or current processes.
