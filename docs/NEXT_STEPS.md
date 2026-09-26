# Current Work Order

Updated: 2026-09-26. This page controls current operations; dated evidence does
not authorize its old proposed next action.

## Context cleanup: completed scope

Current entry pages are consolidated, task-specific source groups are indexed,
and three isolated historical shell launchers are retired with tag recovery.
Shared old helpers remain where imports/tests still need them; this is not a
claim that all historical source has been retired. Follow
[RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) to keep new context bounded.
Scientific evidence, datasets, checkpoints and source provenance are preserved.

The pre-cleanup tracked documents are preserved at annotated Git tag
`research/daily-before-context-cleanup-20260926` (`273157c6`). Recovery and
question-based historical routing are in
[HISTORY_INDEX.md](research/daily_coarse_graining/HISTORY_INDEX.md).

## Next research action: speed only, before training

Do not start neural training at the current throughput. Target an order of
magnitude improvement in effective complete-update throughput, preserving
data counts, daily/recurrent loss mix, parameter/state gradients and physical
checks. Report GPU count/card-hours. Track startup/compile time separately.

The latest bounded native `F16_F16_F32` FFN diagnostic is promising but changes
numerics: finite forward/VJPs, strict exact-route comparison fails. It is not
production precision admission. The next scientific work is to specify a
bounded broader-neural assessment with an explicit numerical-error contract
before integration. Reuse saved actual inputs, prepared references and
unchanged executables. Do not launch a long recurrent compile or training
from fragment evidence alone.

Read only these latest route/evidence pages for that decision:

- [Structural speed route](research/daily_coarse_graining/shared_next_structural_speed_route_20260926.md).
- [FFN precision results](research/daily_coarse_graining/shared_ffn_precision_results_20260926.md).
- [Daily64 balance results](research/daily_coarse_graining/shared_combined_daily64_balance_20260926.md).
- [Unchanged-work complete-update results](research/daily_coarse_graining/shared_combined_full_update_20260926.md).

Daily64 + one T7 + one T30 is a measured load-balancing candidate, not the
adopted training sampling recipe. Doubling daily rows changes sampling;
approximately doubled daily rows/s does not prove same-workload 2x or 10x.
Another small tile/packing sweep is not the default primary route.

## Preserve the method

- Reuse accepted v6 fixed data; no regeneration or v5 joins.
- Preserve the shared 768/eight-block + 384/two-block candidate, all potentially
  active outputs, ownership/PFT/mask semantics and physical precision.
- Use canonical `bind_daily_operator`, conservative updater, retained tail,
  shared science extractors/losses and native checkpointed AD. No independent
  runner cotangent implementation or shortened recurrent gradients.
- Architecture/capacity, sampling or production precision changes need a
  separate explicit decision. An auxiliary label does not justify deletion
  or reduced capacity. Permanently masked-coordinate omission must be proven
  against the declared static layout and restore full shape.
- For changed loss/state/AD/JIT boundaries, reuse relevant prior evidence and
  add bounded deployed-runtime, meaningful-gradient and complete-update gates.
  Documentation/logging-only changes do not require new expensive checks.

## Remaining acceptance gates

| Gate | Current state / next requirement |
| --- | --- |
| Teacher port | Scoped PFT14 evidence accepted; full 669-point Fortran equivalence still separate |
| Data | Coherent v6 fixed release accepted; reuse unchanged |
| Structure | PFT registry/interface, canonical process boundary and replay accepted within scope; PFT2-PFT13 unsupported |
| Training execution | Bounded native AD and complete-update parity passes; 10x goal unmet |
| Final training method | Specify architecture, splits/sampling, seeds/init, losses, trainable scope, transitions, stop/selection rules; execute end to end |
| Science | Demonstrate learning and held-out daily/recurrent accuracy; 365-day/long-site drift and productivity/biomass validation pending |
| Surrogate gradients (D2) | Physical-parameter validation pending; Teacher D1 does not certify the surrogate |
| Product | Final model/IO/restart/release acceptance pending |

## Execution authority

No new paid job, production training or production precision adoption is
authorized by this page or cleanup. Paid `cnall`/`cnmix`/`gnall` submission
requires explicit resources, wall time, directory, command and worst-case-cost
approval. Use Explore1000 under `/WORK/liwei_work/jcwu/`; never compute on login
nodes. Before future HPC work, read
[deployment-explore1000.md](deployment-explore1000.md) and the job-script skill.
Prefer local `ORCJAX`, lightweight tests and output files under `outputs/`.
