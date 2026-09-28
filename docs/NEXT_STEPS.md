# Current Work Order

Updated: 2026-09-28. This page controls current operations; dated evidence does
not authorize its old proposed next action.

## Completed: source and context cleanup

Entry-document consolidation and substantive source retirement are complete.
Current shared helpers are independent of historical runners; abandoned
experimental implementations and their exclusive scripts, declarative
configurations and tests have retired. Current shared model/data/performance
paths, Teacher functionality, prepared-data class identities and scientific
evidence are preserved. Architecture, loss, gradient and data semantics are
unchanged. No further broad cleanup is the active work order.

The source-retirement baseline is tag
`research/daily-before-source-retirement-20260926` (`5682064`). Remaining
dependencies, unit collection and targeted tests were checked; measured
reduction and validation limits are in
[WORKSPACE_INVENTORY.md](research/daily_coarse_graining/WORKSPACE_INVENTORY.md).
Follow [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) to keep new context bounded.

The pre-cleanup tracked documents are preserved at annotated Git tag
`research/daily-before-context-cleanup-20260926` (`273157c6`). Recovery and
question-based historical routing are in
[HISTORY_INDEX.md](research/daily_coarse_graining/HISTORY_INDEX.md).

## Now: speed only, before training

Do not start neural training at the current throughput. Target an order of
magnitude improvement in effective complete-update throughput, preserving
data counts, daily/recurrent loss mix, parameter/state gradients and physical
checks. Report GPU count/card-hours. Track startup/compile time separately.

FP32 union stays opt-in; default field remains. Actual row0 canonical loss and
all64,884,481 parameter gradients pass strict1e-5/1e-6 against saved field
binary replay; physical checks pass. State gradients are finite/undefined-zero
but lack independent reference. Paid job15418739 completed0:0 in13:12
(0.22card-hours; estimated CNY0.6072, not bill). Six route units are persisted;
four physical/loss units reused. Repeats33.939/2.579/2.572s have unknown first-
repeat overhead, not training throughput. Free observed cache replay ended124
at its100s cold-call cap: neural forward hit cache, zero backend calls; no
full result/warm attribution. Cold retracing budget was underestimated.
Next define the original-four-row production micro4 comparison using public
mean daily objective/full parameter-gradient interface and saved field binary;
new union graph compile needed, proposed lower180/backend900/process1500s
budgets informed by prior micro4 timings, not authorized by this result.
No more generic profiler rerun. Retain the full state-gradient admission route;
retain a small saved-field directional state check before production adoption,
not a new gradient-infrastructure workstream. Preserve derivatives and physical
gates. No GPU job is active. No new paid job, production adoption, tolerance
change, recurrent run or training is launched by this result. Component27-
coordinate failure remains separate. Preserve all references/cache artifacts.

Read only these latest route/evidence pages for that decision:

- [Union dynamic-bank component results](research/daily_coarse_graining/shared_dynamic_projection_results_20260927.md).
- [Structural speed route](research/daily_coarse_graining/shared_next_structural_speed_route_20260926.md).
- [FFN precision results](research/daily_coarse_graining/shared_ffn_precision_results_20260926.md).
- [Broader-neural bounded assessment](research/daily_coarse_graining/shared_neural_precision_results_20260927.md).
- [Daily64 balance results](research/daily_coarse_graining/shared_combined_daily64_balance_20260926.md).
- [Unchanged-work complete-update results](research/daily_coarse_graining/shared_combined_full_update_20260926.md).

Daily64 + one T7 + one T30 is a measured load-balancing candidate, not the
adopted training sampling recipe. Doubling daily rows changes sampling;
approximately doubled daily rows/s does not prove same-workload 2x or 10x.
Another small tile/packing sweep is not the default primary route.

## Thread handoff

Continue on `research/daily-coarse-graining`. The next micro4 comparison is
specified in the union results page above but **not implemented or submitted**.
Implement the production-interface comparison first; preserve the original four
distinct rows and saved field reference. Do not rerun the capped row0 profiler
or assume the existing integrated Slurm script implements this next experiment.
No new paid allocation has approval; previous approval covered completed
job15418739 only. Free bounded checks remain within the user's standing scope.

Remote root: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. Reuse immutable checkout
`runtime/checkouts/shared-split-width-v1` at `c599236bcf57b745e9bafed678298010f4c3b58f`.
Under `runtime/outputs/diagnostics/`, preserve
`shared-split-profile-c599236.json`, `shared-expert-real-daily-context-c599236.pkl`,
`shared-union-integrated-20260927.pkl`, its `.field-reference.pkl`, and
`shared-neural-reference-20260927-staged.stages-cache/`. Full result pickles
stay on the server; small reports/logs are also local under `outputs/diagnostics/`.
The old field-reference pickle uses repeated row0, so it is **not** the numerical
reference for four distinct rows; use the admitted field executable with those
original rows. See the dated union report for measured evidence and limits.

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
