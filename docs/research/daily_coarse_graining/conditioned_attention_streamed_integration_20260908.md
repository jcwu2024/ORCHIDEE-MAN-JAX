# Streamed Backward Multi-Artifact Integration

Status: completed at `1e6c521`; original train-only update passes, but overall
integration is not admitted because seven observation-score comparisons fail.

The discarded four-window step `c726446` passed in 890.713 seconds. At norm
1e-4, predicted loss change was -0.00088241257 and actual change -0.00088227829;
the smaller step also descended. Both directional derivative errors were below
0.15%, all four windows improved, and every hard gate passed. No weights were
retained. This admits a test of the complete training update, not model promotion.

## Fixed Experiment

Start from the same accepted update-3 recovery in `45ab263`. Execute the
existing `_train_only_update` for logical update 4, using its original
conditioned persistent-training contract. Replace only the supplied seven-
and thirty-day gradient callbacks with full streamed backward. Keep direct
GPP/state gradients, scalarization, all 43 constraints, step calibration,
descending scales 1 through 1/64, first-passing-scale rule, forward evaluation,
state/budget gates and detailed train-only science gates unchanged.

Use the existing five groups of eight prepared train/train artifacts. The
scheduled group is 541, 717, 596, 867, 322, 528, 278, 442. These supply 64
seven-day windows and 32 thirty-day windows. All 40 artifacts supply direct
GPP constraints. There is no new sampling, data generation or validation read.
The unmodified rejected update-4 record is the historical mechanism reference;
do not repeat its known-bad reverse graph. This is a same-start, same-panel
integration test, not a new architecture/control experiment.

Each horizon callback runs canonical forward, obtains observation cotangents
using the existing global-bias loss, then calls the accepted full streamed
pullback. Include current state-derived PFT fractions. Keep global statistics
fixed only during the gradient; forward scale evaluation recomputes actual
statistics through the unchanged production code. Cache one-day reverse and
dynamic observation-loss graphs across artifacts. Preserve the existing
outer float32 accumulation boundary; the prior audit excluded that boundary
as the material cause, and local tests must verify its numerical equivalence.

## Admission And Stop

Require the original train-only selection/science gates to pass, all 96 horizon
callbacks to execute, and their observation versus production linearized scores
to agree at the already declared atol=1e-6, rtol=1e-5. Record per-window score
differences, execution time, memory estimates and callback counts. A finite-step
or linearization failure is a completed non-admission result; report the actual
values without relaxing thresholds or searching additional scales. Nonfinite
derivatives remain a numerical failure and preserve preceding progress.

Never call the persistence runner's `run`, monitor, or write helpers. Call only
its train-only proposal function, serialize the numerical record, discard
`updated_trainable`, and preserve update-3 recovery. A passing result admits
the existing paired fixed-monitor check before any retained continuation.
Neither the 365-day, Gate D2 nor product gate is opened by this experiment.

## Execution And Local Checks

Runner: `scripts/hpc/audit_conditioned_attention_streamed_integration.py`.
Its wrapper reuses the existing free gln01 launch boundary. One free V100,
one process, 120-minute bound, no paid submission. All outputs are incremental
JSON under the existing runtime outputs directory.

Before launch test the actual `_horizon_gradient` caller with both 8x7 and
4x30 microbatches, changing data, and nonzero global bias; compare with a
monolithic gradient. Test cached reverse reuse, dynamic targets, exact
update-4 panel routing, unchanged direct/state callbacks, no parameter writes,
complete runner serialization, and real launcher argument coverage.

Local qualification on 2026-09-08 passed all 25 integration, streamed-step,
streamed-adjoint and persistent-training regression tests in 40.84 seconds.
The actual horizon caller matches full differentiation across changing data
and parameters, with reverse compilation reused across both horizons. Ruff,
shell syntax and `git diff --check` also pass.

## Free-Node Execution

Launched on 2026-09-08 at approximately 14:07 CST on `gln01`
(`ibc13b02n04`), GPU 1, wrapper PID `39230`, commit
`1e6c521a78d3a2730f42da8beeaf472ea87ef0d3`. The card was idle before launch.
The wrapper has a 120-minute limit; no Slurm job or paid allocation was used.
Command: `bash scripts/hpc/audit_conditioned_attention_streamed_integration.sh`,
with `GPU_DEVICE=1` and `RUN_TIMEOUT=120m` from the project root.
Report and log are under `runtime/outputs/diagnostics/` as
`conditioned-streamed-integration-1e6c521.json` and `.log`.
Completed at approximately 14:44 CST in 2131.7955 seconds. The process has
exited; do not restart it as an unfinished run. Accepted update-3 recovery is
unchanged, and no candidate weights or fixed-monitor evaluation were retained.

## Completed Result

The remote JSON reports `status=completed`,
`admission.classification=streamed_update4_not_admitted` and
`admission.passed=false`. The original training selection nevertheless passes
at scale 1, the first scale, with parameter displacement L2 0.0003825156452.
All 64 seven-day and 32 thirty-day callbacks ran; the reverse compiled once.

Actual canonical forward losses improve after the discarded update:

| Objective | Before | After | Relative decrease |
| --- | ---: | ---: | ---: |
| Direct GPP, all 40 artifacts | 0.8039054332 | 0.7962660602 | 0.95028% |
| Scheduled direct GPP | 0.8128609037 | 0.8053368779 | 0.92562% |
| Scheduled one-day state | 0.0666406633 | 0.0662135695 | 0.64089% |
| Scheduled seven-day science | 0.1577726104 | 0.1567518369 | 0.64699% |
| Scheduled thirty-day science | 0.1241953630 | 0.1235440728 | 0.52441% |

All 40 direct-GPP artifacts improve. Detailed science macro decreases
0.61285%, and the physical/budget gates pass. Seven-day ecosystem-carbon
error increases 0.01162%, within its existing supporting-objective cap; do not
claim every individual science metric improved. Absolute GPP, NPP, growth
respiration and total autotrophic-respiration gates remain open at both
horizons. These are single-update loss reductions, not final prediction error
percentages or proof of long-term accuracy.

The remaining integration failure is observation versus production linearized
score agreement: seven-day zero-based callback ordinals 1, 15, 32, 37, 50, 52,
54 fail the frozen tolerance. The largest absolute difference is
1.83363843567e-5 at ordinal 37 (0.0280761086172 versus 0.0280944450015).
All 32 thirty-day comparisons pass; their largest absolute difference is
1.22813475316e-6, within the combined absolute/relative tolerance. The full
96-window result is therefore 89 passing comparisons and 7 failures.

The startup/log-only interpretation that the complete integration passed was
corrected after inspecting the JSON. Training-side acceptance and overall
integration acceptance must remain separate in future status reports.

## Immediate Next Task

Locate the seven seven-day score differences before any fixed-monitor check
or retained continuation. Keep this architecture, update-3 recovery, original
training panel, loss definition and tolerances fixed. Start by tracing the two
existing extraction paths in source. Map the failing callback ordinals to
their artifact/window identities, then compare identical-input trajectories,
defined/active masks, state-derived PFT fractions, per-variable sufficient
statistics and global-bias normalization. Determine whether the first
difference is in forward trajectories, extraction or score assembly; do not
assume roundoff, a specific physical process, or a gradient defect from these
scalar differences alone.

Use a bounded zero-update replay of the worst failing seven-day window plus
one passing seven-day window on a free node if source inspection is
insufficient. Reuse existing data and executables; do not repeat all 40
artifacts merely to reproduce the known discrepancy. Check the derivative
consequence of the identified difference before claiming the integration is
repaired. Once resolved, run the necessary focused/integration checks and
then the existing paired fixed-monitor evaluation before retaining updates.
365-day/long-held-out-site, Gate D2 and product acceptance remain later gates.

Evidence was read directly from the complete remote JSON. A previous SCP
attempt left a 32768-byte local partial file; do not use that partial copy as
the report. Retrieve and parse a complete copy before any local analysis.
