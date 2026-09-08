# Streamed Backward Multi-Artifact Integration

Status: frozen before real-data evaluation; implementation and local checks pass.

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
