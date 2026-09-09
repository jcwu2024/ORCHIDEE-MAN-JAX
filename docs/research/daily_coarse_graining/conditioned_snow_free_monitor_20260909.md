# Corrected-head fixed-monitor transfer

Status: completed and passed at `6ed00b0` on 2026-09-09. Candidate replay and
paired development-monitor transfer are admitted; absolute accuracy remains open.

The corrected full integration `90f6403` passed all 96 gradient score checks
and the complete train-only step. This experiment compares corrected update 3
with that exact train-selected proposal on the existing monitor artifacts
251,358,465,572,679,786,893,1000. These artifacts are disjoint from the current
40-artifact update panel, but have been used as development monitors before;
this is early transfer evidence, not independent validation or a sealed test.
No direction, step magnitude or alternate candidate may be selected from
monitor values. Absolute science targets remain unchanged and separately reported.

## Recover the discarded proposal without a new search

The preceding contract deliberately retained no weights. Reconstruct the same
update from accepted recovery 3 and its recorded simplex coefficients/step.
Only six direct-GPP gradients and the eight-artifact 30-day mean gradient
have nonzero coefficients. Recompute these through the same corrected
bootstrap, compact daily tape, full streamed backward, float32 accumulation,
unit normalization and original 43-row combining kernel. Exactly-zero rows
are replaced with zeros; omit no positive coefficient, however small.
Preserve original registry order, dtypes, calibration, virtual step and
final selected-scale operation. This is replay of update 4, not reuse of an
old direction to perform an additional training update. No solver/scale search.

Before reading monitor outputs, check recomputed source losses/norms and
combined norm against the completed report (atol 1e-6, rtol 1e-5); require
all 32 replayed 30-day score comparisons. Re-evaluate the candidate's 40
direct-GPP losses and full scheduled science, comparing them with the
recorded selected candidate at the same tolerances. Check zero hard failures
and unchanged science counts. A mismatch is a replay failure, not a monitor
rejection, and does not license a new search or looser tolerance.

Save the successfully reproduced trainable tree as a separate diagnostic
candidate snapshot with corrected-boundary/parent/report metadata, before
monitoring, so it can be reused without more gradient work. It is not an
accepted recovery, optimizer state or promoted model. Failed monitor
candidates remain ineligible for continuation; accepted recovery is untouched.

## Paired evaluation and gate

Use the existing persistent runner's `_monitor` and `monitor_transfer_gate`
on corrected baseline and candidate, including native 7/30-day rollout,
GPP, maintenance/growth/autotrophic respiration, NPP, AGB/BGB, total live
biomass and ecosystem carbon. Preserve per-artifact losses and all physical
counts. Require direct-GPP mean and science macro to decrease, at least six
of eight direct-GPP artifacts nonregressing, critical-key regression <=0.1%,
supporting-key regression <=1%, and physical hard gates. No new thresholds.

This is the first corrected-head monitor pair: its baseline is freshly
evaluated corrected update 3, not a historical uncorrected monitor. It is a
single-step transfer claim, not the old uncorrected cumulative trajectory.
If it passes, review absolute accuracy and freeze a bounded continuation
from the saved corrected candidate. If it fails, attribute the fixed result;
do not use this monitor to tune the same proposal repeatedly. Long holdout,
365-day, Gate D2 and product gates remain later work.

## Execution

One free gln01 V100, at most 90 minutes; no sbatch or paid job. Existing v6
prepared data and runtime only. Test sparse replay against the original full
combination in mixed precision, exact-zero selection, source/score replay
gates, train-before-monitor ordering, paired gate and candidate serialization.
Use the existing CPU/GPU environment and transfer a fast-forward Git bundle.

## Local qualification

Eighteen monitor-replay, integration and persistent-training tests pass in
81.05 seconds. They exercise full versus sparse combining with float32
parameters (including a tiny positive coefficient), the actual 32-window
30-day caller/outer mean against whole-window autodiff, unchanged counts,
train-replay failure preventing monitor access, successful paired gating,
and reload of the separate candidate snapshot without creating a recovery.
The actual preceding JSON also passes 124 field/schema comparison checks;
its science counts are `evaluated_values`, and selected direct losses are
`artifact_changes.after`. Ruff, CLI/launcher arguments and whitespace pass.

## Running handoff

Execution commit `6ed00b0bce99b6e64d0cda1b3cb4a8c34be73aee` fast-forwarded
on the server; both wrappers pass shell syntax. Both GPUs were idle before
launch. The detached run started on gln01 GPU 1 (host `ibc13b02n04`, V100
32 GB) and reached Python `corrected_monitor stage=bootstrap`.

Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
Command: `GPU_DEVICE=1 EXPECTED_GIT_HEAD=6ed00b0bce99b6e64d0cda1b3cb4a8c34be73aee
OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/conditioned-snow-free-monitor-6ed00b0.json
bash scripts/hpc/audit_conditioned_snow_free_monitor.sh` (90m timeout).
The `.log`, incremental `.json`, and, after replay passes, `.candidate.pkl`
share that stem. There is no Slurm ID or paid allocation. Inspect this run
before restarting. Distinguish reconstruction failure, monitor-transfer
failure, and absolute-accuracy status. Accepted recovery remains update 3;
the diagnostic candidate is not an accepted continuation checkpoint.

## Completed result

Complete JSON was retrieved to `outputs/conditioned-snow-free-monitor-6ed00b0.json`.
The first direct SCP stalled; the subsequent compressed transfer is complete
and parses successfully. It reports `status=completed`, `passed=true`,
`classification=corrected_monitor_transfer_passed`, elapsed 1121.3835 seconds
(18.69 minutes). No timeout or paid allocation was involved.

Reconstruction passes every source and candidate-loss comparison and all 32
thirty-day score comparisons. Combined direction norm is 0.421342272648220,
versus recorded 0.421342272648339. The separate candidate was saved at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/conditioned-snow-free-monitor-6ed00b0.candidate.pkl`.
Do not recompute this already saved proposal. It contains corrected-boundary
metadata and proposed logical update 4; the accepted recovery file remains 3.

All eight monitor artifacts improve in direct GPP, mean loss change -0.89965%.
The detailed nine-key/two-horizon science macro changes -0.63715%. Every one
of the 18 key/horizon losses decreases, including ecosystem carbon; all
physical gates pass. Selected changes (loss reduction, not physical error %):

| Objective | 7-day loss decrease | 30-day loss decrease |
| --- | ---: | ---: |
| GPP | 0.62266% | 0.65070% |
| NPP | 0.62848% | 0.64011% |
| Aboveground biomass carbon | 0.68368% | 0.67444% |
| Belowground biomass carbon | 0.56163% | 0.59444% |
| Autotrophic respiration | 0.60267% | 0.66306% |

Absolute GPP, NPP, growth respiration and total autotrophic respiration gates
still fail at both horizons. AGB/BGB, total live biomass, maintenance
respiration and ecosystem-carbon gates pass on this bounded monitor only.
This is positive single-step development-monitor transfer, not proof of
independent generalization, sustained training slope or long-term simulation.

Next: freeze a bounded continuation from the saved corrected candidate, with
freshly computed directions at each update, predetermined training panels,
corrected-model monitor baseline and explicit cumulative/absolute science
reporting. Verify snapshot reload and corrected-head persistence, then run
within a stated resource envelope. Do not silently apply the old uncorrected
recovery/monitor identity or repeat the full discarded reconstruction. Paid
training, 365-day rollout, Gate D2 and final product gates remain unopened.
