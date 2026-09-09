# Ten-update corrected-head continuation

Status: user approved implementation, free-node checks and submission to gnall,
2026-09-09, under the six-hour / CNY 16.56 envelope below. The exact panels are
materialized in `manifests/coarse_graining/conditioned_snow_free_continuation_v1.json`.
The runner is `scripts/hpc/run_conditioned_snow_free_continuation.py`; the
launcher is `scripts/hpc/slurm_conditioned_snow_free_continuation.sh`.
The saved-candidate free admission passed in 518.64 seconds; gnall job
`14979295` subsequently completed all ten updates. See the
[execution record](conditioned_snow_free_continuation_execution_20260909.md).
The [result review](conditioned_snow_free_continuation_result_20260909.md)
admits sustained development-monitor learning, not final model accuracy.

## Research question and starting evidence

Determine whether the admitted small update can accumulate useful improvement
over changing training data, while preserving carbon/biomass behavior, and
measure improvement per wall-clock hour. Ten updates do not establish training
sufficiency or final accuracy; they provide a first sustained learning curve
with the repaired gradient path.

The corrected full step `90f6403` passes all 96 horizon comparisons. Its
reproduced candidate `6ed00b0` passes transfer to all eight existing monitor
artifacts: direct-GPP loss -0.89965%, 18-key/horizon science macro -0.63715%,
all physical gates pass. On that monitor, 30-day GPP normalized RMSE is still
1.08270 (target <=0.25) and signed bias 0.92406 (absolute target <=0.07).
NPP normalized RMSE is 1.28888, bias 1.10701. These are errors divided by the
frozen normalization scales, not percentage physical errors. Short-horizon
AGB/BGB agreement is not proof of annual/long-term biomass stability.

Start by loading the saved `conditioned-snow-free-monitor-6ed00b0.candidate.pkl`
under remote `runtime/outputs/diagnostics/`. This is proposed logical update 4,
already admitted for single-step development-monitor transfer. Do not replay
its gradients again. Initialize a separate corrected-model continuation run;
leave the original accepted update-3 recovery and reports intact.

## Model and training method

- Keep conditioned Attention, 109,464,375 total / 15,960,211 trainable
  parameters, existing input/parameter ownership and retained processes.
- Explicitly enable the warm snow-free correction before binding all forward,
  direct/state-gradient and full streamed 7/30-day-gradient helpers.
- Keep the existing direct GPP, one-day state anchor and 7/30-day science loss,
  global-bias normalization, 43-source direction solver and step calibration.
  Recompute every source gradient and direction at each new parameter state.
  The sparse replay shortcut is inapplicable to a new update, because its
  eventual coefficients are unknown until those gradients are computed.
- Keep the current 2% maximum predicted relative-response calibration and
  descending scales 1 through 1/64. No AdamW/momentum, cosine schedule, new
  backbone, increased parameter count or step-size retuning in this run.
  This is the current multi-objective update rule, not a conventional
  single-loss optimizer with an epoch-based learning-rate schedule.
- Keep full inter-day derivatives; no truncated BPTT, half-hour forcing
  reconstruction, 48-step inference or whole-30-day reverse compilation.

The purpose of holding these choices is to measure sustained behavior of the
now-admitted method before changing its optimization rules. The 43 constraints
are an experimental training strategy, not a scientific necessity: if they
prevent further steps, report which constraints conflict and how they limit
the step. Do not label that outcome an Attention architecture failure.

## Data and exact update plan

Ten additional successful updates: logical 5 through 14. In user reports,
label these new updates 1 through 10, with saved logical update 4 as baseline.
Use only existing prepared v6 train/train artifacts; no new Teacher production,
v5 joining, new sampling run or data download for local training.

Use the existing `build_persistent_scale_training_schedule()` (100 groups of
eight). For logical update u in 5..14, use group IDs
`(u, 4*u+3, 4*u+4, 4*u+5, 4*u+6)`; IDs are one-based and the first group
supplies the state and recurrent losses. This agrees with the old declared
logical-update-5 panel and extends its grouping rule without using monitor
values. Materialize these exact artifact IDs in the execution contract before
training; do not pass an out-of-range update to the old three-update loader.

| New update | Logical update | Schedule group IDs |
| ---: | ---: | --- |
| 1 | 5 | 5, 23, 24, 25, 26 |
| 2 | 6 | 6, 27, 28, 29, 30 |
| 3 | 7 | 7, 31, 32, 33, 34 |
| 4 | 8 | 8, 35, 36, 37, 38 |
| 5 | 9 | 9, 39, 40, 41, 42 |
| 6 | 10 | 10, 43, 44, 45, 46 |
| 7 | 11 | 11, 47, 48, 49, 50 |
| 8 | 12 | 12, 51, 52, 53, 54 |
| 9 | 13 | 13, 55, 56, 57, 58 |
| 10 | 14 | 14, 59, 60, 61, 62 |

Local schedule expansion confirms 400 artifact exposures, 400 distinct
artifact IDs, and no overlap with monitor IDs 251,358,465,572,679,786,893,1000.
This is artifact-level separation; it does not assert 400 distinct locations,
disjoint point-years/days, or that these data have never appeared historically.

Per update: 40 x 208 = 8,320 direct-GPP day positions; the first eight
artifacts also supply 256 one-day state anchors, 64 seven-day windows and
32 thirty-day windows. Across ten updates: 83,200 direct day positions,
2,560 state anchors, 640 seven-day and 320 thirty-day windows. The latter
windows contribute 14,080 recurrent day positions. These are sample uses,
not deduplicated days or an epoch; the sources overlap and must not be summed
as unique data. Report actual completed usage and deduplication when metadata
permit, rather than asserting all planned data were consumed after an early stop.

## Evaluation and preservation

The fixed baseline is the freshly reloaded corrected candidate's monitor,
equivalent to `monitor_after` in `6ed00b0`, not the old uncorrected monitor or
the corrected update-3 pre-proposal scores. Verify reload once with the normal
forward path, including model boundary, parameter shapes/dtypes, finite active
states and normal budgets. Preserve baseline loss/count evidence in the new run.

Keep all existing train-only selection gates. Evaluate the same eight-artifact
monitor automatically after each selected update, inside the same process.
At roughly 75 seconds per hot monitor, this is a modest fraction of an update
and preserves the currently admitted protocol. Require both local transfer
from the previous accepted monitor and cumulative transfer from this run's
corrected baseline under the existing tolerances. A few component/artifact
regressions within those tolerances are allowed; not every metric must
strictly improve on every update. Monitor values do not select a scale or
direction, and failed candidates are not fed back into same-panel tuning.

Save the train-selected candidate before monitoring, and save accepted
continuation recovery after a passing monitor. Store current logical update,
parameter tree, corrected-head setting, data-plan position and monitor records.
If interrupted between selection and monitoring, finish that pending monitor;
do not recompute the proposal. An accepted recovery and a pending candidate
must be distinguishable. Reuse one cached daily reverse across updates.

Human-facing summaries occur after new updates 2, 5 and 10, rather than
asking for confirmation after each update. Every update is recorded. Summaries
show direct GPP, GPP/NPP/respiration/AGB/BGB normalized RMSE and signed bias
at both horizons, cumulative relative changes, physical counts, parameter
step size and limiting source, selected scale, and hot seconds per update.
Plot the fixed-monitor curve; losses from changing training batches are not
a directly comparable learning curve. Heterotrophic respiration and NEE/NEP
remain reported through existing outputs; do not add new duplicate weighted
losses in this continuation.

## Interpretation and stopping

1. Nonfinite gradients/states, physical failure or score-path disagreement:
   preserve completed work and localize the failing input/process.
2. No common direction or no passing train-only scale: preserve evidence of
   conflicting sources and the step limitation. Review loss/constraint and
   step-size strategy; do not switch backbone by default.
3. Failed local/cumulative monitor gate: preserve the candidate for diagnosis,
   stop this bounded continuation, and do not select another candidate on
   the same monitor in the same experiment.
4. Ten updates completed: assess both magnitude and persistence of improvement,
   especially the last five updates, against actual compute consumed. A
   plateau or progressively vanishing step size requires a training-strategy
   review before an expensive extension.

For planning only, sustaining roughly the observed step could yield around
5--10% less direct-GPP loss and 3--6% less science macro loss over ten updates.
These are reference ranges, not guaranteed outcomes or new pass thresholds.
Do not use a 0.6--0.9% single-step result to forecast final convergence time.
Positive transfer plus appreciable continuing reduction supports a longer
bounded training stage. Passing absolute accuracy is still required before
model promotion; it is not required merely to retain a learning-smoke update.
No claim of underfitting, overfitting or architecture impossibility follows
from ten updates alone.

## Execution plan and cost

Preparation: implement the new continuation/persistence boundary, lightweight
local lifecycle tests, then a free-node reload/forward check of the saved
candidate. Verify the required existing prepared artifacts are readable and
CLI/entrypoint paths are correct. Do not rerun derivative/Teacher gates already
passed solely for formality. This phase produces no extra trained step.

Training: one continuous job, proposed gnall allocation of one V100 32 GB and
8 Slurm CPUs, one task, maximum 6 hours. Working directory remains
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`; outputs go to a separate corrected
continuation directory under `runtime/outputs/training/`. Reuse the canonical
GPU environment and existing Singularity wrapper. Proposed launcher and exact
command are implemented and covered by the user's approval through submission.
No multi-GPU or alternative model arm in this run.

Timing basis: the last full integration took 3363.25 seconds. The first detailed
forward artifact took 1071.04 seconds versus about 9.6 seconds hot; the first
gradient artifact took 1349.65 versus about 42 seconds hot. Those first-use
excesses total about 39.5 minutes. Excluding them and bootstrap (~229 s), the
inferred warm full update is about 12.8 minutes; adding an eight-artifact
monitor gives about 14 minutes. Ten steps plus one startup suggest roughly
3--4 hours; allow 3--4.5 hours as a planning range and cap at 6 hours. These
are estimates from free-node timings, not a paid-node throughput benchmark;
extra scales, shape changes and resource contention can increase them.

Measure hot reuse by the second update. If new compilation is repeated at
each equal-shaped update, fix the runner rather than budgeting every step
at cold-start cost. The 6-hour timeout preserves the last accepted recovery;
no automatic extra paid job is authorized if the run is incomplete.

Using Explore1000's stated additive rates, the proposed maximum charge is:
GPU 1 x 6 x CNY 2.20 = CNY 13.20; CPU 8 x 6 x CNY 0.07 = CNY 3.36;
total CNY 16.56. At 3--4.5 hours the corresponding estimate is CNY 8.28--12.42.
Queue waiting is additional elapsed time, outside these compute-time estimates.
The user approved this bounded execution through submission. Display the exact
command and resource envelope before submitting; no second approval is needed.

After a useful sustained-learning result, scale training/data consumption and
then assess independently reserved data. Annual rollout, long holdout, Gate D2
and product acceptance remain separate later milestones.
