# Fixed feedback-loss learning: one prepared-pool epoch

Status: job15061139 COMPLETED1998/1998 updates,03:04:49,exit0:0.
Frozen execution checkout commit:5502ce3.

## Completion and active development evaluation

All updates finite; declared fixed training/excluded-panel total and flux
learning screens pass. Consumed63707 unique days and63936 draws, matching
the declared existing-pool epoch. No automatic extension or checkpoint
selection.100/500/1000/1500/1998 snapshots saved and roundtrip-validated.
Elapsed runner11063.3s; scheduler11089s. Peak actual live device memory
10674317824 bytes (9.941GiB), reserved pool22151763456 bytes (20.631GiB).
Estimated additive charge11089/3600*(2.20+8*0.07)=CNY8.50, belowCNY16.56 cap.
Evidence: outputs/feedback-pool-15061139-report.json downloaded unchanged.

After observing both free gln01 V100s idle, launched the predeclared fixed
16-window7/30-day evaluations at unchanged5502ce3. Actual Bash argument
expansions pass; both processes reached forward_screen bootstrap with no
initial exception.500 usesGPU0/Python98744;1998 usesGPU1/Python98816.
Each bounded75min; logs and JSON outputs share prefix
runtime/outputs/diagnostics/feedback-pool-15061139-forward- with500/1998
suffixes. No new paid work or architecture/gradient-graph change.
These outcomes are pending; one-day learning-screen success does not
establish recurrent fit or independent generalization. The panel has informed
loss development and must not be called a final untouched validation set.

Fixed64 excluded TRAIN-day monitor trajectory:

| Update | Total loss | Flux loss | Canopy change term (unweighted) | Litterpart term (unweighted) |
| --- | --- | --- | --- | --- |
| 0 | 0.794977 | 0.489405 | 64488993 | 2006.86 |
| 500 | 0.041577 | 0.028514 | 464722 | 3.25 |
| 1000 | 0.032148 | 0.020846 | 72381 | 0.97 |
| 1500 | 0.034670 | 0.023952 | 29786 | 0.69 |
| 1998 | 0.030684 | 0.020352 | 15263 | 0.60 |

All fixed train32/excluded64 monitor physical hard counts are zero across
0/100/500/1000/1500/1998. These objective terms show learning from fresh
initialization; they are not percentages of physical prediction accuracy.
Each one-day example starts at Teacher state, so recurrent feedback remains
the separate active scientific question. Do not choose1000 post hoc because
one monitor value later fluctuates; primary checkpoint remains1998.

### Physical interpretation (independent Sol high read, root verified)

Excluded64 TRAIN-domain day panel physical RMSE (all fluxes gC/m2/day;
AGB/BGB daily increments gC/m2) from fresh initialization to final:

| Quantity | Initial RMSE | Update500 RMSE | Final1998 RMSE | Final bias |
| --- | --- | --- | --- | --- |
| GPP | 3.065 | 0.683 | 0.577 | -0.170 |
| NPP | 2.613 | 0.483 | 0.413 | -0.111 |
| Autotrophic respiration | 0.521 | 0.214 | 0.171 | -0.058 |
| AGB daily change | 2.019 | 0.367 | 0.305 | -0.090 |
| BGB daily change | 0.647 | 0.125 | 0.114 | -0.022 |

These are meaningful one-day learning improvements, with remaining negative
bias. Stock level/change residuals coincide because the start state is Teacher
supplied; low relative stock-level error does not prove stable accumulation.

The canopy source-direct change remains badly wrong: excluded64 physical
RMSE43396.25 ->3683.89 ->667.62 gC/m2/day. Final mean predicted change is
-476.573 versus Teacher+7.720, bias-484.293. Large improvement from a very poor
fresh model is not adequate absolute accuracy and still has the wrong sign.

Litterpart physical RMSE worsens0.50005 ->0.69413 ->0.70462 despite normalized
loss2006.86 ->3.25 ->0.60. Final four coordinate predictions are near0.9964:
target-one coordinates fit, target-zero coordinates have about0.9965 error.
This exposes imbalance between the weighted objective and physical coordinate
accuracy. It is not evidence that the AD tool is wrong. Do not silently change
coefficients or select another checkpoint. Complete declared recurrent screens,
then inspect the source-backed coordinate scaling/weight balance before the
next training proposal. Larger data availability does not itself fix this
demonstrated loss/physical-error mismatch. No final model acceptance.

Submission record:2026-09-13, via gln01, exact command/resources below.
Scheduler start17:39:37, node ibc13b03n01,6h limit/end23:39:37 (scheduler
timestamps; completion may occur earlier). Output directory:
`runtime/outputs/training/fresh-feedback-pool-15061139/`; log:
`runtime/outputs/diagnostics/feedback-pool-15061139.log`.
The preparation-time protocol's paid_submission_authorized=false is preserved
as immutable configuration provenance; this explicit user approval and job
record establish execution authority. Do not edit the running checkout.

## Scientific question and fixed method

Can targeted litterpart and canopy-carbon supervision reduce the diagnosed
recurrent GPP loss and inventory drain after actual learning, while preserving
productivity, respiration and biomass accuracy? The prior two-update probe
admitted the actual objective/AD/optimizer path; it did not answer this question.

Start fresh from seed20260911 with all109,464,375 Attention parameters trainable.
Keep the architecture, DOC input conditioning, input/output statistics,
constrained daily updater, retained exact tail and frozen v6 data unchanged.
Inherit neither the exploratory1998-update checkpoint nor the two-update
probe (which deliberately wrote no checkpoint).

Executable model protocol:
`manifests/coarse_graining/fresh_start_feedback_model_v1.json`.
Pool/training protocol:
`manifests/coarse_graining/fresh_start_feedback_pool_v1.json`.
The model protocol's three archived examples initialize templates only; actual
training rows come from the broad catalogue and sampler. This distinction is
recorded explicitly instead of claiming the three examples form the dataset.

Train one complete epoch of the existing prepared TRAIN pool:1998 batch32
updates, microbatch4,63936 draws and63707 distinct Teacher days across2000
prepared artifacts. It is not an epoch of all global v6 data. Pool and sampler
seed20260912 are the same as job15048786; every occurrence of the64 monitor
days is excluded from training. No new data extraction, Teacher run or v5 join.

The loss remains `base + 3e-9 * canopy_change + 2.5e-5 * litterpart_feedback`.
Base is daily independent flux loss plus0.1 state anchor plus0.01 process
supervision. AdamW stays LR1e-5, betas0.9/0.999, epsilon1e-8, no weight decay,
global clip1 after complete batch32 averaging. The changed schedule wording
in the model protocol denotes the same constant learning rate. No coefficient
search, backbone change, automatic extension or loss-plateau early stop.

## Monitoring, checkpoint retention and stop rule

Evaluate fixed first-batch32 TRAIN rows and the64 excluded TRAIN-domain days
at0/100/500/1000/1500/final. Preserve physical fluxes, biomass levels/changes,
source-direct canopy change and separate litterpart coordinates. Report total
autotrophic respiration as well as its components: the two-update probe
showed that improving individual component errors can worsen their sum.

Keep recoverable full state (parameters, Adam state and sampler cursor) every
250 updates and at monitors. Retain named snapshots100/500/1000/1500/1998 so
later evaluation does not depend on an overwritten recovery checkpoint. The
primary result is the final complete epoch1998 checkpoint. Intermediate
snapshots diagnose learning; they are not selected post hoc as a final model.
Stop on numerical/physical/resource failure and preserve the latest complete
checkpoint and explicit error. Time-limit interruption is incomplete, not an
accepted shorter run; any recovery follows the same declared method/cursor.

Run free T7/T30 development evaluation at the fixed500 and1998 snapshots,
using the existing16-window panel (8 per horizon,14 sites/7 years,296 days).
The training/monitor exclusion source is unchanged; verify this metadata once
when preparing the launch rather than reselection or rehashing Teacher files.
The500-step evaluation can proceed on an idle free GPU while paid training
continues, using the immutable snapshot and an atomically read report.
Do not mutate the running training checkout. Do not automatically cancel the
paid job based on interim science results or choose a different checkpoint.

Keep all13 absolute science keys and established hard gates. T7 normalized
RMSE/bias limits are0.35/0.10; T30 limits0.25/0.07. Report physical daily and
cumulative fluxes and stock changes alongside normalized metrics. Teacher
values supply initialization/targets only; free recurrence has no resets.
The panel has informed loss design, so even a pass is development evidence,
not final spatial/temporal generalization or365-day/Gate D2 acceptance.

The completed15048786 run is the unchanged-loss reference at the matched
full1998-update epoch. Reuse it, not a duplicate paid control. Stored prefix
loss/monitor reports are available, but its final weights cannot act as a
500-step control. Compare500 primarily with absolute targets and its declared
initialization/monitor trajectory. Corrected canopy/nonliving reporting must
be used for any physical re-evaluation of the old checkpoint.

## Engineering admission and resources

Extend the existing `run_fresh_start_one_day_pool.py`; reuse the same dynamic
coefficient batch objective and native reverse executable used in the passed
two-update admission. Shared checkpoint format/cursor rules remain. Add
retained paths and explicit evaluation method/update expectations rather than
copying runners or introducing another checkpoint format. Host finiteness
checks keep the proven path and avoid a separate large diagnostic JIT.

Required preparation: focused local integration/recovery/optional-loss tests,
actual Bash dry runs for launch expansion, and real-array CPU shape tracing
of the pool gradient/optimizer/monitor. Reuse the passed deployed29.314s
runtime check and actual2050.326s native AD/JVP/two-update report where the
objective and AD boundary are unchanged. No new independent JVP sweep or
two-update GPU repeat is required solely for checkpoint/host-report edits.

Prior old-loss epoch completed in8677s (2.41h), averaging4.093s/update. The
new two-update probe's hot loop was7.303s/update including host finite checks;
1998 such updates would be4.05h before compilation, I/O and checkpoint overhead.
These are planning anchors, not a proven sustained-throughput bound. Request
6h, with an inner350m timeout; keep intermediate recovery artifacts.

Proposed resources: gnall,1 node,1 V100,8 allocated CPUs,32GiB host memory.
Worst-case additive charge: CPU8*6*0.07=CNY3.36; GPU1*6*2.20=CNY13.20;
totalCNY16.56. This covers training only; the planned bounded rollout checks
use free resources if available. Paid evaluation needs a separate approval.

Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/feedback-pool-v1`.
Submission command, after explicit approval:
`/rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_feedback_loss_pool.sh`.
Outputs: `runtime/outputs/training/fresh-feedback-pool-<jobid>/report.json`
and recovery/retained checkpoints; log `runtime/outputs/diagnostics/feedback-pool-<jobid>.log`.

Implementation, target preflight and expected code commit are recorded below
as they complete. No paid run may be inferred from the existence of this file.

## Implementation verification

Sol(high) extended the existing pool runner and Terra(high) extended the
snapshot evaluator; root reviewed both and fixed the complete-monitor scalar,
raw-physical retention, optional validity and metadata-boundary requirements.
Training-specific24 focused tests and evaluator-specific15 focused tests pass;
root's combined touched-runner suite passes15 tests. Ruff/compile checks pass.
Tests include interrupted/resumed old and amended tiny full loops, retained
checkpoint roundtrip, unchanged default reverse aux/scalar versus enriched
monitor outputs, and two-day recurrent versus Teacher-start canopy changes
with masked litterpart coordinates. No large local neural execution was used.

Retained names are `report.update-000500.checkpoint.pkl` etc.; rolling recovery
remains `report.checkpoint.pkl`. Evaluation requires both `--expected-update`
and `--expected-pool-protocol` when using an explicit snapshot. Training method,
model protocol and checkpoint method/step must agree. The old historical
final1998 default still works. The actual dataset and CUDA admission from
`feedback-loss-updates-v1` are reused; new monitor outputs do not alter reverse
loss semantics or reimplement any cotangent.

## Completed target preparation

Execution checkout `runtime/checkouts/feedback-pool-v1` was fast-forwarded
to5502ce35e2111b1fabb39605c743431b3326ceb5. The previously admitted
`feedback-updates-v1` checkout was not changed. Actual test02 Bash syntax and
`--dry-run` argument expansion pass for all three new shell launchers.

Metadata-only `check_feedback_pool_plan` passes all seven checks: accepted
catalogue/selection, declared epoch counts, excluded monitor, rollout disjoint
from training and monitor, matched control sampling/source, and completed
control epoch. Counts:1998 updates,63936 draws,63707 unique TRAIN days,
64 excluded monitor days,296 rollout days across16 windows. This verification
reads metadata without Teacher payloads and does not regenerate any data.

Actual test02 CPU shape tracing completes with exit0/status passed in
224.161274824291s, classification
`pool_gradient_optimizer_monitor_shapes_only`, zero updates and host peak
4659789824 bytes (4.34GiB). It loads real prepared inputs and traces the
amended batch4 gradient, optimizer and enriched monitor using the new protocols.
Recorded coefficients are base1/canopy3e-9/litterpart2.5e-5. This is integration
evidence, not a numerical GPU or scientific accuracy pass; the unchanged
reverse graph's previously completed deployed AD/JVP/two-update evidence
remains the numerical admission.

Reports reside under remote `runtime/outputs/diagnostics/` and local `outputs/`:
`feedback-pool-v1-plan.json` and `feedback-pool-v1-cpu-shape.json`.
At preparation completion no test remained running and no paid job had been
submitted. The subsequent approval/submission is recorded at the top. Subsequent
documentation-only commits do not require redeployment or repeating these
unchanged checks.
