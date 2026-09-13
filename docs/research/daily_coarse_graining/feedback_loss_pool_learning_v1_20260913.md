# Fixed feedback-loss learning: one prepared-pool epoch

Status: implementation/preflight preparation; no paid submission.

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
