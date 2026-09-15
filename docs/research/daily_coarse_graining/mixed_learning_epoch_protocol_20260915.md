# One prepared-pool pass: method and bounded budget

Paid submission PAUSED by the user's subsequent two-route decision; free
integration completed1270.75s, one mixed update and checkpoint replay, exit0.
Current work is `parallel_training_routes_20260915.md`. Not scientifically accepted. This moves
from the200-update pilot to substantial data coverage with the admitted
Attention FP32 + checked recurrent execution. It does not reopen architecture,
loss or performance research.

## Learning method

Use `mixed_learning_epoch_v1.json`, `mixed_learning_epoch_model_v1.json` and
`mixed_learning_execution_fp32_checked_v1.json` under `manifests/coarse_graining`.
Initialize all109,464,375 weights from seed20260911 and fresh AdamW; do not inherit
an exploratory checkpoint. Keep the admitted model, objective D+.5R7+.5R30,
physical constraints, normalization and train/development separation unchanged.
AdamW retains lr1e-5, beta1=.9, beta2=.999, epsilon1e-8, weight decay0, global
clip1. Constant lr is declared for this entire pass; no ad hoc loss/LR changes
at intermediate checkpoints. This tests data exposure and learning duration.

The daily stream traverses the ready346,539 unique TRAIN rows exactly once,
batch64/micro8, seed20260912,5415 updates. Fixed-shape terminal padding adds21
declared replay occurrences (346,560 total draws); report padding separately.
This is one epoch of the prepared pool, which is about4.04% of eligible Teacher
TRAIN days, not one epoch of the full Teacher release.

Each update also uses one7-day and one30-day window. The existing prepared
recurrent assets offer15,968/7,988 eligible unique windows respectively after
exclusions, sufficient to select5415 of each without repeating a window.
Select using the existing seed20260914 and algorithm; enforce disjoint T7/T30
days within each update. Across updates and between daily/recurrent objectives,
Teacher days may overlap: measure actual union coverage rather than summing
occurrences. Exclude64 fixed monitor days and296 development days from recurrent
selection, and retain the current daily catalogue exclusions. Prepare only
one consolidated schedule JSON; no Teacher payload regeneration or replication.

## Progress visibility and recovery

Evaluate and retain checkpoints at0,200,1000,2500,4000,5415, using the same
fixed TRAIN and8T7+8T30 development panels and physical-unit reports. Monitor
GPP/NPP/RA/RH, AGB/BGB, canopy and nonliving carbon; loss alone is insufficient.
Milestone200 should be available early, and1000 after roughly4-6h including
startup. These are estimates, not deadlines. Do not interrupt/resubmit at each
milestone; the one process continues while intermediate results can be inspected.

Save one rolling recovery every50 updates (about10min warm), with parameters,
AdamW moments, daily/recurrent cursors, method and execution policy. Use the same
immutable checkout on recovery. Six milestones plus one rolling snapshot need
about8.6GiB plus one~1.23GiB atomic-write temporary. Checkpoint/report/log files
are a small fixed set, not one file per sample/update. A timeout can lose at
most the work since the last successful snapshot; no automatic paid restart.

Complete the declared5415 updates unless technical/physical failure or wall
limit occurs. Report all milestones; select update5415 as the final result of
this bounded pass, not the best-looking development checkpoint. Scientific
gates are unchanged and may still fail. Failure to reach them after one pass
does not automatically identify architecture or data as the sole cause.

## Time and cost

The combined fixed-checkpoint benchmark measured11.76113s per complete update:
5415*11.76113/3600 =17.69h. Actual rotating-data training may be slower. With
first-use compilation, reading, six monitored milestones and checkpoint IO,
plan20-24h of execution, excluding queue time. Request24h as a hard spending
cap; it is not a guarantee of completion. If the cap is reached, inspect the
saved progress and request any additional budget explicitly.

One gnall job,1 V10032GB,8 allocated CPU cores,64GiB host RAM,24h wall limit.
Rates are additive:8*.07 +1*2.20 =2.76yuan/h. Expected20-24h corresponds to
55.20-66.24yuan. Worst case: CPU13.44 +GPU52.80 =66.24yuan. The previous200-step
job consumed3.56yuan estimated; this run deliberately does27.075x as many
updates and substantially more unique data. Queue wait is not included in
runtime or this allocated-resource estimate. These are estimates, not bills.

Entry `scripts/hpc/run_mixed_learning_epoch.sh` shares the existing runner and
environment. Paid script `scripts/hpc/job_mixed_learning_epoch.sh` uses:

- Workdir `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-epoch-v1`.
- Command `sbatch scripts/hpc/job_mixed_learning_epoch.sh`.
- Output `runtime/outputs/training/mixed-learning-epoch-v1/report.json` and sibling checkpoints/log.

Before submission, finish the CPU schedule/coverage validation, actual-shell
argument check and bounded fresh one-update integration/recovery check on free
nodes. Reuse accepted native-AD evidence; no new standalone gradient audit.
Paid submission requires explicit approval of the above66.24yuan envelope.

## Preparation evidence and immediate handoff

Implementation:04992ef (runner/manifests/launchers),3146c8b (full-pool metadata),
afb7b5f (cluster Bash compatibility). Local plan/runner suites24 passed; Ruff
and diff checks pass. Target-shell metadata/train/integration dry-run and shell
syntax checks pass. The initial non-dry free launch failed before Python because
old Bash treats an empty array expansion under nounset as unbound; afb7b5f uses
a nonempty mode/argument array. This used no paid resources.

Free test02 schedule preparation completed124.86s, no Teacher payload reads or
parameter updates. Metadata validation PASSED. Actual daily coverage is346539
unique days over535 sites,44 years and17677 point-years,5415 batches with21
terminal replay rows. Recurrent selection has10830 distinct windows covering
198101 unique days over535 sites,44 years and8730 point-years. Within-update
T7/T30 day overlap is zero; across-update repeated-day occurrences2254 are
declared. The daily/recurrent union is not their sum.

Evidence: local `outputs/mixed-learning-epoch-v1-metadata.json`; remote diagnostics
`mixed-learning-5415-recurrent-schedule-v1.json`. The free one-update integration
is running on gln01 GPU0 at immutable afb7b5f in
`runtime/checkouts/mixed-learning-epoch-v1`: Bash93015/Python93060,90min ceiling.
Log `runtime/outputs/diagnostics/mixed-learning-epoch-v1-integration-v2.log`;
report `runtime/outputs/diagnostics/mixed-learning-epoch-v1-integration.json`.
Next collect update/physical/recovery result; do not inherit the integration
checkpoint in the paid fresh run. No paid job has been submitted or approved.
