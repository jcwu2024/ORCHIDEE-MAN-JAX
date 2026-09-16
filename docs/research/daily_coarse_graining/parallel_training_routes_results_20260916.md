# Two-route integration results and continuation

## Staged learning: transition integration accepted

Paid gnall job15209522, immutable eba219d, completed with Slurm exit0:0 in
19m55s on ibc13b02n01. Python elapsed1128.38686s. Resources were1V100,
8 allocated CPUs and64GiB; additive charge estimate0.92yuan against4.14yuan cap.
Report: `outputs/mixed-learning-stages-integration-paid-v1.json` (local),
same basename under remote runtime/outputs/diagnostics. Log:
`mixed-learning-stages-integration-paid-15209522.log`.

Fresh initialization completed one daily-only update then one mixed update.
Daily batch indices0/1 were consumed; recurrent cursor remained0 after daily,
then consumed recurrent row0 and advanced to1. Total128 daily draws, one T7
and one T30 window. Initial and both milestone daily/recurrent/development
physical and population checks pass. Checkpoint forward replay is true.
Scientific acceptance remains false: this validates the executable phase
transition and recovery, not trained accuracy or the merits of staging.

First daily update28.859s and first mixed17.207s include startup effects.
Their ratio MUST NOT determine the matched-compute comparison budget.

## Dual device: startup timeout, not a numerical result

Free gln01 dual profile v1 at dea4083 reached the90min external timeout,
wrapper exit124. Bash7920/Python7963 have exited. The JSON's last running
status is stale; it did not receive a final emission after external termination.
Metadata took7.1s, profile setup reached148.2s. Daily_GPU0 compiled in990.599s;
T7_GPU0 in2150.379s. T30_GPU0 started at elapsed3294.6s and did not complete.
No serial reference, GPU1 real-model executable, parity or warmed speed result
was produced. Do not report a speedup or dual numerical acceptance.

Evidence: `outputs/mixed-learning-dual-profile-v1.json` locally; remote
runtime/outputs/diagnostics contains that JSON and matching `.log`.
Completed compilations may be reusable through the existing persistent cache;
confirm retry hits rather than assuming all startup costs disappear.

## Independent next steps

1. Use the existing single-device compilation/profiling path and trained200
   diagnostic checkpoint to warm complete daily-only and mixed updates.
   Include sample reads, physical checks, gradient merge/clip and AdamW;
   discard proposals. Interleave repeats and report timing spread.
2. Freeze joint update count by the already declared formula
   round_to_nearest10(400*t_daily/t_mixed+400), before inspecting learning
   results. Staged stays400daily+400mixed, same fresh seed/full parameters.
3. Continue bounded dual startup/parity/timing with cache evidence and a fresh
   output basename. Preserve existing reports; do not repeat integration or
   rewrite AD. Dual completion does not block the staged pilot.
4. Prepare the final paid pilot resource proposal after timing. The provisional
  1V100/8CPU/64GiB/8h envelope is22.08yuan maximum and is NOT submitted or
   authorized by the earlier90min integration approval. The5415-update job
   remains paused. Fixed-data timing is not rotating-data throughput acceptance.

The continuation uses a small free-node sequence: independent single-GPU
budget timing first (90min ceiling), then dual v2 (150min ceiling) if no other
GPU process is present. Dual reuses the original immutable dea4083 checkout
and device order0,1; cache compiler messages are logged to distinguish hits
from new compilation. No new derivative implementation or precision change.
Both diagnostics keep separate logs and report identities. The sequence does
not wait indefinitely for free GPUs and does not submit any paid job.

Implementation checks:47 focused tests passed (including stage/checkpoint
coverage); parent independently ran40 profiling/runner/stage/dual tests, all
passed. Independent Terra source review found no blocking issue; Sol implemented
the budget replay. Shell syntax and diff checks pass. Budget profiling skips
legacy HLO precision-text inspection, retains normal compile/memory checks,
warms both objectives and interleaves3 repeats. This is orchestration evidence,
not a new gradient or scientific acceptance claim.
