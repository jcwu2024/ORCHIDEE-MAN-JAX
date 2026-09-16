# Two-route integration results and continuation

Latest: dual v2 failed on peer validator device placement; repaired v3 is
running free. See final section. Paid scientific learning continues independently.

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

## Free continuation launch

At server time2026-09-16 10:45:08, launched outer Bash73219 on gln01 after
both GPUs were confirmed idle. Budget code is e3150b9 in
`runtime/checkouts/mixed-learning-budget-profile-v1`. Target-shell syntax and
actual `--dry-run` expansion passed before launch. The dual child retains
dea4083 in `runtime/checkouts/mixed-learning-parallel-routes-v1`.

Under `runtime/outputs/diagnostics/`:

- `mixed-learning-routes-followup-v1.log`: aggregate start/exit statuses.
- `mixed-learning-budget-profile-v1.{json,log}`: singleGPU0 budget profile.
- `mixed-learning-dual-profile-v2.{json,log}`: subsequent twoGPU retry.

The wrapper can be checked on gln01 with
`ps -p 73219 -o pid,etime,pcpu,comm`. Its lifetime includes both children;
consult individual logs to distinguish which diagnostic is active. GPU1 is
not occupied during the first diagnostic. No paid submission or learning
outcome is implied by this launch. Do not rerun either output identity.

## Dual v2 device-placement failure and v3 repair

V2 started11:01:17 and exited1 at12:27:22 server time, Python5120.9666s.
Error: `Buffer passed to Execute() as argument 0 to replica 0 is on device
cuda:1, but replica is assigned to device cuda:0.` The traceback identifies
`wrap_separate_checked` calling its compiled validator after the peer raw
backward. This is a device-placement runtime failure, not a scientific-accuracy
or gradient-numerics rejection. No real-model dual parity/timing exists yet.

Primary daily/T7/T30 cache hits are explicit in the log; compile calls took
102.48/224.43/225.50s including lowering/loading. Secondary T7/T30 compiled
in2111.07/2148.48s. All five main programs completed compilation. The
complete serial reference remains at `mixed-learning-dual-profile-v2.serial-reference.pkl`;
do not erase it or the persistent cache. JSON copied locally under outputs.

Repair b31c523 adds optional explicit device binding to the existing split
wrapper, using SingleDeviceSharding for validator input/output plus explicit
argument placement. Dual callers pass their device. The default single-device
wrapper remains unchanged in behavior; paid checkouts are immutable0d18d3c.
No loss, architecture, optimizer or AD mathematics changed. The prior synthetic
dual probe omitted the lazy split-validator path; this was a test-coverage gap.

New `probe_split_validator_placement.py` exercises primary-captured constants,
primary then threaded peer validation, T7/T30 shapes, changed expected values
and repeated calls with one compilation per wrapper. Local twoCPU JAX0.10.1
and actual twoGPU JAX0.4.38 probes pass; deployed elapsed2.3838s. Existing6
split/dual tests pass. Deployed runtime preflight passed60.0869s. Importantly,
the legacy control also passed on these small inputs: this is positive evidence
for explicit placement, NOT an exact reproduction or full-model acceptance.
The precise inference/capture interaction behind the old full-model assignment
is not isolated beyond the observed wrong validator executable device.

Actual-shell dry-run passed; both free GPUs idle before launch. V3 runs from
`runtime/checkouts/mixed-learning-dual-placement-v1` atb31c523, Bash100711,
60min ceiling, Python100778 confirmed active. New diagnostics basename `mixed-learning-dual-profile-v3`,
compiler cache logging enabled. Reuse all existing main-program caches and
preserve prior reports. Collect full gradient/AdamW/physical parity and timing
before any dual adoption. No paid submission was made for this repair.

Graph project/coverage tools returned Transport closed; bounded source review
used the exact traceback files and tests, without exhaustive impact claims.
