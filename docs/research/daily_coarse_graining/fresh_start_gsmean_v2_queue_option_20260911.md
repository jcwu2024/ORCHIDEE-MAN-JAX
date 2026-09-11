# Gsmean v2 resource check and queue option

The user authorized direct submission if a free GPU is available, or if a paid
node has an available GPU; if neither is available, consult before queueing.

Free gln01 initially showed about 471 MiB and 0% on each GPU, but both had live
Python GPU processes. A follow-up showed 2505 MiB and 80%/78% utilization: the
low utilization was temporary, not an available unused GPU.

Scheduler commands use `/rmprog/slurm/v22.05.7/bin/` (not on the default SSH
PATH). `sinfo -N -p gnall -O NodeList,StateLong,Gres,GresUsed,CPUsState` reports
all ten gnall nodes with all eight V100s allocated. Even ibc13b03n07, marked
IDLE with CPUAlloc=0, has GresUsed gpu:v100:8 and AllocTRES gres/gpu=8.
Therefore no available paid GPU was established and no job was submitted.

## Concrete optional queue submission

- Partition: gnall; one node, one V10032GB, eight allocated CPUs, 32 GiB host RAM.
- Wall limit: one hour; application timeout55m to leave cleanup time.
- Maximum additive charge: GPU CNY2.20 + CPU8*1*0.07=CNY0.56, total CNY2.76.
- Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
- Command: `/rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_fresh_start_gsmean_v2.sh`.
- Same v2 seed/data/architecture/applicability/loss/optimizer and at most 12
  all-parameter updates, after full numerical/physical/AD admission. No new
  Teacher data or inherited weights. No node pinning, so the next compatible
  free GPU can be allocated.
- Unique job-specific report/checkpoint directory and `%j` log naming prevent
  overwriting prior experiments.

The user subsequently approved paid queue submission. The frozen protocol's original paid-submission
field records the initial free-node proposal; execution authority is the user's
current session instruction, not a JSON checkbox. No paid submission occurs
without that instruction's resource-availability/queue conditions being met.

## Submitted queue job

Job **15005758**, submitted from gln01 under executable checkout **85e6b39**.
Scheduler SubmitTime: 2026-09-11T21:29:54. Submission command:
`/rmprog/slurm/v22.05.7/bin/sbatch --parsable scripts/hpc/slurm_fresh_start_gsmean_v2.sh`.
The returned job ID and `scontrol show job` confirm gnall, one GPU, eight CPUs,
32G host RAM, one-hour limit and the intended working directory/command.

Initial state: PENDING, reason Priority, no allocated node or runtime. The
scheduler's current estimated start is 2026-09-13T12:51:52; this is a changing
scheduler estimate, not a guaranteed appointment. Maximum additive fee remains
CNY2.76. No second job or automatic retry is authorized.

Log: `runtime/logs/gsmean-v2_15005758.txt`.
Report/checkpoint directory:
`runtime/outputs/training/fresh-start-gsmean-v2-15005758/`.
`report.json` records admission and training; optional `report.checkpoint.pkl`
is the fresh prototype checkpoint. Preserve this server checkout while queued
and running. Next inspect this job's status/log/report when it starts or ends.

## Switched to available free GPU

On the user's next resource/queue-efficiency request, both free gln01 GPUs were
empty (0 MiB, 0% utilization, no compute processes). Paid job15005758 was still
PENDING/Priority; its scheduler start estimate had moved to September13 13:57.
Root placed the pending paid job on hold with `scontrol hold 15005758` and
verified PENDING/JobHeldUser before launching the same approved probe free.
This is not cancellation and does not promise preservation of queue priority;
the held job cannot start or duplicate the free run until explicitly released.

Free run: gln01 GPU1, executable checkout85e6b39, Python PID107826,
RUN_TIMEOUT60m, same scientific protocol and at most12 updates. Report/log:
`runtime/outputs/diagnostics/fresh-start-gsmean-v2-free.json` and
`runtime/logs/fresh-start-gsmean-v2-free.log`.
No new paid job or larger resource commitment. Keep the checkout stable during
the free process. Track the free result next; do not unhold a duplicate while
it is running.

The user's queue-efficiency concern is accepted as a planning constraint:
future paid experiments should bundle preregistered admission, larger training
and monitoring/checkpoints in one persistent process where dependencies permit,
so compilation and queue wait are amortized. Multiple paid jobs should represent
independent justified experiments, not repeated probes of the same unresolved
boundary. Larger budgets/sampling/objectives must still be specified before
submission; this request does not silently increase the current12-update run.

## Canceled on user instruction

The user explicitly requested cancellation after the free run ended. Root
verified job15005758 was PENDING/JobHeldUser, then ran `scancel 15005758`.
Scheduler confirms CANCELLED, RunTime00:00:00, no allocated node. Recorded
cancel time is 2026-09-11T23:28:06. No paid compute ran. Do not release or
resubmit this job; next work is free-node FD convergence diagnosis.
