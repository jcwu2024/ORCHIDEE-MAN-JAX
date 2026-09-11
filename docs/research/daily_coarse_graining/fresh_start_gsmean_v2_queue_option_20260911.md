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
