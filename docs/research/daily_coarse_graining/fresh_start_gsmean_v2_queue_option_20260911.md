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

Queue authorization is pending. The frozen protocol's original paid-submission
field records the initial free-node proposal; execution authority is the user's
current session instruction, not a JSON checkbox. No paid submission occurs
without that instruction's resource-availability/queue conditions being met.
