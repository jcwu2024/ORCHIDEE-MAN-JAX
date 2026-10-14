# Fixed CPU48 v5 proposal — not submitted or authorized

The accepted identity repair is packaged at exact execution SHA
`0100c8b09be5137b822173e0a0be66cb64939727`, ref `codex/global-cpu48-b64-v5`,
on historical parent `60dccbddde10be4295aab7e7fe55fa7864d3c3f3`.
Only three accepted production files and three tests come from research
`978946bbf90593b5ea1c93d414dc6410a82b3f59`; physical/scientific files are unchanged.

Proposed allocation: **cnall, one node, one task, 48 allocated CPUs, zero GPUs,
3600 seconds**, nonexclusive, no requeue, no added memory or placement restriction.
Unverified worst-case rate scenarios are **CNY 2.40 / 3.36**
(`48 × 1 hour × 0.05 / 0.07`). Old paid grants remain closed.
Preparation/deployment/free checks do not authorize this paid command.

After a new exact approval, the literal command would be:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/PROPOSED_SUBMISSION.sh
```

The sbatch working directory is the new clean immutable checkout
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v5`.
The proposal fixes canonical CPU Python, exact candidate SHA and the normal
new scalar-admission path/hash (`4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5`).
It clears inherited overrides before selecting these paths. `--dry-run` uses
the exact same environment and unchanged launcher, and submits no job.

The unchanged launcher/coordinator retain their established v4 output namespace:
the next run root is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/${SLURM_JOB_ID}`.
A new scheduler job ID produces an isolated directory, checked absent at launch.
This directory name does not change the execution SHA or repeat v4 job15853683.
The log is `runtime/logs/global_cpu48_b64_v5_%j.txt`; scontrol evidence retains
the launcher's existing unique `scontrol-job-${SLURM_JOB_ID}` filenames.
No production logic was changed for a version label.

The proposed scientific workload remains one native FULL64 B64 vmap, 64 true
points, Days 2–8 and 9–15, 896 point-days, 760 native leaves/masks/dtypes, original
time recursion, precision, atol1e-8/rtol1e-10 and every original gate. It reuses
job15820467's scalar snapshot and audit; no scalar generation or GPU/B2048 work.
Its prior 61/64 passes, three finite failing points/four comparisons,
sealed-reset-proof-only limitation and scientific_acceptance=false remain.

If separately approved, this is a **single attempt**: any substantive failure
or timeout preserves evidence and stops. No repair/resubmit, extension,
cancellation or reuse of old grants is implied. Follow the sole job to terminal,
collect with the unchanged pinned CPU48 collector, and count the new whole parent
allocation separately from old scalar job15820467. Preparation's one-CPU pass
does not certify real48 admission, physical compilation/run, candidate numerical
gates, warm throughput or scientific acceptance. No paid command was executed here.
