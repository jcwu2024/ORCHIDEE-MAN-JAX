# Fixed1024 job15662711: two strict stages accepted, budget closed

Updated2026-10-09 Asia/Shanghai. The single approved job completed `0:0` on
ibc13b03n05 in3371 parent seconds. Exact execution code remains clean at
`bd9e4c04af68d77e79b984bfc8aa064977f893d3`; allocation was1node/1task,
4 Slurm CPUs and1 V10032GB,90min/no-requeue/no-exclusive. Submission/start/end
were01:01:36/01:02:02/01:58:13 Asia/Shanghai (UTC17:01:36/17:02:02/17:58:13).

The deployed strict collector passed on test02, canonical CPU env, affinity0–3,
OMP/BLAS1 and Python `-B`, in149.98s,exit0:2 valid stages,0 withheld,0 incomplete.
Each has1027 passing gates,760 native output leaves and1024 saved-array
FULL64 comparisons. Both use the same14336 lane-days:64 genuine FULL64 points
repeated16 times, continuous Days2–8 then9–15,48 steps/day, native dtypes/full
outputs/masks/fixed tolerances. Historical job15657822 supplies numeric/schema/
compiler/RSS provenance only; none of its timing is doubled into this comparison.

| New same-process stage | Core median [full range],s | Wrapper median [full range],s | Wrapper lane-days/s | Warm wrapper CNY/lane-day, primary |
| --- | --- | --- | --- | --- |
| B512×2 chunks |45.543218 [45.537509–45.564560]|52.195214 [52.032337–52.241993]|274.661200|2.427233e-6|
| B1024×1 chunk |40.864703 [40.859622–40.882893]|47.405972 [47.297268–47.626702]|302.409157|2.204519e-6|

Each statistic comes from five synchronized samples. B1024 wrapper is1.101026×
as fast and has9.175634% lower median time/warm unit cost in this comparison.
It is the lower measured warm unit cost of these two shapes, not a proven
optimum for independent inputs or production IO. Warm figures exclude startup,
checkpoint IO and full parent allocation; they are not total experiment cost.

| Stage | Lower,s | Compile,s | First full1024 call,s | Host stack,s | Transfer,s | Checkpoint serialization / atomic write,s |
| --- | --- | --- | --- | --- | --- | --- |
| B512×2 |92.842230|889.883491|45.714354|5.398977|2.561924|0.480900 /1.395775|
| B1024×1 |64.840254|990.697544|41.100882|5.584297|0.816445|0.562827 /0.342320|

Parent-only accounting is0.936388889 GPU card-hours and3.745555556 allocated
CPU-hours: primary CNY2.247333 / conservative additive CNY2.322244, below
approved CNY3.60 /3.72 ceilings. Billing remains unverified; queue time is
unbilled and .batch/.extern/step rows are retained without summing them.
The GPU model child wall was3362.76s; saved-array collection is separate.
No K-amortization estimate is reported because complete setup attribution is
unavailable. No new CPU/GPU ratio, Fortran, annual/669-point/50-year,10×,
training/AD or general scientific acceptance follows.

Both full checkpoints are preserved remotely and locally, with local SHA
matching the terminal manifest: B512 440821931 bytes,
`678a116038c5c8ad404dfb1d4eb691e1e4a43b2680871fd7353ef557243b07e6`;
B1024 440822967 bytes,
`6bea479fcf3fa3d81d334c132106641572ca92a58886b26e764bcd7dee6069d5`.
[Terminal summary](terminal-summary.json), [raw collector result](collected-fixed1024.json),
[collector log](job-15662711-collector-log.txt), [raw9 accounting](job-15662711-saved-result/sacct-raw9.txt),
[terminal log](job-15662711-terminal-log.txt) and [artifact index](terminal-artifact-index.json)
are the actual evidence. The verified compact [evidence ZIP](job-15662711-terminal-evidence.zip)
retains exact collector JSON/report/manifest/events/diagnostics; checkpoint binaries
remain in `job-15662711-saved-result/capacity` and the immutable remote run root.
No original results were deleted.

Startup identity is GPU global0,UUID GPU-51897aa5-dac4-9d20-523e-830c362cf7a6,
V100-SXM2-32GB,driver525.85.12; selected-card process list was empty.
Affinity24,25,80,81 maps2 physical cores for4 allocated Slurm CPUs; startup
load18.67,19.75,20.12. The approved n05 was MIXED at submission, not idle or
reserved. Compiler buffer estimates and GPU snapshots are not VRAM peak
measurements. Stage cumulative host peak RSS was24653754368/27865956352 bytes;
these are host memory. Original planning arithmetic remains separately labeled.

[Paid ledger](../server-cost-preparation/PAID_EXECUTION_LEDGER.json) closes this
independent budget; older budgets stay closed and are not reused. No retry,
new job, cancellation, extension or automation was performed. Prior159 tests
and helper/interface evidence are reused, not rerun. The two ignored `.pyc`
files remain after the earlier automatic approval review rejected cleanup;
no alternate deletion was attempted.
