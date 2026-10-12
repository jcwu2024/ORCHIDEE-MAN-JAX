# CPU48 B64 Execution Result

Job `15782454` reached `FAILED`, exit `2:0`, after `00:00:01` on
`ibc12b07n02` with 48 allocated CPUs and 0 GPUs. This was a launcher startup
failure: the launcher read or parsed an empty `AllocTRES`, and the Slurm log
contains `Slurm did not report AllocTRES`. The raw `scontrol` output was not
saved, so the underlying reason the launcher could not read that field is
unknown. The saved
parent accounting row reports `billing=48,cpu=48,node=1`; its exact bytes and
hash are preserved in [terminal-parent.sacct.raw](terminal-parent.sacct.raw).
The matching Slurm log and final attempt marker are preserved beside this
report and indexed in [execution-result.json](execution-result.json).

This assignment used its sole authorized `sbatch` invocation (`1` total). No
run root, stage outputs, model execution, collector execution, or performance
measurements exist. Therefore there is no CPU B64 throughput or scientific
acceptance result. The collector was not run because there were no stage
artifacts to collect.

The whole parent allocation is charged once at 48 CPUs for the one-second
accounted elapsed time. At the unverified planning rates of CNY 0.05 and 0.07
per CPU-count-hour, the scenarios are CNY 0.000666667 and CNY 0.000933333,
respectively; neither is a verified bill.

Comparison boundary: the intended GPU B64 comparator, job `15579869`, covers
the same 64 ordered points and 896 point-days, so it remains the same-workload
reference but cannot be compared for throughput without CPU measurements.
GPU job `15671732` at B1024 x 2 covers 28,672 repeated lane-days and is a
different-total-work reference; it must not be presented as a same-workload
CPU/GPU speed ratio. No ratio is produced by this failed CPU attempt.

The remote runtime registry at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/deployment/registry.json`
records this terminal result under `latest_global_cpu48_b64_execution`. The
prior `latest_global_cpu48_b64_deployment` entry remains preserved.
