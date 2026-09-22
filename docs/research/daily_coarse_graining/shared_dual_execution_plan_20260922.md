# Shared dual-device execution comparison

The user requested concurrent use of idle paid GPUs on2026-09-22. Prepared
this comparison alongside free microbatch wrapper38944, without modifying
its immutable checkout or waiting for its result. This is execution work,
not a new architecture, objective, data-generation or learning experiment.

## Fixed method and acceptance

Use the shared100-update checkpoint from4408daf as a diagnostic asset only.
Freeze `shared_dual_execution_benchmark_v1.json`: artifact742 for fixed-weight
replays; artifacts742 and275 for two consecutive changing-data updates.
Both paths use the same scalar32 daily mean plus oneT7 and oneT30 loss,
unchanged precision, native checkpointed AD, full-network gradients and AdamW.
Each recurrent objective already contains its division by9; combine T7 and
T30 with weights1.0, not the earlier architecture's weights0.5.

Serial reference executes everything on GPU0. Dual execution runs daily work
on GPU0 and T7/T30 sequentially on GPU1, then moves recurrent gradients toGPU0,
adds components, clips/updates once and synchronizes new parameters toGPU1.
Compile device-bound executables under `device_helper_cache` to respect
deployed JAX0.4.38 persistent-cache placement. Preserve a complete serial
reference before compiling the second device's recurrent graph.

Require finite gradients/proposals, zero hard physical violations and
full-gradient/full-parameter/AdamW parity at rtol1e-5/atol1e-6. Check the next
step after peer synchronization on different data, not only repeated source
weights. Do not loosen tolerances to obtain a pass. No proposals become
retained learning checkpoints; preserve bounded diagnostic evidence.

One warmup and three timed replays per execution path, interleaved after
compilation, establish complete update time including transfers and peer sync.
Report component timing/balance, speedup, GPU-hour and estimated charged-time
ratio, compilation separately, and memory on both devices. These preloaded
data timings are not end-to-end dataset throughput. Per-graph compiled live
memory ceiling24GiB; report actual device peak separately. Any passed scalar
dual comparison remains separate from admission of combining dual execution
with a later admitted microbatch size.

## Bounded deployment and pending paid approval

Launcher: `scripts/hpc/run_shared_dual_execution.sh`; Slurm entry:
`scripts/hpc/job_shared_dual_execution.sh`. Existing free-only launchers stay
free-only. Resolve CODE_ROOT explicitly from a new immutable checkout because
Slurm runs a spooled copy of the job script. CPU mode validates source100 and
protocol prerequisites ontest02; actual target-shell dry-runs cover argument
expansion. Run tiny two-device tests in deployed GPU Python using CPU devices
before paid execution. At job start run bounded deployed runtime/interface
and two-visible-GPU checks, then the real benchmark. No repeated long JVP
suite for the unchanged objectives.

Proposed allocation: gnall, one node, two V10032GiB, eight allocated CPUs,
64GiB host memory, three-hour hard limit. Main comparison has170min ceiling,
plus two synthetic checks of at most5min each. Finish and release immediately
on completion; three hours is a ceiling, not a predicted runtime. At current
documented rates, max additive cost is(2*2.20+8*0.07)*3 = CNY14.88.
The two-device recompile dominates uncertainty; do not imply the benchmark
itself needs three hours of warm computation. No paid job has been submitted.

## Prepared checkout and completed checks

Implementation8cce49e is deployed independently at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-dual-8cce49e`.
Sol/high implemented the bounded executor; the lead reviewed integration,
added repeated non-persisting replay checks and prepared deployment.
Local tests: two new integration/preparation tests and seven existing runner
tests pass; Ruff passes. The new integration test uses a separate process
with two CPU devices and checks repeat-without-mutation, two changing-data
updates, clipping, full optimizer/gradient/parameter parity, peer placement,
zero-gradient components, nonfinite inputs and physical failures. It is not
real-model GPU acceptance. The separate synthetic device runtime check also
passes two local CPU-device updates.

On free test02, the actual GPU Python environment (JAX0.4.38, CPU backend)
passes both new tests in34.71s. Source100 method/checkpoint/metadata preparation
passes in31.0s. Actual target-shell CPU/GPU dry-runs and the Slurm entry's
simulated allocated-device argument expansion pass. The original free38944
benchmark remains independent and was running at the last observation.

Pending approved submission from cln01, working directory and exact command:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-dual-8cce49e
CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-dual-8cce49e /rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/job_shared_dual_execution.sh
```

The script declares gnall2GPU/8CPU/64GiB/03:00:00, CNY14.88 worst case.
No `sbatch` has been invoked. Main real-model GPU placement, numerical parity,
memory and measured speed/cost remain the purpose of the proposed allocation.

## Approved launch

The user approved the above exact request. Submitted once from cln01 as
15328832 on2026-09-22 at16:32:58. Slurm reports RUNNING onibc13b02n03,
start16:33:09 and hard end19:33:09, all times from the cluster clock.
Allocation matches2GPUs/8CPUs/64GiB/3h. Checkout8cce49e remains immutable.
Output prefix:`runtime/outputs/diagnostics/shared-dual-15328832`.
The earlier pending/no-submission statements describe the preparation phase.
Free microbatch38944 is still running independently. Collect the real GPU
runtime checks and benchmark stages; launch alone establishes no speedup.

Query with `squeue -j 15328832`; inspect the report/log at the prefix above.
After queue removal, use `sacct -j 15328832 --format=JobID,State,Elapsed,ExitCode`.
Cancel only on explicit instruction with `scancel 15328832`.
