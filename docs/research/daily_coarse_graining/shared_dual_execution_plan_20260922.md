# Shared dual-device execution comparison

The user requested concurrent use of idle paid GPUs on2026-09-22. Prepare
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
