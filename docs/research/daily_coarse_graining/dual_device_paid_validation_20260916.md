# Dual-device completion plan and paid validation envelope

Latest: job15230086 COMPLETED0:0 in28m45s. Real-model update parity passed;
warm12.5849s ->7.2221s,1.7426x speedup. Full results/limits are recorded in
`dual_device_paid_results_20260916.md`. Preparation and submission records below
are retained as history. No duplicate/retry or longer training authorized.

User prioritizes completing the dual-device route after both learning pilots
finished and permits paid GPUs so the run has a Slurm job identity. This file
defines the concrete job; submission awaits approval of the resource/cost
envelope below. There is no new scientific training arm or automatic retry.

## Repair and bounded evidence

V4 already passed the explicitly boundGPU1 split validator. It subsequently
failed in `recurrent_branch` at `norm(gradient)`: a JAX0.4.38 persistent cache
entry assigned toGPU0 was deserialized forGPU1. The same cache-key mechanism
affects other helper kernels, so fixing only the named norm call is insufficient.

`device_helper_cache` now covers whole daily and recurrent execution scopes,
their input preparation, and primary-device gradient merge/finite checks/AdamW.
Each compilation thread gets a platform/process/device cache-key suffix, with
the previous hook preserved outside that scope. Existing corrected validator
suffixes are retained. Expensive daily/T7/T30 AOT raw program compilation stays
outside the helper scopes, retaining its existing cache identities. No change
to model, loss, gradient formula, clipping, optimizer or parity tolerance.
The Slurm launcher also respects allocated `CUDA_VISIBLE_DEVICES` rather than
unconditionally selecting physical0,1.

Before the paid job, run a tiny synthetic full-profile regression on the same
deployed JAX/CUDA runtime: persistent writes forced in an isolated probe cache;
exercise actual serial/dual orchestration, microbatch accumulation, T7/T30,
validators, norms, gradient merge, finite checks, AdamW and weight transfer.
Require the old helper path to reproduce the assignment error and the fixed
path to pass fresh and persisted-cache reloads. No Teacher reads. This tests
execution, not real-model scientific accuracy or useful speedup. Results are
recorded below when available. Reuse prior unchanged AD/runtime evidence.

## Paid job, ready for approval after the tiny regression

- Partition gnall; one node, one process,2 V10032 GPUs,8 allocated CPUs,64GiB RAM.
- Slurm limit90min; Python timeout85min; no requeue or automatic paid retry.
- Previous full-profile cache startup took about22min. Expect25-40min if the
  same raw cache entries load; this is an estimate, not a completion guarantee.
- Additive maximum: CPU8*1.5*.07=0.84yuan; GPU2*1.5*2.20=6.60yuan;
  total7.44yuan. Scheduler/job IDs and actual elapsed time will be recorded.
- Working directory:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-dual-paid-v1`.
- Command: `sbatch scripts/hpc/job_mixed_learning_dual_profile.sh`.
- Log/report under runtime diagnostics:
  `mixed-learning-dual-paid-<jobid>.log` and `.json`; serial reference `.serial-reference.pkl`.
- Existing trained200 checkpoint is used only for fixed-weight execution/cost
  diagnostics. All update proposals are discarded; no final model is claimed.

## Completion criteria and follow-up

Keep the same D64/micro8+.5T7+.5T30 update. GPU0 owns daily accumulation and
AdamW;GPU1 owns T7 thenT30 from the same old weights. Merge onGPU0, clip once,
apply AdamW once, include updated-weight transfer toGPU1 in timing. Preserve
the single-GPU reference before peer execution. Compare full gradient,
parameters, optimizer state, scalar diagnostics and physical counters at the
existing tolerances. Then run three interleaved warmed repeats with reads and
checks included. Report speedupS, GPU-hour ratio2/S and total charge ratio
1.7971/S for the8CPU envelope; do not equate less wall time with lower cost.

Passing establishes same-update numerical/timing admission only. Rotating-data
continuous two-GPU training and checkpoint recovery remain a subsequent bounded
integration step if the measured speed/cost tradeoff is useful. Do not switch
either completed learning arm to dual execution or submit a longer training
job automatically. Neither a speed result nor tiny numerical parity establishes
long-term scientific accuracy.

## Preparation completed

Immutable deployed checkout is b2e1562 at the path above. Local6 split/dual
tests, Python compilation/Ruff and diff check pass. Sol/high independently
reviewed all relevant helper/device edges; Luna/high implemented the bounded
synthetic fixture, which calls the actual production profiler. Graph coverage
was metadata-stale, so exact current sources were read for these conclusions.

Deployed JAX0.4.38 probe `dual-profile-cache-v1` PASSED exit0 in38.13245s on
gln01 GPUs0+1. Its negative control reproduced the exact XLA assignment error
afterGPU1 T7 validator compilation. Two positive complete profile runs passed,
each with3 interleaved timing repetitions; gradient, updated-parameter and
AdamW-state tree errors were zero, physical counters/scalars agreed. Active
clipping was exercised. Memory caches were cleared between positive runs;
compiler logs show disk-cache hits for corrected helper and validator entries.
The isolated tiny cache never changes thresholds for the production cache.
No real-model timing or persistent training outcome is inferred from this test.
Local evidence: `outputs/dual-profile-cache-v1.{json,log}`; remote same basename
under runtime/outputs/diagnostics. Prior unchanged expensive AD evidence reused.

Target shell syntax and actual job `--dry-run` passed on gln01 with simulated
Slurm job424242 and CUDA_VISIBLE_DEVICES=3,5. Expanded launcher preserves
GPU_DEVICE=3,5,85min inner timeout, reviewed checkout and job-specific outputs.
The fake job number is a dry-run label, NOT a scheduler submission. Live gnall
inspection at15:29 server time found4 idle nodes; availability may change.

All preparation is complete; no paid job submitted. Await explicit approval
of2GPUs/8CPUs/64GiB/90min/7.44yuan envelope. After approval, use exactly the
declared sbatch command and preserve b2e1562 while running. Do not repeat this
tiny probe or the old gradient audit merely for unchanged submission metadata.

## Approved submission

Executed `sbatch scripts/hpc/job_mixed_learning_dual_profile.sh` from the exact
declared clean checkout, unchangedb2e1562. Slurm job15230086 submitted15:54:49
and started15:54:50 on2026-09-16 server time, nodeibc13b03n01. Verified RUNNING,
2GPUs/8allocatedCPUs/64GiB/90min, no requeue/restarts; deadline17:24:50. The
launcher confirms codeb2e1562, allocatedGPU0,1 and the job-specific output.
Logs/report: runtime/outputs/diagnostics/mixed-learning-dual-paid-15230086.log
and.json; inner85min timeout. Do not edit the running checkout or resubmit.

Check active status with `squeue -j 15230086`; completed status/exit with
`sacct -j 15230086 --format=JobID,State,ExitCode,Elapsed`. Scheduler RUNNING
does not certify numerical parity or speed; collect the full report when done.
