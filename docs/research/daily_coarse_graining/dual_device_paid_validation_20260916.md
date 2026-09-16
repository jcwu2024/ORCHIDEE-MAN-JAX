# Dual-device completion plan and paid validation envelope

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
