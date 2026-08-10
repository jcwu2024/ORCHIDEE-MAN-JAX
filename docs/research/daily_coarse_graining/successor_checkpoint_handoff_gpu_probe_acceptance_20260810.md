# Successor Checkpoint Handoff GPU Probe Acceptance

Date: 2026-08-10.

Status: accepted. This closes only the free-GPU serialization preflight; no
training checkpoint is promoted and no paid fit is authorized by this report.

## Execution Identity

- Consumer commit: `c15cd86f3f3de8ceb20781abc603187cc6b272cc`.
- Clean checkout:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/checkpoint-handoff-5b77cb2`.
- Node/device: free `gln01`, physical GPU 1, Tesla V100-SXM2-32GB.
- Runtime backend/device visible inside the container: `gpu` / `cuda:0`.
- Checkpoint handoff contract SHA256:
  `a587b8a93b57bb69f06776aa80fd2268f6ebe2f2e13c07b70e45a3d6632315b0`.
- Probe report:
  `runtime/outputs/training/successor-checkpoint-handoff-gpu-probe-c15cd86/probe_report.json`.
- Probe report SHA256:
  `a923d6371473bc19bef9ea89e3a390a4798b02ac13e23782d2520ab22616cf5a`.

## Result

Both declared arms began as JAX arrays on the GPU. Their parameter and
optimizer trees were transferred to host memory, staged through the production
atomic checkpoint-set writer, rehashed, loaded through the production reader,
and compared leaf by leaf. The round trip was exact.

The temporary set contained two 1,562-byte arm files and a manifest with
SHA256 `3466ab12...7be02`. The temporary directory was removed before the
probe report was published. An independent directory listing found only the
1,242-byte `probe_report.json`; no checkpoint payload remained.

The serialization operation took `0.7236` seconds. Initial container startup
and JAX backend discovery brought the outer command to about 70 seconds, but
there was no training or Teacher preparation.

The report records:

- `training_updates = 0`;
- `production_checkpoint = false`;
- `teacher_data_read = false`;
- `teacher_data_regenerated = false`;
- `historical_v5_used = false`;
- `paid_gpu_allocation_used = false`.

The consumer checkout remained clean after the probe. The accepted prepared
manifest and its producer checkout at `d493699` were not modified or read by
this serialization-only test.

## Gate Effect

Local failure-path tests and the free GPU serialization test now jointly close
checkpoint persistence preflight. The next possible action is to present a
fresh paid `gnall` approval block for the exact 500-update fit at execution
commit `c15cd86`. The fit remains unsubmitted. Its successful completion would
open, not pass, the separately authorized 7/30-day rollout screen.
