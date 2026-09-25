# Saved-Executable Dual-Process Throughput Diagnostic

Status: small deployed GPU worker proof passed; complete-update benchmark running
as of 2026-09-25. No measured full-model dual speedup is claimed here.

## Boundary

The immutable `c599236` method and three trusted JAX 0.4.38 V100 executable
files remain the source of daily micro4, T7 and T30 gradients. The direct-load
two-changing-artifact replay already passed all-parameter and full-AdamW
comparison at `rtol=1e-5, atol=1e-6`. The new diagnostic does not lower,
trace or compile a full-model gradient and does not alter the architecture,
loss, masks, optimizer, training batch recipe or saved reference.

The saved GPU executable embeds logical device 0. A tiny actual JAX 0.4.38
retarget test found that `default_device` and explicit deserialization
assignment do not move its native replica to a second visible GPU. The
diagnostic therefore runs two isolated processes: the parent sees physical
GPU0 as logical device 0 and loads daily/T7/T30; a persistent child sees
physical GPU1 as logical device 0 and loads the same trusted T7/T30 binaries.
The child starts with its CUDA selector set before importing JAX. It uses only
inherited private pipes, with no listening port or per-update files. Standard
output is reserved for framed pickle messages and ordinary child logging goes
to standard error.

For each measured dual proposal, the parent transfers the full current
parameter tree and recurrent cases to the child while it evaluates the full
daily microbatch mean gradient. The child returns both full recurrent
gradients. The parent applies the unchanged sum, physical/numerical checks,
gradient clipping and AdamW once. Timed dual wall includes host/device and
pipe transfers, both recurrent executions, daily execution, gradient merge
and optimizer update; no transfer is hidden outside the timed proposal.

## Checks and Scope

`probe_shared_serialized_dual.py` hooks the existing learning runner's
compile and update boundaries. It compares two actual, changing 32-row
technical updates against serial execution at every full gradient,
parameter and AdamW coordinate, plus losses and optimizer scalars. It checks
the final diagnostic checkpoint against the trusted serialized-replay
reference. The runner's technical checkpoint is not formal training.

The subsequent fixed-weight benchmark times at least three interleaved,
blocked serial and dual complete proposals each for 32 rows and a genuine
64-row daily batch formed from the two distinct 32-row artifacts. The 64-row
case uses one recurrent T7/T30 case and is a throughput proposal, not the
declared training recipe. Distinct artifact IDs do not prove every daily row
is unique, and no unique-coverage claim is made. Reports include daily rows
per second, full-update medians and GPU-hour/estimated-charge ratios, plus
complete-tree parity at the same initial weights.

The deployed small proof compiled only synthetic recurrent gradients on the
parent logical GPU0, serialized them, loaded them in the child logical GPU0
on physical GPU1 and passed two changed parameter/data RPC rounds with full
synthetic gradient and loss comparison. It read no Teacher data and did not
load the full model. Evidence:
`runtime/outputs/diagnostics/shared-serialized-dual-worker-smoke.json`.
The worker's tiny recurrent/total times were 0.8766/0.8848 seconds on the
first round and 0.00416/0.00921 seconds on the second. These are protocol
checks, not full-model throughput measurements.

The full saved-executable benchmark is running through
`scripts/hpc/run_shared_serialized_dual.sh`; its report and log are
`runtime/outputs/diagnostics/shared-split-serialized-dual-20260925.json`
and `runtime/outputs/diagnostics/shared-split-serialized-dual-20260925.log`.
Its throughput, transfer overhead and cost outcome remain pending.

## Code and Local Verification

- `scripts/hpc/probe_shared_serialized_dual.py`: complete-update benchmark,
  immutable-checkout diagnostic overlays and synthetic worker smoke mode.
- `scripts/hpc/probe_shared_serialized_dual_worker.py`: private recurrent
  worker that loads the unchanged trusted T7/T30 executables.
- `tests/unit/test_shared_serialized_dual.py`: artifact, pipe, proposal
  timing and CPU-sized full-gradient/optimizer behavior.
- Targeted local suite: 9 passed, including existing serialized replay and
  two-CPU-device microbatch integration. Ruff and Python compilation pass.

The deployed full benchmark is the only evidence that can establish whether
the process split materially improves complete-update sample throughput.
