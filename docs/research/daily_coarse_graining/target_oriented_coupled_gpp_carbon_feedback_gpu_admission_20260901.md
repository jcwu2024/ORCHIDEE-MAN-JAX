# Coupled GPP/Carbon-Feedback Real-V6 GPU Admission

Date: 2026-09-01

Status: passed; zero-training GPP-to-next-state reachability audit is next

Job: `14897461`

Git commit: `989a457db945e56c4c392d3b8589b1375dc3f7ab`

Report SHA256:
`cf8eb1d2ba66c70a2d8580caef32647d59765f863c0043a749761a9148d7e623`

Architecture contract SHA256:
`c7f81436ff6ba1179b01adad0f76b47ddd2aab68208e97719c2c5464f3e7e74d`

## Result

The zero-update real-v6 admission passed on one `gnall` V100. It loaded the
accepted broad update-1000 structured-GPP checkpoint and evaluated one
train/train day from landpoint `001.0-071.0`, year 1961, without constructing
an optimizer or writing a checkpoint.

All 15,960,195 first-stage trainable coordinates have finite gradients. All
190 parameter leaves are nonzero, including both carbon-state cross-attention
blocks, all three residual injections, and every inherited structured-GPP
family. The complete candidate has 109,464,359 parameters; the retained
93,504,164-parameter parent remains frozen and exactly unchanged.

The candidate preserves every non-GPP output bit-exactly relative to its
parent for the same day-start input. `vcmax25` has a finite nonzero GPP
gradient (`0.0106969182`), while all declared maintenance, allocation, and
mortality parameter routes have exact-zero GPP gradients.

The complete bound daily transition produces finite values for every declared
next-state coordinate. Water and carbon budget residuals are both exactly
zero. The eager and JIT process outputs agree within the frozen tolerance.

## Runtime

- Slurm state and exit code: `COMPLETED`, `0:0`;
- elapsed allocation: `00:22:33`;
- JIT compile and first forward: `216.743 s`;
- cache-hot forward replay: `0.057055 s/day`;
- trainable-gradient compile and execution: `206.302 s`;
- complete-transition compile and execution: `271.238 s`;
- peak live JAX device allocation: `1,547,798,784` bytes.

The roughly 25.56 GB device-memory limit reported by JAX is the allocator
limit, not live tensor use. The measured peak live allocation is about 1.55 GB.

## Execution Boundary

The run performed zero optimizer updates, created no optimizer, and wrote no
checkpoint. It did not regenerate Teacher v6, use v5, read validation values or
the sealed test, reconstruct 48 forcing records, or run a subdaily state scan.

The machine-readable report remains under the Explore1000 runtime root at
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-carbon-feedback-real-v6-14897461.json`.

## Next Gate

Run a zero-training dependency audit of the complete daily operator. Treat
`daily.fast_gpp` as the differentiated process boundary and identify the
coordinates of next-day `slowproc_stomate_previous_step_state` with finite,
nonzero derivatives. Group the admitted coordinates by source field and
freeze that inventory before defining the one-day state-anchor loss.

Objective coefficients, the 48-days-per-point-year direct monitor, the
train-only 7/30-day monitor, a discarded one-update preflight, the paid
training ceiling, and the stop rule remain subsequent frozen gates. No
training is authorized by this admission.
