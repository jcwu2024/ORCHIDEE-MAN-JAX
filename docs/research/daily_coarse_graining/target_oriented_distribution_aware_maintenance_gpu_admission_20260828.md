# Distribution-Aware Maintenance Real-V6 GPU Admission

Date: 2026-08-28

Status: accepted zero-update execution evidence

Executable commit:
`54ada5444b8700191a9637e1fff939706fa118ab`

Machine report:
`runtime/outputs/training/target-oriented-distribution-aware-maintenance-admission-54ada54/report.json`

Report SHA256:
`376d8016f4c3805e3652b7ef9be7034581b6509edf24654d4231b23afc66ece5`

## Result

The zero-update real-v6 admission passed on free `gln01` GPU 0. It loaded the
exact update-1000 Attention checkpoint, replaced only the 769-parameter direct
maintenance projection, initialized the `9,750,529`-parameter adapter, and
evaluated the complete `93,504,933`-parameter candidate.

All 160 non-maintenance learned outputs are bitwise equal to the parent. Eager
and JIT outputs are finite and agree within the checkpoint tolerance. JIT
compile plus first execution took `176.278 s`; hot replay took `0.06276 s`.
This replay is an admission forward measurement, not yet a complete product
inference benchmark.

Reverse mode differentiated only the first-stage adapter subtree while the
complete frozen parent participated in forward and loss. All `9,750,529`
gradient elements are finite, all 36 adapter leaves are nonzero, and the
context projection, both Attention blocks, both FFN blocks, and final
projection have nonzero norms. Compile plus gradient execution took
`153.185 s`.

The canonical bound-day transition produced all 181 named outputs, finite
declared next state, and exact zero water and carbon budget residuals. Its
compile plus first execution took `214.668 s`.

## Memory Interpretation

`nvidia-smi` showed about 25 GiB allocated because JAX reserved a device pool.
The JAX memory report distinguishes that reservation from live tensors: the
allocator limit was `25,559,728,128` bytes, while peak live allocation was
`1,367,850,752` bytes. The candidate therefore did not require 25 GiB of live
model state for this admission.

## Boundary

The run created no optimizer, update, checkpoint, or recovery artifact. It did
not regenerate Teacher data, use v5, evaluate validation values, read sealed
test data, reconstruct half-hour forcing, or scan subdaily state.

The next gate is a separate post-admission training contract. Before optimizer
creation, it must freeze the Teacher-day-start guard coefficient from the
declared eight train/train calibration batches as the median free-gradient norm
divided by the median guard-gradient norm, clipped to `[0.1, 10]`.
