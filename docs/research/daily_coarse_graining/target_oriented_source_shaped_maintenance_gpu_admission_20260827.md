# Target-Oriented Source-Shaped Maintenance GPU Admission

Date: 2026-08-27

Status: **accepted zero-update real-v6 GPU admission**

## Decision

The `target_oriented_source_shaped_maintenance_v1` candidate is admitted to a
separately designed train/train screen. This result proves that the complete
84,148,646-parameter candidate can load the exact update-1000 diagnostic
parent, execute and differentiate on accepted Teacher-v6 data, and pass one
canonical daily transition. It is not evidence of 7/30-day scientific
improvement and does not promote a checkpoint.

The run used one free `gln01` V100 32 GB, GPU 0, one process, and zero optimizer
updates. It did not regenerate Teacher data, use historical v5 or validation
values, read sealed-test rows, reconstruct half-hour forcing, scan subdaily
states, or write a checkpoint.

## Identity

- Repository commit: `cf1feadb46165af73b0d87f59877dc034654c3b0`.
- Dataset manifest SHA256:
  `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`.
- Parent update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`.
- Source-shaped contract SHA256:
  `3c4d0c085c767f14b001837ccefb76aca291a56b8fd9d3a5e33279602af57fc4`.
- Admission report SHA256:
  `b5684fb49ba830fba7fd2e4151fe76054f94c349b737c3d49101fbf4b5119b46`.

The canonical report is
`runtime/outputs/training/target-oriented-source-shaped-maintenance-gpu-admission-cf1fead/report.json`.
The workstation evidence copy is
`outputs/diagnostics/target-oriented-source-shaped-maintenance-gpu-admission-cf1fead-report.json`.

## Result

- Backend and device: `gpu`, `cuda:0`.
- All 6,674 declared-defined values in 181 output fields were finite in eager
  and JIT execution.
- The 160 non-maintenance fields were exactly equal to the parent output.
- Eager forward took `39.23 s`; cached JIT compile plus first forward took
  `24.52 s`; hot replay took `0.06096 s`.
- The complete loss and full-gradient pass took `66.73 s` with the persistent
  compilation cache.
- All 84,148,646 gradient elements were finite. Every retained Attention
  parameter family had finite nonzero gradients.
- The new aboveground and belowground branches both had finite nonzero
  gradients, with L2 norms `0.0664654` and `0.000623640` respectively.
- The bound canonical transition took `28.34 s`, produced 5,737 finite
  declared-defined next-state values, and had exactly zero water and carbon
  budget residuals.
- JAX peak live allocation was 2,732,977,152 bytes, about 2.55 GiB. The roughly
  25 GiB shown by `nvidia-smi` was allocator reservation, not live tensors.

The admission proxy loss was `1149.4188`. It is a one-day zero-update
differentiability probe, not the train/train objective and not a promotion
metric.

## Admission Repair

The first execution at commit `b628634` completed every structural, gradient,
state, and budget check but returned failed because eager/JIT float32 reduction
ordering produced a `4.7251e-5` difference in `gsmean`. That was 1.098 times the
old generic tolerance while the candidate's non-maintenance eager outputs were
bitwise equal to the parent.

Commit `cf1fead` parameterizes the shared comparison helper without changing
its existing defaults and applies `atol=1e-5`, `rtol=5e-6` only to this loaded-
checkpoint admission. The repeated difference is 0.510 times that tolerance.
No model, data, loss, parameter, or optimizer behavior changed.

## Next Gate

Freeze and locally admit a source-shaped train/train screen that starts from
the exact update-1000 parent parameters, discards the old maintenance head and
all parent optimizer state, initializes a fresh optimizer, and reuses accepted
Teacher-v6 train assets. The screen must evaluate the frozen 7/30-day science
panel before any validation or sealed-test access.

Scientific promotion still requires at least 20% median maintenance-score
improvement at both horizons, nonpositive median changes for GPP, growth and
total autotrophic respiration, NPP, AGB, BGB, total live biomass, and ecosystem
carbon, no group regression above 1%, and zero hard/nonfinite counts. Paid
training, 365-day rollout, Gate D2, validation, sealed test, and product
acceptance remain closed pending a separate executable contract and resource
approval.
