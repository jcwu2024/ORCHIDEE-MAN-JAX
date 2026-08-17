# Hydrology And Canopy Output v2 Real-v6 Admission

Date: 2026-08-17.

Status: accepted. The next bounded gate is one complete 364-day, one-epoch
training smoke on free `gln01`.

## Execution

The zero-update admission ran on one free `gln01` V100 against commit
`17a0e75f1fcec9e5abf190fae2325d251e99cf4b`. It read row 0, day index 2, from
accepted train/train landpoint `001.0-071.0`, year 1961.

The formal report is
`outputs/research/daily-coarse-graining/mechanistic_output_real_v6_gpu_admission_v2_20260817.json`.
Its SHA256 is
`cd376e484d4bdd6754cc1e917355193f4165a923ec229062035a30d006ccd0d8`.

## Result

- All 181 named outputs and all 6,674 Teacher-defined scalar values are finite.
- Eager and JIT forward agree within the frozen tolerance; the maximum
  tolerance ratio is `0.113519`.
- Complete loss is finite at `1323.8714967`.
- The full 49,743,001-element parameter gradient has 678 leaves, all values
  finite, L2 norm `208.5264922`, and 643 nonzero leaves.
- The canonical one-day transition emits all 181 outputs, has 5,737 finite
  defined next-state values, and has exactly zero water and carbon budget
  residual.
- No optimizer update or checkpoint was written. No Teacher data were
  regenerated; no v5, sealed-test, reconstructed 48-record forcing, or
  48-step state scan was used.

`nvidia-smi` showed roughly 25 GiB allocated during compilation, while JAX's
memory report records only 1.441 GB peak bytes in use. The larger number is
allocator reservation, not live model tensors.

## One-epoch Training Smoke Contract

This next operation is an execution/stability gate, not a model-ranking
experiment, so it has no matched control. It uses the accepted v2 candidate
from a fresh fixed seed and every one of the 364 valid train/train days exactly
once. Batch size 8 gives 46 AdamW updates, with zero-weight padding only in the
last batch. It writes a JSON report and no checkpoint.

The smoke passes only if all 46 updates complete, all losses, parameters, and
gradients are finite, the final complete process loss is below initialization,
the initial/final loss masks and counts are identical, and all prediction,
next-state, water/carbon budget, and nonnegative-inventory hard counts are
zero. The formal 99% total, 95% per-expert, and 0.05 primary NRMSE gates are not
used to classify this one-epoch execution.

A pass opens the separately evaluated 100-epoch one-point memorization gate.
It does not open 64-point-year, full-v6, rollout, paid, Gate D2, sealed-test, or
product stages.
