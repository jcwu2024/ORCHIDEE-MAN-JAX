# Successor Small-Fit 500-Batch Preparation Acceptance

Date: 2026-08-10.

Status: the complete 500-entry CPU-prepared training input is accepted; the
bound GPU fit remains unsubmitted pending separate approval.

## Bound Evidence

- Slurm job: `14517346`.
- Execution commit: `d4936996635b02fcf04cd5f8c4926be3cc010124`.
- Partition/node: `cnmix` / `ibc11b11n22`.
- Allocation: one node, four tasks, one CPU per task, no GPU.
- Requested limit: 4 hours; approved worst-case charge: CNY 1.12.
- Slurm result: `COMPLETED`, exit `0:0`, elapsed `01:58:09`.
- Output directory:
  `runtime/outputs/training/successor-small-fit-prepared-500u-d493699`.
- Prepared manifest SHA256:
  `0182b7f7a6d5eae09025a336f26bc67ba5fb54706c3529a846d3885c16d80462`.
- Frozen 500-update schedule SHA256:
  `f6c0d36f500f1dc47699cd8783c0bbf2379a9d0918751d9f1b99280e6a74103b`.
- Shared retained trace signature:
  `13ed0a2eb5895d9b9c8325a1e122a8e07d51a615fb842676b1ac17cf0cc51d8b`.

No Teacher data were regenerated, no historical v5 data were read, no sealed
test data were read, no model training occurred and no GPU was allocated.

## Worker Result

All four workers completed their exact 125-entry round-robin partitions and
exited zero. Their elapsed times were `01:52:36`, `01:54:33`, `01:54:08` and
`01:53:31`. Maximum RSS ranged from 5,165,680 to 5,171,244 KiB; the largest
was approximately 4.93 GiB. No worker swapped.

The finalizer completed with exit zero. Its internal full artifact admission
took 106.73 seconds, while its complete fresh process including runtime load
took `00:03:28`. It emitted exactly one final manifest only after every worker
partial passed.

The elapsed-runtime cost estimate is approximately
`4 CPUs x 1.969 h x CNY 0.07 = CNY 0.55`; this is not a billing statement.
The approved worst-case ceiling remains CNY 1.12.

## Independent Admission

A separate read-only `test01` process re-read the final manifest and all
artifact bytes. It required:

- manifest status `passed`, worker count 4 and execution HEAD `d493699`;
- exactly 500 entries ordered `1..500` with 500 unique paths;
- every file to exist under the admitted output directory;
- every recorded byte size and SHA256 to match independently;
- the physical byte sum to equal the manifest total;
- Teacher regeneration and historical v5 flags to remain false.

All checks passed. The audit independently hashed 16,717,190,876 bytes in
95.40 seconds. The manifest records the same byte total, approximately
16.72 GB decimal or 15.57 GiB.

## Immutable GPU Input

The accepted manifest binds execution HEAD `d493699`. The current server
execution worktree must remain clean and pinned to this commit until the GPU
fit consumes the manifest. Advancing that worktree would correctly fail
prepared-manifest admission and would require regenerating all 500 artifacts.
Documentation commits may advance locally but must not be merged into this
execution worktree before the bound fit.

The next gate is a separate explicit approval for one same-process 500-update
fit using only this manifest. Passing that fit opens, but does not itself pass,
the separately authorized 7/30-day screen. The 365-day rollout, long-held
points, Gate D2 and product acceptance remain closed.
