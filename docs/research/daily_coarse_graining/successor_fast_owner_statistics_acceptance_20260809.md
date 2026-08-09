# Successor Fast-Owner Statistics Acceptance (2026-08-09)

Status: **accepted; the no-checkpoint GPU probe is now the active gate**.

## Bound Inputs

The accepted scan used clean research commit
`c9f35c0a38b0eec4c1b2b11ff4bdd6b7e98d0055` in
`runtime/worktrees/fast-owner-statistics-c9f35c0`. It bound the frozen v6
Teacher manifest, v6 acceptance report, base statistics, accepted 101-field
process statistics, successor candidate contract, and 160-leaf fast-owner
contract by their recorded identities.

The scan read only existing accepted-v6 `state_trajectory` and
`fast_day_target` arrays. It did not regenerate Teacher data, read historical
v5 rows, train either neural arm, write a checkpoint, reconstruct 48
half-hour forcing records, or execute a 48-step candidate state scan.

## Execution

A one-shard correctness preflight ran on free `test01` before submission. It
read 364 train/train rows, wrote and reloaded all 79 added value statistics,
and recorded 364 `rveget`-defined values over 728 eligible compact-PFT slots.

The explicitly approved production allocation was Slurm job `14515519`:

- partition `cnmix`, one node, one task, eight CPU cores, zero GPUs;
- wall-time limit `02:00:00`;
- worst-case charge CNY `1.12`;
- completed with exit code `0:0` in 8 minutes 57 seconds;
- all eight one-core worker steps completed in about 8 minutes 2 seconds.

The approximate allocation charge from elapsed wall time is CNY `0.08`.

## Accepted Artifact

The eight partitions merged over exactly 8,591,565 train/train rows and
23,540 source shards. A fresh Python process on `test01` independently loaded
the merged artifact and verified:

- exactly 79 added value-output fields;
- zero never-defined output columns;
- finite mean, M2, variance, and scale arrays;
- complete 181-field output normalization;
- complete 180-field value-loss scales;
- 4,771,338 `rveget`-defined values over 17,183,130 eligible slots, a defined
  fraction of `0.2776757203140522`.

The accepted files are:

- `runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json`,
  SHA256 `b2e61a171156071d404d12b833130f2d0e98064564d1c3e007e16f0210734719`;
- `runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.npz`,
  SHA256 `890d1fd6110912a835c760b4fe58e9320481a929797858cad5b70b1bf03eb99f`.

## Gate Decision

The added-head statistics gate is accepted. The next bounded operation is a
free-`gln01`, no-checkpoint forward/loss/gradient probe for both matched arms
using this artifact. No bounded fit is admissible until that probe passes.
The 7/30-day rollout, 365-day validation, long-held-point validation, Gate D2,
and product acceptance remain closed.
