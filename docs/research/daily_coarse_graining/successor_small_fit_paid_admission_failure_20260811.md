# Successor Small-fit Paid Admission Failure

Status: **job rejected before training; coefficient schedule-ownership bug
repaired and exact production assets admitted on free CPU; replacement
submission not authorized**.

Date: 2026-08-11.

## Result

Approved Slurm job `14517771` started on `gnall` at
`2026-08-11T02:48:04` and ended after `00:02:52` with state `FAILED` and exit
code `1:0`. It ran execution commit
`c15cd86f3f3de8ceb20781abc603187cc6b272cc`, allocated 16 CPUs and one V100,
and failed while admitting the frozen objective coefficients. The training
loop executed zero updates. The intended output directory is empty and no
checkpoint staging or final checkpoint directory was published.

At the observed elapsed time, a simple elapsed-time estimate is approximately
CPU CNY 0.05 plus GPU CNY 0.11, totaling CNY 0.16 if CPU and GPU charges are
additive and billed by actual allocation time. This is not a center invoice.

## Root Cause

The frozen coefficients were computed once from the first train/train batch
of an accepted two-update prefix probe. Its source schedule SHA256 is
`2c6ff41808b97b833850ecee667ddeb2e0561ad9d64dba7ce18835a7ce49fadc`.
The complete 500-update prepared product has schedule SHA256
`f6c0d36f500f1dc47699cd8783c0bbf2379a9d0918751d9f1b99280e6a74103b`.

The first prepared artifact is byte-identical in both lifecycles, SHA256
`d0b0fdaee6275546f5d740c670846a6d8e0849b0fa0c1d25ea36afe2987349ee`.
The coefficients depend only on this first artifact. The full production
schedule is separately hash-bound by the prepared manifest and checkpoint
handoff.

The loader nevertheless compared the source probe's complete schedule hash
against the production fit's complete schedule hash. This conflated source
coefficient evidence with consuming fit ownership and necessarily failed even
though the coefficient-defining batch was exact.

## Repair

`successor_frozen_coefficients.py` now:

- binds the source report to the source schedule recorded in the frozen
  coefficient contract;
- continues to require the exact first-artifact SHA256, runner, candidate,
  owner, coefficient values, target ratios and source-report SHA256;
- leaves production schedule admission to the existing prepared-manifest and
  checkpoint-handoff boundaries.

No coefficient was recomputed and no training objective, optimizer, model,
prepared artifact, data split or acceptance threshold changed.

A new CPU-only production-asset probe admits the exact 500-update schedule,
prepared manifest, handoff, first artifact, frozen coefficient contract and
source report in one process. It records zero training updates and writes no
checkpoint. Local focused regression passes `37/37`, including the distinct
source-versus-production schedule case. Ruff, Python compilation, shell
syntax and `git diff --check` pass.

## Next Gate

Before requesting any replacement paid run:

1. commit the repair and current 7/30-day package;
2. transfer it to a clean Explore1000 worktree;
3. run `probe_successor_production_admission.sh` on `test01`-`test04` against
   the exact production assets;
4. require a passed report and record its SHA256;
5. present a fresh paid-resource approval block.

These steps are complete through item 4. The accepted report SHA256 is
`50d090e6382e13d3f735fd2d9a92c2feb0fbc9f0c6fa2acb55ecd6d2418ea089`;
see `successor_production_fit_asset_admission_20260811.md`.

No replacement submission, free GPU screen, 7/30-day numeric evaluation or
later gate is authorized by this repair alone.
