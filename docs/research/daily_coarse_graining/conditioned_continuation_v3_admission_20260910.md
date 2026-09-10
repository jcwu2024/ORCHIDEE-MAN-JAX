# Continuation v3 preparation and admission

Status: all admission checks completed; ready for paid-submission approval.
Execution code: `ce7cd7985e28fb5b53ae2d578d11f6b1d44761db` on the workstation
and server at admission launch. Later offline-report/document changes do not
alter this running execution. Plan: `conditioned_continuation_v3_plan_20260910.md`.

## Completed preparation

- Local 29-test set passes: corrected-continuation v1/v2/v3 lifecycle, pending
  recovery, completed resume, continued sampling, parent-34 loading and original
  monitor preservation, and existing validation selection/transfer behavior.
- Ruff and whitespace checks pass on the execution changes.
- Git bundle transfer fast-forwards the clean server checkout from `3d94543`
  to the execution code; no environment installation or dataset regeneration.
- Free test02 shell syntax passes. The server CPU environment lacks pytest;
  no dependency was installed. A direct CPU check instead loads the **actual**
  completed update-34 recovery via the new starting-point path and confirms
  start 34, end 74, 40 panels and direct monitor loss 0.8566602935651434.
- Sampling uses 1,600 exposures and 678 distinct existing packages, excludes
  monitor IDs, and exactly continues the preceding 20 blocks of seed 20260910.

## Free GPU admission

Node gln01, GPU 1 (V100 32GB) was idle at launch. GPU 0 had another process
using about 8.5 GiB. The test uses preallocation fraction 0.55, a 30m inner
timeout, PREFLIGHT_ONLY=1 and the v3 manifest. No updates are trained.

The first shell invocation supplied the abbreviated execution commit and
exited before Python/GPU computation because the existing wrapper requires
the full value. Corrected the launch argument; no model or wrapper change.
The actual free test loaded its data and reproduced all eight fixed monitor
packages at 520.1s, then began the existing 24-window validation with trajectory
capture. All 24 windows completed in **1236.23 seconds**, with zero trained
updates and zero physical hard failures. Both horizon evaluators compiled;
the validation timing includes cold compilation (maximum window 386.14s),
so its 713.64s total is not a hot inference throughput benchmark. No OOM
occurred; the GPU returned to 0 MiB after completion.

Local comparison to job 14984032's saved update-34 result passes **252 scalar
metric/count comparisons** (all acceptance cells and direct-GPP cells), with
identical hard counts. All **624 complete trajectories** (24 windows times
13 quantities times Teacher/model) match the earlier independent exporter.
The embedded-curve analysis CLI also reproduces the earlier offline diagnosis
exactly, using real before/after curves. It now produces the proposed physical
progress review alongside the errors; that review never selects checkpoints.
Local evidence: `outputs/conditioned-v3-preflight-ce7cd79.json`.

Remote log: `runtime/logs/conditioned-v3-preflight-ce7cd79.log`.
Remote report: `runtime/outputs/diagnostics/conditioned-v3-preflight-ce7cd79/report.json`.

Ready command on gln01, in `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`:

```bash
EXPECTED_GIT_HEAD=ce7cd7985e28fb5b53ae2d578d11f6b1d44761db sbatch scripts/hpc/slurm_conditioned_snow_free_continuation_v3.sh
```

Proposed gnall envelope: one V100, 8 Slurm CPUs, 32GB host memory, 12h ceiling,
additive maximum CNY 33.12. The server remains at this tested execution commit.
No paid submission has been made. Do not submit from this admission alone.
