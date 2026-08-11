# Successor 7/30-day Formal Cache Timeout

Status: **infrastructure failure; no model classification; shared-filesystem
persistent cache removed locally; free formal-shape validation required before
any replacement paid submission**.

Date: 2026-08-11.

## Outcome

Approved `gnall` job `14520704` ran for its complete `00:45:00` wall-time
limit on one V100 and four allocated CPUs, then Slurm terminated it for time.
The evaluator emitted no completed-artifact event and did not publish
`rollout_screen_report.json`. Therefore zero formal windows were scored and
neither arm passed or failed a scientific gate.

The job used the accepted immutable inputs:

- source commit `753db81bd39edf55f17c7cbec069859f2f2b1cda`;
- prepared-manifest SHA256 `6f4c4a30...23d0074`;
- window-plan file SHA256 `71a1a1ac...f397965`;
- fit-report SHA256 `ea4b66cd...a259a`;
- dual-checkpoint manifest SHA256 `f8eed186...75e40`.

No Teacher data, checkpoint, prepared artifact, window plan, model, objective,
or gate was changed.

## Diagnosis

The Python evaluator remained alive and held about `24.7 GiB` of V100 memory,
but repeated five-second sampling reported zero GPU compute and memory
utilization. Its main thread waited in `cl_sync_io_wait`. The configured JAX
persistent-cache directory on shared `/WORK` grew through many tiny cache and
atime files, with new entries appearing roughly once per minute while no
rollout batch completed. At 34 minutes the directory contained 137 files but
only about 4.1 MiB.

The failure is shared-filesystem persistent-cache synchronization, not model
numerics, GPU memory, the formal data panel, or the classifier. The evaluator
already has a process-local `jax.jit` cache and runs all 24 artifacts in one
process, so cross-process persistence provides no required behavior.

## Repair

`run_successor_rollout_screen.sh` no longer sets
`JAX_COMPILATION_CACHE_DIR` or creates a shared cache directory. It retains the
same JIT evaluator and its in-process executable reuse audit. A regression test
requires the launcher to remain free of `CACHE_ROOT` and
`JAX_COMPILATION_CACHE_DIR`.

Focused validation passes 13 tests, Ruff, shell syntax, and `git diff --check`.

## Next Gate

Run a bounded free-`gln01` formal-batch-shape probe with the repaired launcher.
It must complete the first four-window artifact without shared-cache writes and
show expected process-local executable reuse. Do not submit another paid job
until that evidence passes and a fresh resource/cost block receives explicit
approval.

The formal scientific screen, 365-day rollout, long-held points, Gate D2,
sealed test, and product acceptance remain open or closed exactly as before;
this timeout supplies no scientific decision.
