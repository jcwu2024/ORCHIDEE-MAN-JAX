# Successor Small-Fit Runner And Reuse Acceptance

Date: 2026-08-10.

Status: production runner and free cross-shard reuse probe accepted; exact
paid envelope frozen but no `gnall` submission authorized by this record.

## Bound Evidence

- Git commit: `6710b3517db898234dfb88e9aef493abf1bf7033`.
- Runner contract SHA256:
  `8317fc74b6ba921b0ed60e774743884ddc81b233c75ce1823f3246211d1640de`.
- Two-update schedule SHA256:
  `2c6ff41808b97b833850ecee667ddeb2e0561ad9d64dba7ce18835a7ce49fadc`.
- Report:
  `runtime/outputs/training/successor-small-fit-2u-6710b35/runner_report.json`.
- Report SHA256:
  `bcb47740c9400d01391515c8fb4c56dfa305d8f0e7db78077b882bd53ec7ed54`.
- Backend/device: JAX GPU on `cuda:0`, one Tesla V100-SXM2 32 GB on free
  `gln01`.
- Local verification: 55 successor, retained-tail, parameter-routing and
  long-horizon tests passed; Ruff, formatting, `py_compile`, Bash syntax and
  `git diff --check` passed.

The run used only accepted v6 train/train shards and existing statistics.
Teacher data were not regenerated, historical v5 was not read, and no model
checkpoint was written.

## Runner Boundary

`scripts/hpc/run_successor_small_fit.py` executes both frozen arms in one
process and feeds each arm the same deterministic annual-shard batch before
moving to the next schedule entry. The 16-row one-day anchor, two real 30-day
windows, shared candidate-derived coefficients, AdamW schedule, maximum 500
updates and fail-closed hard gates remain contract-bound.

The retained-tail scientific owner remains bound through
`bind_daily_operator`. Landpoint/year arrays are explicit dynamic JIT inputs;
only source controls that can change the retained-owner executable enter the
trace signature. Every scheduled shard is Teacher-replayed before either arm
updates. A changed landpoint/year is admitted to an existing executable only
when replay preserves the complete state, masks, undefined representation,
discrete state and budgets.

## Free Reuse Result

The schedule switched from landpoint `141.0-073.0`, year 1988, to landpoint
`099.0-111.0`, year 1999. Both entries had the same retained trace signature.
Both arms completed both AdamW updates, all gradients were finite, and all
rollout hard-failure counts were zero.

| Stage | Cold/first entry | Switched-point hot entry |
| --- | ---: | ---: |
| Batch preparation | 38.28 s | 37.39 s |
| Retained runtime preparation | 32.65 s | 49.17 s |
| Teacher replay | 31.72 s | 1.32 s |
| Attention update | 582.52 s | 1.22 s |
| Graph-control update | 604.79 s | 0.90 s |

For replay, attention and control, the hot entry started and ended with JIT
cache size one and added zero executables. This proves same-process reuse
across both landpoint and year for the frozen PFT14 contract. The hot marginal
cost was 90.01 seconds, of which 86.57 seconds was host batch/runtime
preparation. End-to-end two-update time was 2,113.55 seconds (35.23 minutes),
including the 426.78-second coefficient preflight and 220.01 seconds of final
evaluation. External `nvidia-smi` observation peaked near 25.44 GiB and the GPU
returned to zero allocation at exit.

The first-batch losses dropped strongly after two updates, but this bounded
probe intentionally did not evaluate promotion gates. It is evidence of
execution and compile reuse, not evidence that attention is better than the
matched control or that either arm passes the 500-update fit.

## Paid Envelope

Using the observed 90.01-second hot marginal gives a direct 500-update
projection of 13.04 hours. Replacing it with the previously accepted
122.41-second switched-landpoint Teacher benchmark gives a conservative 17.52
hours. The frozen request therefore uses one `gnall` node, four allocated
CPUs, one V100 and a 20-hour wall-time limit. The process is single-GPU and
does not justify more GPUs.

Worst-case additive charge at the requested limit is:

- CPU: `1 node * 4 CPUs * 20 h * CNY 0.07 = CNY 5.60`;
- GPU: `1 node * 1 GPU * 20 h * CNY 2.20 = CNY 44.00`;
- total: `CNY 49.60`.

The exact script is `scripts/hpc/slurm_successor_small_fit.sh`. From the
accepted worktree, the proposed command is:

```bash
SUCCESSOR_SMALL_FIT_EXPECTED_GIT_HEAD=<explicitly-approved-head> \
  sbatch --export=ALL,SUCCESSOR_SMALL_FIT_EXPECTED_GIT_HEAD \
  scripts/hpc/slurm_successor_small_fit.sh
```

This acceptance does not authorize that command. Submission requires a new
explicit user approval of the partition, resources, wall time, working
directory, command and worst-case cost. The 7/30-day screen, 365-day rollout,
long-held points, Gate D2 and product acceptance remain closed.
