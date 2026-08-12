# Successor Staged-Training GPU Smoke Acceptance

Status: **accepted free-GPU execution preflight; not architecture promotion evidence**.

## Decision

The staged-v3 matched training path is executable on one V100. The smoke
accepts the mechanics required before a larger fit: both arms receive finite
gradients, both parameter updates are applied, and the training and milestone
validation rollouts preserve all declared state and budget hard gates.

This result does not rank or promote either arm. Two updates cannot establish
biomass, productivity, or long-rollout accuracy. The pair remains:

- `source_routed_process_attention_v2` candidate;
- `source_routed_graph_mlp_control_v2` matched control;
- 8,722,689 parameters per arm at the canonical `bind_daily_operator(...)`
  boundary.

## Execution Identity

- consumer commit: `9b4a1cee6dca5c156eb26848232f9212e8fa9d52`;
- prepared-artifact producer commit:
  `a2c2638f06bc21cd0e798b05c1ad3b8e9532417d`;
- staged contract SHA256:
  `53d21d3a03a1502263b333aaba9719b2d42f096672844497b1a213dd54c23c7b`;
- prepared training manifest SHA256:
  `6ee7d949ba045ff35b9881d294a42ccbc5d6f2516f92df50d2deee3b63f7d8fc`;
- prepared validation manifest SHA256:
  `6897757e9c548819721dc7caea300dc0eaa9058318b66eea2d8b76f13d7b50db`;
- report path:
  `runtime/outputs/training/successor-staged-smoke-9b4a1ce/probe_report.json`;
- report SHA256:
  `67b4ba0748e4f2fdc918de68587fa5ea4834271152940aba011c60fd06839f4c`.

The launcher repair separates the current consumer commit from the immutable
prepared-artifact producer commit. It also recognizes the observed physical
hostname behind the `gln01` alias and prints explicit preflight failures.

## Numerical Evidence

Each arm completed two AdamW updates. All four losses and gradient norms were
finite, every update was applied, and every nonfinite-gradient count was zero.
Losses changed from `1.9038566906` to `0.4133849572` for attention and from
`1.8904181606` to `0.3645876150` for control. These two heterogeneous batches
are a numerical smoke only, not a learning-curve comparison.

One fixed prepared validation artifact from each temporal, spatial, and joint
non-test slice was evaluated for both arms. Across training and validation:

- nonfinite prediction: zero;
- defined-mask mismatch: zero;
- discrete-state mismatch: zero;
- negative declared stock: zero;
- budget residual failure: zero.

An independent `test01` process re-read the report and checked four training
arm records, six validation arm records, all finite/update conditions, and all
hard-gate counts.

## Runtime Evidence

The free `gln01` run completed in `26:35.37` wall time with exit status zero.
Maximum host RSS was `11,648,856` KiB. Observed GPU allocation peaked near
`21.9` GiB on a 32-GiB V100. XLA reported one `jit_train_step` compilation of
`5:57.59`; compilation dominates this two-update smoke, while the prepared
inputs avoid per-update Teacher replay.

The smoke wrote no checkpoint or recovery state. It did not regenerate
Teacher data, read historical v5, access sealed test data, reconstruct 48
forcing records, or execute a 48-step candidate state scan.

## Next Gate

Use these measurements to freeze a resource and cost proposal for the first
`1-500` preparation/training stage. Do not prepare that stage or submit paid
work until the user explicitly approves its resources, wall time, working
directory, command, and worst-case cost. The 365-day rollout, long-held-point
validation, Gate D2, sealed test, and product gates remain closed.
