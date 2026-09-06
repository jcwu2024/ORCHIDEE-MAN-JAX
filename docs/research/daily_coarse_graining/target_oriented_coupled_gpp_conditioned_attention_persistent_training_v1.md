# Coupled GPP Conditioned Attention Persistent Training V1

Date: 2026-09-06

Status: frozen before any persistent conditioned-Attention update or monitor
value is evaluated

Machine contract:
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.json`

Canonical contract SHA256:
`ac65c5be2b4202edad81ad8ad371e47ec8cf39443c4e4eedd2e80ab02e5c9aff`

## Question

The discarded update-3 preflight found a well-conditioned common direction,
passed all train-only finite-step gates, and improved every fixed-monitor
artifact. It did not retain the proposal and did not test whether the same
mechanism remains feasible after changing its own parameters.

This smoke asks whether the Q/K-conditioned Attention candidate can retain
three consecutive artifact-aware updates while every update improves both its
train-only panel and the fixed monitor. It is bounded persistent training, not
model promotion or a long production run.

## Exact Start And Data

The run starts from the same identity-exact parent recovery after accepted
logical update 2 that was used by the discarded preflight. The rejected parent
update 3 and the discarded conditioned update 3 are both absent. The 16
conditioned temperatures are attached at `sqrt(48)` and the resulting
15,960,211-coordinate trainable tree is serialized to immutable `start.pkl`
before the first gradient.

Logical updates 3, 4, and 5 use these schedule groups:

| Logical update | Scheduled group | Four guard groups |
| ---: | ---: | --- |
| 3 | 3 | 15, 16, 17, 18 |
| 4 | 4 | 19, 20, 21, 22 |
| 5 | 5 | 23, 24, 25, 26 |

The 120 train-only artifacts are unique and disjoint from fixed monitor
updates `251, 358, 465, 572, 679, 786, 893, 1000`. All inputs come from the
accepted Teacher-v6 `train/train` release. No Teacher data are regenerated,
and no v5, fresh-validation, or sealed-test value is read.

## Per-Update Rule

Every logical update begins by evaluating its scheduled 7/30-day science
baseline. It then recomputes 43 unit gradients:

- 40 individual direct-GPP artifact gradients;
- one scheduled reachable-state gradient;
- one scheduled 7-day science gradient; and
- one scheduled 30-day science gradient.

The minimum-norm simplex direction must have cosine at least `1e-4` with all
43 constraints. Neither a prior direction nor simplex weights are reused.
AdamW, momentum, and optimizer state are absent.

The full parameter delta is recalibrated at every update from the current
losses, gradient norms, direction cosines, and trainable-tree norm. It uses
the same `2%` maximum linearized-response target and `0.01` absolute and
`1e-5` relative L2 caps as the accepted discarded preflight. The old parent
step and the preceding conditioned step are not inherited. Descending
multipliers are `1` through `1/64`.

A scale is train-only admissible only when all 40 direct-GPP artifacts,
scheduled state, 7/30-day science macro and protected keys, and every physical
gate pass. The first passing scale is selected without reading monitor values.

## Monitor And Retention

After train-only selection, the candidate is evaluated on the fixed monitor.
Two gates are required:

1. step-local transfer against the preceding accepted monitor state; and
2. cumulative transfer against the immutable initial monitor state.

Each gate requires strict direct-GPP mean and nine-key science-macro
improvement, at least `6/8` nonregressing direct-GPP artifacts, the frozen
critical/supporting per-key caps, and zero physical hard failures. A candidate
is written to recovery only if both gates pass. A monitor failure is recorded
and the candidate parameters are discarded, leaving the preceding recovery
unchanged.

The absolute 7/30-day RMSE and bias gate is reported after every selected
train-only step and monitor. It is deliberately not weakened and is not
required to retain one smoke update. It remains mandatory before any model
promotion.

## Persistence And Output

`recovery.pkl` is written atomically after the initial monitor and after each
monitor-admitted update. It contains only the conditioned trainable tree,
identity, accepted update records, and monitor records. Fixed parameters and
optimizer state are not serialized. An interrupted run resumes from the exact
latest recovery without replaying an approximate parameter trajectory.

Only three passing updates can create
`checkpoint-update-0005.pkl`. The checkpoint is diagnostic and not promotion
eligible. The runner also writes compact `progress.json`; a rejected candidate
is written separately and never becomes recovery state.

## Execution Boundary

One resumable free-`gln01` run is allowed with a 240-minute process timeout.
The discarded preflight measured an approximately 12 GB JAX allocator peak;
one V100 is sufficient. Most startup time is compilation, while subsequent
same-shape updates reuse compiled executables.

The final inference boundary remains one true daily process head bound through
the conservative updater and retained exact tail. It reconstructs no 48
half-hour forcing records and performs no 48-step state scan.

Paid execution, fresh validation, 365-day rollout, Gate D2, sealed test, and
product acceptance remain closed regardless of this smoke's outcome.

## Implementation

- Contract and calibration:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_conditioned_attention_persistent_training.py`
- Identity-exact persistence:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_conditioned_attention_persistence.py`
- Runner:
  `scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.py`
- Free-node wrapper:
  `scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh`

Local qualification is complete: 57 related tests plus Ruff, Python
compilation, shell syntax, manifest/result loading, CLI wiring, and whitespace
checks pass. The active gate is one free run from a clean synchronized commit.
See
`target_oriented_coupled_gpp_conditioned_attention_persistent_training_local_acceptance_20260906.md`.
