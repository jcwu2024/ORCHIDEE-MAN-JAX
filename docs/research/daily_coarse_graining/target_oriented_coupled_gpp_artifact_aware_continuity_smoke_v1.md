# Coupled GPP Artifact-Aware Continuity Smoke V1

## Question

The preceding zero-training audit produced the first direct-GPP correction
that improved every one of 40 train-only artifacts and every one of eight
fixed-monitor artifacts while preserving scheduled one-day state and 7/30-day
science. That is one accepted parameter displacement, not evidence that the
same construction remains feasible after changing its own parameter state.

This smoke asks whether the result survives three consecutive, independently
recomputed steps. It does not change the Attention architecture, regenerate
Teacher data, introduce an optimizer, or compare against a weak control.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.json`,
canonical SHA256
`451a1003514c465b1e191f741e07cbe18285bf117eee6ae71020706d577e20f3`.

## Train-Only Sequence

The runner reconstructs the same logical pre-update-2 parameter state once.
It then attempts logical updates 2, 3, and 4 in order. The five schedule groups
used for each update are frozen as follows:

| Logical update | Scheduled group | Four additional train-only groups |
| ---: | ---: | --- |
| 2 | 2 | 11, 12, 13, 14 |
| 3 | 3 | 15, 16, 17, 18 |
| 4 | 4 | 19, 20, 21, 22 |

Each group contains eight point-year artifacts and each artifact contains 208
accepted Teacher days. A direction therefore exposes 40 artifact gradients,
representing 8,320 day records. The three panels contain 120 distinct
artifacts and are disjoint from the fixed eight-artifact monitor.

At every logical update, the runner recomputes all 43 unit gradients: the 40
individual direct-GPP gradients plus scheduled reachable-state, 7-day
science, and 30-day science gradients. It solves the same `43 x 43`
minimum-norm simplex and does not reuse the preceding step's weights.

## Backtracking And Stop Rule

The full direct displacement norm remains
`0.007758973351050004`. The descending scale ladder remains
`1/2, 1/4, 1/8, 1/16, 1/32, 1/64`. The first scale passes only when:

1. all four scheduled source means pass the existing finite-step rule;
2. all scheduled one-day and rollout physical gates pass;
3. all 40 individual direct-GPP losses are nonincreasing within relative
   tolerance `1e-6`; and
4. the combined 40-artifact direct-GPP mean strictly decreases.

The largest passing scale is applied only in memory. If no common direction
or no passing scale exists, the smoke stops immediately and does not read the
monitor. No later panel is evaluated from a state that failed its predecessor.

## Monitor Isolation

Only after all three train-only updates have been selected does the runner
evaluate the fixed monitor at the initial state and the three post-update
states. Monitor success requires its mean direct-GPP loss to remain strictly
below the initial value after each update. Individual monitor artifact changes
are reported for diagnosis but are not an additional hidden acceptance gate.

This ordering ensures that monitor values cannot choose a direction, scale,
or stopping point. The first logical update repeats the predecessor panel and
therefore checks reconstruction repeatability, but cross-GPU bit identity is
not required; all comparisons are paired within this execution.

## Decision Boundary

Three train-only steps with a continuously improved monitor open a separately
frozen bounded persistent-training design. A train-only stop or monitor
failure moves to broader randomized sampling or a quantile/CVaR objective;
it does not trigger another adjustment of fixed group means.

The smoke retains no optimizer state, parameter proposal, recovery,
checkpoint, or trainable parameter. It uses no fresh validation, sealed test,
historical v5 data, or regenerated v6 data and performs no 48-record forcing
reconstruction or 48-step state scan. One free `gln01` execution is allowed
with a 240-minute timeout. Paid execution remains closed.

## Implementation

- Contract:
  `manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.json`
- Contract logic:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_artifact_aware_continuity.py`
- Runner:
  `scripts/hpc/audit_target_oriented_coupled_gpp_artifact_aware_continuity.py`
- Free-node launcher:
  `scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_continuity.sh`
