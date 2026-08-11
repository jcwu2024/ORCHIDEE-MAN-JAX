# Successor 7/30-day Formal Screen Result

Status: **attention candidate rejected; matched control not promoted; stop and
attribute the bounded failure**.

Date: 2026-08-12.

## Decision

The frozen 96-window formal screen completed on the free `gln01` V100 after
the shared-filesystem persistent-cache repair. The attention candidate fails
all six primary-composite improvement gates, 46 of 72 per-key RMSE gates, and
42 of 72 signed-bias gates. All 30 hard gates pass.

The declared decision is
`stop_and_attribute_bounded_7_30_day_failure`. Do not run the 365-day rollout,
long-held points, Gate D2, sealed test, or product acceptance for this
candidate. The graph-MLP control is a matched benchmark, not an independently
accepted surrogate, and is not promoted by the candidate's rejection.

## Evidence Identity

- evaluation commit:
  `3430ec4a6605c7d6518b6dfd7fdb16d9a1e9c77f`;
- report SHA256:
  `7e8dc04f025c1be19c17f0414b6fefec9b8dfc96ed4cfd8fc1dcee43a5b9d52b`;
- prepared-manifest SHA256:
  `16a785a1c7559fef8f262a9453b709cb6949394cff54ca0d07fd16e3ee956ede`;
- window-plan file SHA256:
  `71a1a1ac7747536b4f0ba9f750b163a59aac86bf7f2f5c3d9002b23f9f397965`;
- internal window-plan SHA256:
  `2476309696b3ba24f62b435f7c08e47192c58279242e754c12b0cd4654f722b6`;
- fit-report SHA256:
  `ea4b66cd2d09d20f5ac4754de461382d560e4feedca5d52e211bebe77a4a259a`;
- dual-checkpoint manifest SHA256:
  `f8eed186a90c2f5ecad39c6f9114b343fb29ab468266fed39dfe70e353275e40`.

The report is archived locally at
`outputs/diagnostics/successor-7-30-formal-free-probe-3430ec4/rollout_screen_report.json`
and remains on Explore1000 under
`runtime/outputs/training/successor-7-30-formal-free-probe-3430ec4/`.

The run used all 96 predeclared non-test windows, did not regenerate Teacher
data or read historical v5, and left the sealed test split closed.

## Primary Result

Negative relative change would favor attention. All observed changes favor
the control instead.

| Slice | Day | Attention composite | Control composite | Relative change |
| --- | ---: | ---: | ---: | ---: |
| temporal | 7 | 3.595401 | 0.965847 | +272.25% |
| temporal | 30 | 2.040070 | 0.965863 | +111.22% |
| spatial | 7 | 1.243290 | 0.453758 | +174.00% |
| spatial | 30 | 0.982273 | 0.419680 | +134.05% |
| joint | 7 | 2.214495 | 1.152646 | +92.12% |
| joint | 30 | 1.838784 | 1.144891 | +60.61% |

The classification contains 78 relative gates: all six composite gates and
46 of 72 key gates fail. Temporal Day 7/30 fail 11/10 relative gates,
spatial Day 7/30 fail 8/9, and joint Day 7/30 fail 7/7. Seven candidate keys
in every slice/horizon cell exceed the absolute normalized signed-bias limit,
for 42 of 72 bias-gate failures.

## Scientific Attribution Signal

GPP alone is not the dominant failure. Attention improves GPP RMSE at temporal
Day 30 and both spatial and joint horizons. It nevertheless damages the
carbon-use balance:

- NPP RMSE is `3.46-6.92x` control in temporal, `4.38-4.90x` in spatial,
  and `3.53-3.94x` in joint Day 7/30;
- maintenance-respiration RMSE is `2.88-6.96x`, `2.57-4.13x`, and
  `2.48-4.47x` control over the same cells;
- total-live-biomass RMSE is `3.41-6.90x`, `4.38-4.85x`, and `3.50-3.94x`
  control;
- AGB and BGB show the same repeated multi-fold degradation;
- ecosystem-carbon RMSE is worse in every cell, from `1.06x` to `2.15x`
  control.

This pattern is consistent with a learned GPP/maintenance/NPP allocation
imbalance that accumulates into live-biomass drift. It is an attribution
hypothesis, not yet a causal proof.

## Numerical Integrity And Runtime

Both arms have zero nonfinite predictions, negative declared stocks,
defined-mask mismatches, discrete-state mismatches, and budget-residual
failures on all three slices. The candidate is therefore structurally stable
and conservative over 30 days but scientifically inaccurate relative to the
matched control.

Removing the shared persistent cache resolved job `14520704`'s infrastructure
timeout. The complete formal run took `312.730` seconds. Attention compiled
and evaluated its first artifact in `81.538` seconds and control in `53.573`
seconds. Every later artifact added zero executables and took about
`0.25-0.31` seconds for attention and `0.18-0.19` seconds for control.

## Next Gate

Perform one bounded, no-training attribution on the existing formal windows:
separate GPP, maintenance respiration, NPP, and live-biomass errors by named
process, day, slice, and source-routed edge. The purpose is to determine
whether the failure originates in attention weighting, the shared decoder,
or the long-horizon objective balance before discussing any successor
architecture.

Do not tune against the formal windows, alter the frozen gates, train another
candidate, or promote the control as part of that attribution.
