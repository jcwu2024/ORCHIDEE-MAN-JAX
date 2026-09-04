# Coupled GPP Artifact-Aware Continuity Result (2026-09-04)

## Outcome

The corrected free-`gln01` execution completed at commit `c345e64` in
`5,467.939` seconds. Its classification is
`artifact_aware_three_update_continuity_passes_monitor`: all three train-only
updates find a 43-constraint common direction and an admissible finite scale,
and the fixed-monitor mean direct-GPP loss remains below its initial value
after every update.

The server report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-artifact-aware-continuity-c345e64.json`,
SHA256
`aa326bb4cfb237528cb0785e7d03d7f440f91647bcab4959d0334afc0a452d69`.

## Consecutive Train-Only Updates

Each update recomputes its direction from 40 individual direct-GPP artifact
gradients and the scheduled reachable-state, 7-day, and 30-day gradients. The
three panels contain 120 distinct train-only point-years. Results are:

| Logical update | Minimum cosine | Simplex support | Selected scale | Combined 40-artifact GPP change |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `0.048521` | `19/43` | `1/32` | `-0.2860%` |
| 3 | `0.026846` | `25/43` | `1/16` | `-0.2244%` |
| 4 | `0.064102` | `16/43` | `1/2` | `-5.2176%` |

Every selected step improves all `40/40` direct-GPP artifacts. The weakest
artifact changes remain negative: `-0.0616%`, `-0.0394%`, and `-0.9911%` for
updates 2, 3, and 4. All scheduled physical gates pass. The scheduled source
changes are:

| Update | Direct GPP | Reachable state | 7-day science | 30-day science |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `-0.4180%` | `-0.0991%` | `-0.0916%` | `-0.0464%` |
| 3 | `-0.3577%` | `-0.0465%` | `-0.0649%` | `-0.0557%` |
| 4 | `-5.1037%` | `-1.8790%` | `-0.7872%` | `-0.5978%` |

## Fixed-Monitor Trajectory

The monitor is evaluated only after all three train-only choices are final.
Its mean direct-GPP loss changes from `0.06277738` to `0.06262310`,
`0.06250080`, and `0.05887112`. Relative to the initial state, those are:

| State | Mean direct-GPP change | Improved artifacts |
| --- | ---: | ---: |
| After update 2 | `-0.2458%` | `8/8` |
| After update 3 | `-0.4406%` | `7/8` |
| After update 4 | `-6.2224%` | `7/8` |

The final non-improving artifact is update 251 at `+4.5727%`; all other fixed
monitor artifacts improve. Individual monitor artifacts were report-only, as
frozen. The aggregate trajectory therefore passes without hiding a stricter
post hoc gate.

## Reproducibility Qualification

The within-run paired result is valid, but the reconstructed logical
pre-update-2 state is not numerically stable enough to serve as a reusable
training start. This run's first accepted scale is `1/32`, compared with
`1/2` in the predecessor one-step audit. Its train-only and monitor baselines
are `0.06050664` and `0.06277738`, compared with `0.05846474` and `0.06073579`
in the predecessor execution. These are material changes, not bit-level noise.

The failed-report execution before the one-line monitor-pairing correction had
also completed train-only scales `1/2`, `1/2`, and `1/8`. That process retained
no parameters, so it is operational corroboration only, not a second formal
result. Together the executions show that the artifact-aware mechanism can
repeatedly find admissible steps, but not that repeated bootstrap reconstruction
produces a stable parameter trajectory.

## Integrity And Decision

All `93,504,164` fixed parameters and the parent recovery remain unchanged.
All gradients are finite, all physical hard gates pass, and reported JAX
allocator peak use is `12,285,025,536` bytes. The run writes no parameter,
optimizer, checkpoint, or recovery state and reads no fresh validation or
sealed test data. It does not regenerate v6, use v5, reconstruct 48 forcing
records, or execute a 48-step state scan.

Accept the three-update direction-transfer mechanism, not the model for
promotion. The next training contract must materialize one exact starting
parameter snapshot, verify its round trip, and write identity-bound recovery
after every accepted update. It may then extend the same artifact-aware rule
over rotating train-only panels with fixed, post-selection monitor milestones.
365-day rollout, Gate D2, sealed test, and product acceptance remain closed.
