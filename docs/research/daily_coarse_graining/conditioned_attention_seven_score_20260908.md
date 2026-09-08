# Seven-Day Observation/Production Score Localization

Status: bounded zero-update diagnostic frozen before real-data execution.

Local qualification passes 21 focused and related tests in 45.25 seconds,
including complete two-window statistics, streamed gradients, finite
differences, JSON serialization and qualified launcher argument routing.

Integration `1e6c521` completes actual train-only descent but fails seven
seven-day scalar comparisons. Keep the same accepted update-3 recovery,
conditioned Attention, v6 data, loss and tolerances. No training update,
checkpoint, fixed-monitor evaluation or paid job is part of this diagnostic.

## Source Findings And Fixed Inputs

`StreamedHorizonCallbacks` and production `window_statistics` call the same
science sufficient-statistic and linearized-loss functions. Predicted state,
GPP masks, current PFT fractions and Teacher-target extraction follow the same
formula as `trajectories_from_rollout`. Both use the per-artifact total over
eight seven-day windows; no missing averaging factor is identified. A shared
formula does not prove that the compiled forward trajectories are identical.

Production `_horizon_gradient` iterates windows 0-7 for each scheduled
artifact, in order 541,717,596,867,322,528,278,442. The failing zero-based
ordinals map to (artifact, window) as follows:

| Callback | Artifact | Window |
| ---: | ---: | ---: |
| 1 | 541 | 1 |
| 15 | 717 | 7 |
| 32 | 322 | 0 |
| 37 | 322 | 5 |
| 50 | 278 | 2 |
| 52 | 278 | 4 |
| 54 | 278 | 6 |

Replay artifact 322/window 5 (largest scalar difference) and window 1
(passing callback 33). Evaluate all eight production windows only to recover
their original global bias/count total. No other artifact is loaded.
The complete remote integration report confirms callback 33 differs by only
9.35990e-10, while callback 37 differs by -1.8336384e-5.

## Measurements

Compare the unchanged production score, the existing observation-gradient
score, science statistics computed separately from the verbose trace, and
statistics computed alongside/separately from compact production trajectories.
The compact diagnostic exposes the existing `window_trajectories` callable;
default production outputs and graphs are unchanged. Added diagnostic outputs
can themselves affect compiler optimization, so record their discrepancy from
the unchanged production evaluator rather than treating them as exact truth.

Report per-variable daily values/masks and sufficient-statistic differences,
trace fractions/active masks, and physical failures. If the observation score
matches verbose-trace statistics but both differ from production, investigate
the compiled rollout boundary before changing scientific loss assembly.

For each selected window, compute full seven-day streamed backward once.
Use its negative unit gradient only as a diagnostic direction. Compare the
gradient projection with central differences at L2 step sizes 1e-4 and 1e-5
through both unchanged production and verbose-observation paths, accounting
for actual float32 parameter displacement. Hold global statistics fixed for
this linearized scalar comparison; do not interpret it as a fresh full-panel
objective. Record values/errors without widening existing tolerances. No
whole-window reverse graph or monolithic JVP is required.

## Execution

Runner: `scripts/hpc/audit_conditioned_attention_seven_score.py` with the
matching `.sh` wrapper and existing conditioned bootstrap arguments. One free
gln01 V100, one process, 45-minute bound, incremental JSON under runtime outputs.
The result is diagnostic evidence, not model or continuation admission.
Local checks must exercise both windows, complete scalar/gradient paths,
serialization and launcher routing before launch. Use the result to choose
the smallest source-backed follow-up; do not restart the full integration
until the discrepancy has an explanation and the derivative consequence is
understood.
