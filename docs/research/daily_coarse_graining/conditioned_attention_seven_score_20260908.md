# Seven-Day Observation/Production Score Localization

Status: verbose replay `c530f90` completed; compact adjoint-tape replay is
running at `d31e93c`.

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

Launched 2026-09-08 at approximately 17:02 CST on `gln01` (`ibc13b02n04`),
GPU 1, wrapper PID `62329`, commit
`c530f902e1dcb1c094695c288807b8d012b8b6aa`. Both GPUs were idle at the
prelaunch check. Server and local execution commit agree; shell syntax and
Ruff pass. The command is `bash scripts/hpc/audit_conditioned_attention_seven_score.sh`
from the project root, with `GPU_DEVICE=1` and `RUN_TIMEOUT=45m`.
The report/log names are `conditioned-seven-score-c530f90.json` / `.log`
under `runtime/outputs/diagnostics/`. The run completed in 1153.55476 seconds;
wrapper PID `62329` has exited. No weights were retained.

## Completed Verbose Replay

Both compact-trajectory variants give exactly the unchanged production score
on both selected windows. Separating the compact trajectories from loss
assembly changes neither score. The verbose-trace score matches the
observation-gradient score: window 5 differs from production by -1.8336384e-5,
while window 1 differs by only 9.35990e-10. Counts and masks agree; all recorded
physical failures are zero. This localizes the discrepancy to the compiled
rollout/output boundary, not a missing term in observation-loss assembly.

In window 5, differences become pronounced in maintenance respiration on day
5 (0.00252744), litter carbon on day 6 (0.11424480), and GPP on day 7
(0.00785883). These are absolute differences between compiled paths, not
errors against Teacher truth. Day-6 ecosystem-carbon difference is 0.11623933.
Window 1 maximum GPP difference is 6.51070e-7. This does not yet identify a
causal physical process or prove a particular compiler pass is responsible.

Directional derivatives remain consistent with actual float32 displacement:

| Window | Step | Production relative derivative error | Observation relative derivative error |
| ---: | ---: | ---: | ---: |
| 5 | 1e-4 | 0.013352% | 0.003078% |
| 5 | 1e-5 | 0.344881% | 0.040946% |
| 1 | 1e-4 | 0.001202% | 0.000879% |
| 1 | 1e-5 | 0.023644% | 0.018738% |

All secants and gradient projections have the descent sign; the reverse
executable compiled once. The checked derivative is credible but scalar
consistency still has to be restored before full integration admission.

## Compact Adjoint Tape: Fixed Follow-Up

The verbose trace exports the full prediction and fast-update diagnostics.
The proposed tape exports only complete continuous/discrete recurrent state,
state-defined masks, daily GPP, and current PFT fractions/active masks. It calls
the same canonical daily rollout and transition, without changing neural
weights, precision policy, forcing, process formulas, loss or gradient length.
This is an output projection for autodiff bookkeeping, not a neural-output
inventory change or a new architecture. Production training remains unchanged
until this candidate is admitted.

Run the same two windows, original eight-window totals, and the same two
finite-difference step sizes with `--trace-mode compact` (`TRACE_MODE=compact`
through the existing wrapper). Retain the original verbose evaluator only to
report per-day source-mapped state differences. Existing JSON names beginning
`verbose_` describe the recorded trace; `trace_mode` identifies which tape was
used. Compare tape scores with the unchanged production evaluator at the
existing atol=1e-6, rtol=1e-5; require physical failures zero and directional
derivative errors <=1% with matching descent signs on both windows/steps.
These bounds are unchanged from preceding score and derivative checks.

One free gln01 V100, 45-minute limit, zero retained updates and no fixed-monitor
read. If the compact tape passes, verify all seven previously failing windows
and the complete original integration before fixed-monitor evaluation. If it
does not pass, use the recorded state locations to select the next source
inspection; do not cycle through arbitrary compiler flags or relax tolerances.

Compact-tape local qualification passed 22 focused/related tests in 51.88
seconds. The compact-mode test exercises the actual canonical rollout scan,
state input reconstruction boundary, full streamed gradient and finite
differences, with an auxiliary output excluded from the tape. Ruff and diff
checks pass.

The compact replay launched at approximately 19:48 CST on 2026-09-08 on
free `gln01` GPU 1, wrapper PID `74821`, commit
`d31e93c03a38bec1b87ac3e216bf16da2b8c35ec`. Both cards were idle before launch.
The same shell command uses `TRACE_MODE=compact`, `GPU_DEVICE=1`, and
`RUN_TIMEOUT=45m`. Output is
`runtime/outputs/diagnostics/conditioned-compact-tape-d31e93c.json` / `.log`.
The run completed in 615.589 seconds and is not admitted. Window 5's score
gap remains -7.37018197e-6; production/observation derivative errors at 1e-5
are 1.258%/1.828%, above 1%. Window 1 passes and all physical failures are
zero. No retained update or paid job was started. The first state differences
include Day-1 snowheat; source inspection and the next bounded diagnosis are
recorded in `conditioned_attention_thermal_source_20260908.md`.
