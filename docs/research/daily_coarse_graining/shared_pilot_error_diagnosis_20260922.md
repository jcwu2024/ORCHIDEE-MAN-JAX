# Shared pilot held-out curve diagnosis

## Scope

This is an offline readout of the saved physical curves in
`outputs/shared-learning-4408daf-gpu.json`. It runs no model or Teacher
inference, uses no sealed test data, and changes neither the architecture nor
the objective. The input is the same 24 fixed held-out T30 development windows
(eight joint, eight spatial, eight temporal) at updates 0, 20, and 100 used by
`shared_learning_results_20260922.md`.

For each day, flux errors are cumulative from day 1 through that day. Stock
errors are change from the common actual day-0 stock. Masks are honored exactly
as in the saved curves. The helper and machine-readable result are:

- `scripts/hpc/analyze_shared_pilot_curves.py`
- `outputs/shared-learning-4408daf-curve-diagnosis.json`

The previously reported update-100 pooled T30 RMSEs (GPP 30.332, NPP 28.169,
AGB change 20.005, and BGB change 8.135 gC/m2) and their positive biases are
not new findings. The window concentration, daily development, component
association, and coverage results below are new readouts from the saved curves.

## Primary-error concentration at update 100

| Target | Bias | Largest-window SSE share | Top-three SSE share | Effective windows from SSE |
| --- | ---: | ---: | ---: | ---: |
| GPP | +12.686 | 31.47% | 57.07% | 6.27 |
| NPP | +16.511 | 28.32% | 58.78% | 6.63 |
| AGB change | +11.832 | 25.51% | 56.96% | 6.93 |
| BGB change | +4.576 | 34.05% | 61.20% | 5.79 |

The same three point-year windows lead all four targets:

| Artifact / split | Point-year / start row | GPP error | NPP error | AGB error | BGB error |
| --- | --- | ---: | ---: | ---: | ---: |
| 14 / temporal | `271.0-111.0`, 2007 / 67 | +83.36 | +73.43 | +49.50 | +23.26 |
| 27 / spatial | `131.0-065.0`, 1986 / 148 | +57.43 | +57.94 | +42.99 | +14.61 |
| 26 / spatial | `001.0-073.0`, 1974 / 132 | +48.52 | +49.43 | +34.25 | +14.75 |

Artifact 10, temporal point-year `187.0-091.0` in 2006 starting at row 8,
is the largest negative offset for every primary target: -47.96 GPP, -32.21
NPP, -21.37 AGB, and -10.46 BGB. Consequently, pooled bias hides substantial
between-window cancellation. GPP has 17 positive and 7 negative endpoint
errors: their sums are +409.15 and -104.68 gC/m2. NPP and AGB are positive in
20/24 and 21/24 windows; BGB is positive in 19, negative in 3, and exactly zero
in 2.

Across the 24 windows, signed endpoint-error correlations among the four
primary quantities range from 0.953 to 0.996. This establishes co-location,
not causality: shared forcing, state, algebraic relations, and recurrent error
can all induce that association. It does not identify which process should be
changed.

## Development within 30 days

Each cell is pooled cumulative RMSE / bias in gC/m2 at update 100.

| Day | GPP | NPP | AGB change | BGB change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.033 / -0.115 | 0.861 / +0.173 | 0.594 / +0.126 | 0.275 / +0.047 |
| 7 | 6.671 / +0.727 | 5.908 / +2.287 | 4.163 / +1.650 | 1.784 / +0.633 |
| 14 | 12.982 / +3.487 | 12.176 / +6.075 | 8.606 / +4.351 | 3.616 / +1.705 |
| 21 | 20.018 / +7.039 | 18.904 / +10.372 | 13.394 / +7.438 | 5.540 / +2.888 |
| 30 | 30.332 / +12.686 | 28.169 / +16.511 | 20.005 / +11.832 | 8.135 / +4.576 |

Cumulative RMSE increases on every day for each target. GPP pooled bias is
slightly negative for days 1-3 and positive from day 4; NPP, AGB, and BGB bias
is positive from day 1. RMSE per elapsed day changes only modestly between day
7 and day 30 (GPP 0.953 to 1.011, NPP 0.844 to 0.939, AGB 0.595 to 0.667,
BGB 0.255 to 0.271). The curves therefore show persistent accumulation, not a
single late endpoint excursion. They do not by themselves distinguish daily
readout bias from causal recurrent-state feedback. In particular, cumulative
RMSE naturally grows with horizon even for zero-mean daily errors; the evidence
for signed drift here is the bias trajectory, not increasing RMSE alone. These
30-day curves support no claim about 50-year behavior.

## Respiration compensation

At update 100, T30 maintenance error has bias -9.876 and RMSE 11.718, while
growth-respiration error has bias +6.052 and RMSE 9.710. Their exact saved-curve
sum is autotrophic respiration (maximum identity residual `3.73e-14`), whose
bias is -3.825 and RMSE 7.504.

Maintenance and growth errors have Pearson correlation -0.582 across windows;
16/24 have opposite signs. Relative to the sum of the two components' absolute
errors, 68.7% cancels in their sum. The MSE decomposition is also exact:
`56.315 = 137.316 + 94.282 - 175.282`. Thus the smaller aggregate RA error
does not demonstrate that both components are accurate. The correlation and
cancellation are descriptive associations, not evidence that training one
component caused correction in the other.

## Canopy, litter, DOC, and RH

Canopy organic-carbon change remains broadly negative: bias -115.23 gC/m2,
19/24 negative windows, and RMSE 182.34. Its top three windows account for
60.56% of SSE; 60.37% of all canopy SSE lies in the spatial split. The largest
errors are artifact 27 (-495.27), artifact 26 (-388.14), and temporal artifact
12, `123.0-097.0` in 2006 (-295.42).

Litter error is even more concentrated. Aboveground litter's top three windows
account for 82.43% of SSE: artifact 27 (-15.76), joint artifact 44 at
`211.0-059.0`, 2006 (-15.71), and artifact 26 (-12.80). Belowground litter's
top three account for 70.26%: artifacts 26 (-19.72), 27 (-18.28), and spatial
artifact 29 at `141.0-071.0`, 1993 (-16.50). Spatial windows contribute 62.22%
and 73.63% of above- and belowground litter SSE. The update-100 pooled
belowground litter RMSE improves from update 20 but remains worse than update
0, as already reported; the new localization shows that this conclusion is not
uniform across the panel. DOC is similarly concentrated (70.40% of SSE in its
top three), so its update-20 to update-100 regression should also be read as a
distributional result, not one homogeneous failure.

The T30 panel contains 720 defined daily RH target values and zero nonzero
Teacher values at tolerance `1e-12`; predictions are also zero throughout.
Therefore neither zero RH error nor continuation on this panel can demonstrate
a learned nonzero RH response.

## Criteria for the declared continuation

At the declared update 100/200/300/500 checkpoints, retain the same fixed
panels and report all of the following without introducing a post hoc pass
threshold:

1. Pooled RMSE and signed bias for GPP, NPP, AGB, and BGB at days 1, 7, 14,
   21, and 30, including whether bias magnitude continues to accumulate.
2. Per-window errors plus top-one/top-three SSE shares and effective window
   counts. Report artifacts 14, 26, and 27 explicitly, but also whether the
   remaining windows improve; aggregate progress concentrated in those three
   is not broad-panel evidence. Their rank is a development-only diagnostic,
   not a reason to oversample them during training.
3. Artifact 10 and the positive/negative error sums, so a smaller pooled bias
   cannot be mistaken for uniformly smaller error. Do not drop that opposing
   window post hoc.
4. Maintenance and growth respiration separately, alongside RA and their
   cancellation diagnostics. An RA improvement produced only by stronger
   component cancellation is not component-level progress.
5. Canopy, above- and belowground litter, and DOC by split and leading window,
   not only their pooled metrics. Keep all existing physical/hard gates.
6. Continue to label RH accuracy as unsupported by this panel. A nonzero-RH
   evaluation requires separately declared development coverage; it cannot be
   inferred from these zero-target curves.

These are diagnostic readouts for the already-declared extension, not final
scientific acceptance thresholds and not evidence for a loss, head, or
architecture change. The development windows have been inspected and are no
longer independent of method decisions; final claims still require the
predeclared longer-horizon and held-out evaluations in the active roadmap.
