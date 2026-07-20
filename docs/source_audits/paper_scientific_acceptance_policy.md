# Paper PFT14 Scientific Acceptance Policy

## Purpose

This policy is fixed before the representative multi-landpoint pilot. It
separates development-time numerical diagnostics from the scientific decision
about whether compiled JAX output reproduces the downloaded Fortran paper
results. Thresholds must not be enlarged after observing a failed landpoint.

## Scientific Gate

The gate applies to annual PFT14 `AGB_model`, `BGB_model`, `GPP_model`, and
`NPP_model` for each requested landpoint and year.

- Per point-year-variable tolerance:
  `abs(JAX - Fortran) <= 1e-5 + 1e-3 * abs(Fortran)`.
- Across the requested year series, each variable must have absolute
  normalized bias no greater than `1e-3`.
- Across the requested year series, each variable must have normalized RMSE
  no greater than `1e-3`.
- Every requested year must complete and have annual reference truth.
- Missing, stopped, or non-finite output is a failure, not an accepted error.

These limits allow at most approximately 0.1% annual relative error while
still rejecting persistent drift. Correlation is reported for multi-year
series but is not used alone because a highly correlated result may retain a
scientifically relevant bias.

## Numerical Diagnostic Gate

The existing diagnostic tolerance remains
`abs_error <= 2e-6 + 1e-7 * abs(reference)` and applies to annual modelout,
underlying annual history fields, and available process diagnostics. Failure
of this stricter gate does not by itself fail scientific acceptance, but is
retained for first-divergence diagnosis and regression tracking.

## Failure Classification

Compiled mode runs first. If it fails the scientific gate, the same point and
window run in strict mode:

- strict passes: compiled performance-path regression;
- strict also fails: shared semantic, input-binding, aggregation, or reference
  mismatch;
- missing output or incomplete year: execution failure.

Server trace is not an acceptance prerequisite. It is considered only after
input identity, annual aggregation, strict/compiled classification, existing
source ledgers, and local micro-cases cannot explain a scientifically material
failure. Any such trace is limited to the first bad year/day and relevant
day-end fields.
