# Gate D1: Canonical Teacher Physical-Parameter Gradients

Status: accepted locally; the aggregate evidence command binds acceptance to
the current Git commit and manifest hash.

## Claim

Gate D1 validates selected physical-parameter derivatives of the canonical
half-hour PFT14 Teacher. It does not validate neural-network weight gradients,
does not validate a future daily surrogate, and does not claim that every
ORCHIDEE parameter is suitable for inversion.

The acceptance policy compares JAX forward AD and reverse AD with centered
finite differences at `h` and `h/2`, followed by Richardson extrapolation.
Smooth active pairs require at most 1% relative error. Inactive pairs use an
explicit absolute-zero gate. Threshold-adjacent cases are reported separately
without claiming a smooth derivative at a discontinuity.

## Accepted Matrix

| Scope | Pairs | Result |
| --- | ---: | --- |
| Source-backed process cases | 7 | passed |
| Complete 1961 Day 2 Teacher | 7 | passed |
| Complete 1961 Days 2-3 Teacher | 3 | passed |
| 1962 Day 1 after yearly restart rebase | 4 | passed |
| Selected source-extracted Fortran finite differences | 2 | passed |

The process matrix covers active, inactive, and threshold-adjacent classes and
includes parameters outside the four paper-calibrated values. The complete
Teacher matrices retain all 48 half-hour transitions per day. The multiday
case differentiates one 96-step state transition, and the restart case uses a
model-produced state, the audited yearly rebase, 1962 forcing, and first-call
lifecycle behavior.

The two selected Fortran comparisons are:

- `vcmax25 -> VCMAX` in the original `stomate_vmax.f90::vmax` procedure;
- `maint_resp_slope_c -> maintenance respiration` in the original
  `stomate_resp.f90::maint_respiration` procedure.

The harnesses add only dimensions, module parameters, IO, and a parameter
input. Scientific procedure bytes are extracted unchanged and recorded by
SHA256. Fortran `h/h2` finite differences agree with JAX forward/reverse AD,
and their baseline forward values agree within `1e-12` relative error.

## Defects Repaired

The initial complete-day matrix found finite forward AD and finite differences
but `NaN` reverse AD for `g0 -> Day 2 GPP`. Bounded half-hour diagnosis proved
that the local photosynthesis derivative was correct and localized the defect
to dormant first-step branches whose undefined intermediates entered the
reverse graph through day state.

The accepted Teacher changes prevent source-inactive divisions, logarithms,
and undefined PFT diagnostics from entering active calculations in HYDROL,
CONDVEG, DIFFUCO, and ENERBIL. They preserve active forward formulas and retain
intentional undefined PFT diagnostics at their public diagnostic boundary.
The restart matrix additionally found and repaired a Python cache-key
conversion of traced `altmax`; traced yearly first-step root initialization now
uses the same pure-JAX source-backed function, while ordinary host constants
retain the existing cache.

After repair, the original `g0 -> Day 2 GPP` pair has forward/reverse absolute
error about `2e-14` and AD/Richardson relative error about `1e-8`. The multiday
`vcmax25 -> Day 3 GPP` pair has forward/reverse absolute error below `5e-17`
and AD/Richardson relative error about `4.1e-10`.

## Commands And Evidence

```powershell
conda run -n ORCJAX python scripts/dev/verify_teacher_physical_parameter_gradients.py --scope one-day --output outputs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients/one_day_comparison.json
conda run -n ORCJAX python scripts/dev/verify_teacher_physical_parameter_gradients.py --scope multiday --output outputs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients/multiday_comparison.json
conda run -n ORCJAX python scripts/dev/verify_teacher_physical_parameter_gradients.py --scope restart --output outputs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients/restart_comparison.json
conda run -n ORCJAX python scripts/dev/verify_selected_fortran_parameter_gradients.py
conda run -n ORCJAX python scripts/dev/aggregate_teacher_physical_parameter_gradients.py
```

The aggregate command fails unless all declared cases pass, all values are
finite, all Teacher evidence uses the current manifest hash, and all Teacher
assets bind to the current Git commit. Evidence paths and declared pair counts
are frozen in
`manifests/coarse_graining/teacher_physical_parameter_gradient_v1.json`.

The complete-day reverse graphs are intentionally large. On the local CPU,
the one-day and restart matrices each took about 20 minutes, while the two-day
matrix took about 34 minutes; most time was XLA reverse-graph compilation.
This cost belongs to a release/scientific gate, not ordinary regression tests.

## Remaining Boundary

Gate D1 validates the canonical Teacher only. A future accepted daily
surrogate still requires Gate D2 against Teacher/Fortran parameter responses
over declared ranges and held-out parameter combinations before it can be used
for inversion or scientific sensitivity claims.

Before adding more tunable channels, build a source-driven candidate registry.
Each parameter must record its owner, process consumer, continuous or discrete
nature, valid range or prior, expected observables, confounding risks, and
whether it belongs to retained exact formulas or the learned daily operator.
Only candidates with identifiable scientific meaning and a passing gradient
matrix should be promoted to an inversion interface.
