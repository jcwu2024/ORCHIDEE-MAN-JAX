# Numeric Tolerance Ledger

Purpose: record numerically bounded, source-backed differences that are not
process gaps. This ledger does not relax kernel parity by default. It only
defines where whole-run comparisons may include audited cross-runtime floating
point roundoff.

## Warm-start HYDROL drainage at 069 and 319

The PFT14 warm-start runs use 11 liquid-water layers. For the captured profiles at 069/2010 step 858482 and 319/1962 step 17521, initial column storage agrees exactly between the original Fortran accumulation order and the JAX reduction. The final storage from the Fortran order is lower by one binary64 ULP, `2.842170943040401e-14 kg/m²`, at stores of about 162 and 130 kg/m². Trial drainage is `1.0625047875804086e-14 kg/m²` per 1800-second step. The signed budget residual determines whether HYDROL adds water to drainage or subtracts up to the available drainage. The 48 half-hour carbon outputs of each first-day capture match their respective annual C108 records. [Captured operands and source-order component](../../outputs/reviews/main-physics-20261127/representative-warm-drainage-origin/components/replay.json) · [Independent review](../../outputs/reviews/main-physics-20261127/representative-warm-drainage-origin/review/README.md)

- **069, step 858482.** JAX leaves the trial drainage in place. With the same liquid profiles the source-order residual is −1 ULP. The resulting drainage of `3.904675730620809e-14 kg/m²` equals the native short-window HYDROL write and STOMATE read. Changing only the water input in the carbon replay recovers the two organic components and DIC to a maximum relative difference of `5.68e-16`. The strict water comparison still exceeds its original limit of `1.0000000000039047e-14 kg/m²`. The native short-window log records the final water transfer. [Native producer and receiver](../../outputs/reviews/main-physics-20261127/representative-warm-drainage-origin/069-native/budget-replay.json)
- **319, step 17521.** The observed JAX residual is +1 ULP and consumes all trial drainage. Source-order accumulation on the same profiles gives a zero residual and positive drainage. The three resulting carbon fluxes agree with native B to within `3.51e-10` relative. B recorded positive carbon production and reception. Its initial moisture differs slightly from the JAX restart, and its own budget operands and correction arm are unavailable. [Native start and carbon measurements](../../outputs/reviews/main-physics-20261127/representative-warm-drainage-origin/319-native/README.md)

This attribution covers the two controlled steps. At 069 step 859408 and 319 step 34854 the native water budget, liquid-water denominator and free DOC pools are still missing. The paired annual or partial-year organic-drainage differences remain `5.77096e-26` and `1.94223e-28 gC/m²` of grid-cell area respectively. The four point-years passed all 16 predeclared paper-indicator comparisons. [Numerical evidence and limits](../../outputs/reviews/main-physics-20261127/representative-warm-drainage-origin-acceptance/controller-acceptance.json) · [Annual comparisons](../../outputs/reviews/main-physics-20261127/representative-annual-c108/README.md)

## Driver SWdown Solar Redistribution

- Status: accepted numeric roundoff budget for whole-driver paper-case
  comparisons.
- Scope: active 1961 paper-case forcing path, single point PFT14, CRUNCEP
  6-hour forcing redistributed to 1800 s SECHIBA steps.
- Fortran source:
  - `fortran_source/ORCHIDEE/src_driver/readdim2.f90::forcing_read_interpol`
    lines 1526-1550 compute `mean_coszang`.
  - `fortran_source/ORCHIDEE/src_driver/readdim2.f90::forcing_read_interpol`
    lines 1599-1639 compute model-step `coszang` and
    `swdown = swdown_n * coszang / mean_coszang`.
  - `fortran_source/ORCHIDEE/src_global/solar.f90::solarang` lines 55-181
    compute the GSWP solar angle.
- Validation truth:
  - `outputs/server_1961_diffuco_active_after_main_trace_20260624_2244/traces/orchjax_diffuco_trans_co2_trace.txt`
    tag `after_diffuco_trans_co2_pft14`, field `swdown`.
  - Local regression:
    `tests/parity/test_driver_1961_domain.py::test_1961_model_step_swdown_matches_active_fortran_solarang_trace`.
- Observed bound:
  - Full scanned PFT14 `diffuco_trans_co2` trace rows: 17,476.
  - Max local-vs-Fortran `SWdown` absolute error: `2.201900368703491e-7`.
  - P99 absolute error: `1.6809082836743983e-7`.
  - The selected Day14 first-divergence row `kjit=645` differs by
    `1.8612383989591308e-7`.
- Downstream impact observed in current local GPP daily accumulation checks:
  - Day3 max abs error: `1.1692100088112056e-6`.
  - Day7 max abs error: `5.369147402234375e-7`.
  - Day14 max abs error: `8.77348065841943e-6`.
  - Day30 max abs error: `9.527808288112283e-6`.
  - Day60 max abs error: `6.8564258981496096e-6`.
  - Current output files:
    `outputs/day3_gpp_compare_current.json`,
    `outputs/day7_gpp_compare_current.json`,
    `outputs/day14_gpp_compare_current.json`,
    `outputs/day30_gpp_compare_current.json`,
    `outputs/day60_gpp_compare_current.json`.
- Interpretation:
  - Same-input `diffuco_trans_co2` parity remains at machine precision; this
    budget is for driver-forcing-to-daily-output comparisons that include
    `solar.f90::solarang` trigonometric roundoff.
  - Do not compensate with hard-coded offsets, tuned constants, or trace-based
    rounding. Revisit only if the forcing difference exceeds `2.3e-7`, if a
    threshold branch flips, or if a same-input process kernel fails its strict
    parity tolerance.

## Tolerance Policy

- Process kernels with identical inputs should continue to use strict parity
  tolerances, normally `1e-12` or tighter when the Fortran trace supports it.
- Whole-driver multi-day comparisons may include the audited forcing budget
  above. Current paper-case daily GPP comparisons should treat `<=1e-5` max
  absolute error as closed unless a first-divergence check attributes the error
  to a process kernel rather than the audited forcing path.
- `scripts/dev/compare_stomate_daily_gpp_trace.py --atol 1e-5` is the
  intended command form for whole-driver windows that include this forcing
  budget. Keep the default `1e-12` for strict same-input diagnostics.
- Any new tolerance entry must include Fortran file/subroutine/line-span
  provenance, trace or reference truth, observed bounds, and an explicit
  statement of what remains strict.

## STOMATE Maintenance Whole-Driver Accumulation

- Status: accepted numeric roundoff budget for whole-driver daily-window
  diagnostics only.
- Scope: active 1961 paper-case cold-start PFT14 path, daily windows advanced
  from local forcing through SECHIBA and STOMATE maintenance respiration.
- Fortran source:
  - `fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main` lines
    3244-3267 schedules maintenance respiration and accumulates
    `resp_maint_part`.
  - `fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90::maint_respiration`
    lines 122-376 computes part-wise maintenance respiration.
- Validation truth:
  - `outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_maint_trace.txt`
    tag `maint_after`.
  - Local diagnostic:
    `scripts/dev/compare_stomate_daily_window_trace.py`.
- Observed bound:
  - Day180 max abs error: `5.130673663700236e-12`.
  - Day240 max abs error: `5.6600280018415106e-12`.
  - Day300 max abs error: `6.753264614189902e-12`.
  - Day365 max abs error: `5.408784531368838e-12`.
  - Current output file:
    `outputs/semantic_day180_240_300_365_stomate_window_current.json`.
- Interpretation:
  - These windows have no missing components and no matching allocation/NPP
    process divergence; the differences are cross-runtime floating
    accumulation noise at the whole-driver window boundary.
  - Same-input maintenance kernels and focused maintenance unit tests remain
    strict. Do not tune maintenance coefficients, reorder Fortran semantics,
    or round model state to meet this diagnostic budget.
  - `scripts/dev/compare_stomate_daily_window_trace.py` uses `--maint-atol
    1e-11` by default for this whole-driver window check.
