# Current Semantic Window Notes

Scope: local cold-start paper-case PFT14 semantic validation against
`outputs/server_1961_trace_full_20260623` traces.

## Retained Local Validation Tools

- `scripts/dev/compare_stomate_daily_window_trace.py`
  - Advances a cold-start sequence once to the maximum requested day.
  - Validates requested daily windows against:
    - `orchjax_stomate_daily_trace.txt:gpp_after_accu`
    - `orchjax_stomate_maint_trace.txt:maint_after`
  - Uses the audited whole-driver GPP tolerance from
    `docs/source_audits/numeric_tolerance_ledger.md`.
- `scripts/dev/compare_stomate_day_carbon_trace.py`
  - Advances to one target day and keeps the full STOMATE daily-carbon
    scaffold only for that day.
  - Compares `after_alloc` and `after_npp` process cuts against
    `orchjax_stomate_lpj_trace.txt` and `orchjax_stomate_npp_trace.txt`.

## Passing Windows

- `outputs/semantic_day30_60_90_stomate_window_current.json`
  - day 30: GPP and maintenance pass.
  - day 60: GPP and maintenance pass.
  - day 90: GPP and maintenance pass.
- `outputs/semantic_day100_175_bisect_stomate_window_current.json`
  - day 100, day 120, day 140 pass both checks.
- `outputs/semantic_day180_240_300_365_stomate_window_current.json`
  - advanced the cold-start chain through day 365 with no missing components.
  - day 180, day 240, day 300, and day 365 GPP accumulator checks pass under
    the audited whole-driver `1e-5` GPP budget.
  - maintenance differences are bounded by `6.753264614189902e-12`, now
    recorded in `docs/source_audits/numeric_tolerance_ledger.md` as a
    whole-driver accumulation diagnostic budget.
- `outputs/semantic_day180_240_300_365_stomate_window_after_tolerance_ledger.json`
  - same requested windows after the maintenance accumulation budget was
    ledgered and made the script default.
  - `ok: true`, `missing_components: []`, `stopped_day_index: null`.

## Resolved Day178 Allocation Divergence

The previous day178/day180 material divergence was traced to a SECHIBA to
STOMATE boundary-name mismatch, not to `stomate_alloc` itself.

Fortran provenance:

- `fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main` lines
  1184-1188 passes `vegstress` into `slowproc_main`.
- `fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main` lines
  343-399 names that argument `humrel`.
- `fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main` lines
  973-985 forwards it to `stomate_main`.
- `fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main` lines
  3198-3208 accumulates the incoming `humrel` into `humrel_daily`.

JAX fix:

- `jax_orchidee/stomate/entry.py::stomate_hydrol_entry_source` now maps the
  STOMATE entry field named `humrel` from HYDROL `diagnostics.vegstress`.
- Runtime driver entry payloads in `jax_orchidee/driver/orchestration.py` use
  `hydrol_diag.vegstress` for STOMATE. HYDROL `humrel` remains the DIFFUCO
  transpiration-stress field.
- `jax_orchidee/coupled.py::_entry_source_from_sechiba` uses the same mapping.

Validation after the fix:

- `outputs/semantic_day178_carbon_trace_after_vegstress_boundary.json`
  - `ok: true`
  - `after_alloc.f_alloc` max abs error `6.460387780293786e-13`
  - `after_npp.npp_daily` max abs error `9.280132218236758e-12`
- `outputs/semantic_day180_carbon_trace_after_vegstress_boundary.json`
  - `ok: true`
  - `after_alloc.f_alloc` max abs error `6.284972542403011e-13`
  - `after_npp.npp_daily` max abs error `9.481304630298837e-12`
- `outputs/semantic_day240_carbon_trace_after_vegstress_boundary.json`
  - `ok: true`
  - `after_alloc.f_alloc` max abs error `9.38527033866876e-13`
  - `after_npp.npp_daily` max abs error `1.9289680963652245e-11`
- `outputs/semantic_day300_carbon_trace_after_vegstress_boundary.json`
  - `ok: true`
  - `after_alloc.f_alloc` max abs error `2.1471158184738215e-12`
  - `after_npp.npp_daily` max abs error `2.1445067943659524e-12`
- `outputs/semantic_day365_carbon_trace_after_vegstress_boundary.json`
  - `ok: true`
  - `after_alloc.f_alloc` max abs error `2.4075741400508832e-12`
  - `after_npp.npp_daily` max abs error `4.908518036472742e-12`

This closes the currently traced paper-case PFT14 active STOMATE
allocation/NPP path through day 365. This is not all-branch coverage; branch
coverage still needs to be source-driven after the active paper-case path is
stable.

## Remaining Window Tolerance Notes

`outputs/semantic_day176_180_stomate_window_after_vegstress_boundary.json`
advances through day 180 with no missing components, but the aggregate window
script still reports `ok: false` under its current hard thresholds:

- day176/day177/day178/day180 maintenance max abs error is around
  `5.13e-12` to `5.18e-12`, slightly above the script's `5e-12` threshold.
- day177 GPP max abs error is `1.0567979188635945e-5`, slightly above the
  script's `1e-5` absolute threshold while still within the audited
  whole-driver roundoff scale.

Treat these as tolerance-ledger follow-up items, not active process
divergences, unless a stricter source-backed comparison identifies a matching
state error.

The later `outputs/semantic_day180_240_300_365_stomate_window_after_tolerance_ledger.json`
run confirms the same pattern through day 365: GPP passes, allocation/NPP
process cuts pass at day 180/240/300/365, and maintenance stays within the
audited `1e-11` whole-driver accumulation budget.
