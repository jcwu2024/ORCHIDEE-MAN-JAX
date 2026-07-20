# Restart Year Handoff Audit

Scope: local paper-case PFT14 driver handoff from one forcing year to the next.
This audit covers the in-memory year transition used by development runners;
it does not claim NetCDF restart-file parity.

## Fortran Runtime Contract

Fortran provenance:

- `fortran_run_scripts/paper_250919/Job0_bio` lines 315-322 rolls restart
  outputs from a completed year into the next executable year.
- `fortran_source/ORCHIDEE/src_driver/dim2_driver.f90` lines 839-908 advances
  each forcing-year timestep sequence from the new yearly driver counter.
- `fortran_source/ORCHIDEE/src_driver/dim2_driver.f90` lines 1144-1180 reuses
  previous surface state for the next main call.
- `fortran_source/ORCHIDEE/src_stomate/stomate.f90` lines 2409-2448 and
  4225-4364 carry mutable STOMATE state between calls and restart writes.
- `fortran_source/ORCHIDEE/src_stomate/stomate_io.f90` writes the broader
  STOMATE restart fields used by yearly rollover.

The source-backed behavior is: the new forcing year restarts its driver
timestep counter at zero, while the dynamic SECHIBA/STOMATE state is carried
from the previous completed year.

## JAX State Contract

Implemented entry points:

- `jax_orchidee.driver.orchestration.driver_year_handoff_state_gaps`
  checks whether a `DriverPreviousStepStatePacket` contains the state needed
  to seed a new forcing year.
- `jax_orchidee.driver.orchestration.rebase_driver_state_for_year_start`
  preserves the dynamic packet fields and provenance, but rebases `tstep` to
  `-1` so the existing next-step invariant `previous_state.tstep == tstep - 1`
  remains explicit for restart-year `tstep=0`.
- `jax_orchidee.driver.orchestration.paper_1961_driver_restart_year_start_day_result`
  runs day 1 of the restart year only if the handoff contract is complete.
- `jax_orchidee.driver.orchestration.paper_1961_driver_restart_year_multiday_modelout_lite_run`
  continues the restart year after day 1 using the normal later-day runner.

Missing handoff state is not fabricated. The restart-year runners report
`year_handoff_state` in `missing_components` when the previous-year packet is
absent or incomplete.

## Local Validation

Retained tests:

- `tests/parity/test_driver_1961_orchestration.py::test_driver_step_bundle_uses_requested_year_for_co2_ccanopy`
  confirms requested-year step bundles do not silently reuse 1961 CO2/ccanopy
  when a later year is requested.
- `tests/parity/test_driver_1961_orchestration.py::test_multiday_lite_end_state_satisfies_year_handoff_contract_and_rebases_counter`
  confirms a local multiday run emits a packet satisfying the handoff contract
  and that rebasing preserves component provenance.
- `tests/parity/test_driver_1961_orchestration.py::test_restart_year_start_day_result_reports_missing_handoff_state_without_fabricating`
  and
  `tests/parity/test_driver_1961_orchestration.py::test_restart_year_multiday_lite_reports_missing_handoff_state_without_fabricating`
  confirm incomplete restart state fails closed.

Retained dev smoke:

- `scripts/dev/run_restart_year_handoff_smoke.py`
  runs a short source-year sequence, checks handoff gaps, then runs a
  restart-year multiday sequence from that carried state.
- `scripts/dev/run_multiyear_modelout_lite.py`
  chains consecutive years in memory. Year 1 uses the cold-start lite runner;
  later years use the restart-year handoff runner and the carried
  `DriverPreviousStepStatePacket`.
- Latest multiyear CLI smoke:
  `outputs/multiyear_modelout_lite_1961_1962_1day_smoke.json`
  - years: 1961 and 1962
  - days per year: `1,1`
  - `ready_for_requested_years: true`
  - 1962 handoff gap count before year: `0`
- Annualized modelout smoke:
  `outputs/multiyear_modelout_lite_1961_1962_1day_with_annual_modelout.json`
  - same 1961-to-1962 in-memory handoff.
  - each completed year reports `annual_history_modelout`, computed from the
    local annual-history aggregation layer.
- Latest output:
  `outputs/restart_year_1961day3_to_1962day3_smoke.json`
  - source run: 1961 days 1-3, `ready_for_requested_days: true`
  - handoff gaps after source: none
  - restart run: 1962 days 1-3, `ready_for_requested_days: true`
  - restart missing components: none
  - restart final driver counter: `tstep=143`
- Full source-year smoke:
  `outputs/restart_year_1961day365_to_1962day3_smoke.json`
  - source run: 1961 days 1-365, `ready_for_requested_days: true`
  - handoff gaps after source: none
  - restart run: 1962 days 1-3, `ready_for_requested_days: true`
  - restart missing components: none
  - restart final driver counter: `tstep=143`
  - local timing: 1961 source run `361.57108110000263 s`; 1962 restart
    run `6.22090290000051 s`

## Limits

- The current validation is an in-memory dynamic-state handoff, not a
  byte-for-byte NetCDF restart-file read/write audit.
- The full source-year smoke confirms the in-memory state/counter handoff
  from 1961 day 365 into 1962 days 1-3. It is still not NetCDF restart-file
  parity.
- No 1962 Fortran trace has been introduced in this audit. Numerical parity
  for 1962 year-start outputs should be compared against a source-generated
  trace or an existing reference history before declaring cross-year numerical
  closure.

## 1964 Restart-Year Leaf-Age Boundary Fix

Status: fixed and validated for the active paper-case PFT14 path.

Fortran provenance:

- `fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90`, subroutine
  `npp_calc`, lines 447-490: maintenance respiration, growth respiration, and
  `lm_old(:,:) = biomass(:,:,ileaf,icarbon)` after possible maintenance tissue
  pumping and before adding `bm_alloc`.
- `fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90`, subroutine
  `npp_calc`, lines 542-587: leaf age class mass and fraction update using
  `lm_old` and `bm_alloc`.
- `fortran_source/ORCHIDEE/src_stomate/stomate_turnover.f90`, subroutine
  `turn`, lines 289-292: downstream `leaf_meanage` is the weighted sum of
  `leaf_age * leaf_frac`.

Bug found:

- The JAX age/SLA bookkeeping used the day-entry biomass as `lm_old`.
- Fortran uses biomass after maintenance respiration may have pumped tissue
  biomass, but before adding allocation.
- In 1964 this made PFT14 `leaf_frac` sum slightly above 1 entering turnover,
  raising `leaf_meanage` by about 0.5-0.6 days at days 90-120.
- The accumulated LAI deficit crossed a DIFFUCO LAI-layer threshold one day
  early, causing the day235 GPP cliff.

Implemented fix:

- `jax_orchidee/stomate/carbon_kernels.py::NPPUpdateResult` now carries
  `biomass_before_alloc`, the source-backed `lm_old` boundary.
- `jax_orchidee/stomate/integration.py::stomate_daily_carbon_explicit` passes
  that boundary to `npp_leaf_age_sla_age_update`.
- Regression:
  `tests/unit/test_stomate_integration.py::test_npp_age_sla_uses_post_maintenance_leaf_mass_for_leaf_fractions`.

Validation artifacts:

- Narrow Fortran trace:
  `outputs/server_1964_turn_vmax_trace_20260706_1430/orchjax_turn_vmax_pft14_1964_day090_120.txt`.
- Local day90/day120 after-fix diagnostic:
  `outputs/reference_mode/inspect_1964_turnover_classes_day090_120_after_lmold_fix.json`.
  `leaf_frac` sums are restored to 1.0 entering turnover; `leaf_meanage`
  day90/day120 errors drop from about 0.51/0.60 days to about
  0.0028/0.0026 days.
- Local day233-236 after-fix diagnostic:
  `outputs/reference_mode/inspect_1964_day233_236_after_lmold_fix.json`.
  Day235 GPP error drops from about `-1.68e-1` to about `-5.6e-7`.
- Full 1964 same-run-def annual validation:
  `outputs/reference_mode/multiyear_1964_from_1963_cache_after_lmold_fix_365d_ref_run_def.json`.
  Annual absolute errors: GPP `2.9e-6`, NPP `9.4e-6`, AGB `3.3e-5`,
  BGB `1.6e-5`.
