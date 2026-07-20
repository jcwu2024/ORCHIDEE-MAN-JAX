# STOMATE Local Closure Before Fortran Trace

Scope: paper-case PFT14 mangrove active branch, with code retaining
`npts/nvm/pool/element` axes. This document summarizes the maximum local state
reached before exact Fortran process traces are required.

## Implemented Local Pieces

Parameter and modelout boundary:

- `jax_orchidee/stomate/parameters.py`
  - Loads base and reference `run.def` PFT14 overrides.
  - Source provenance: `pft_parameters.f90` audited parameter-read spans and
    `constantes.f90` mangrove constants.
- `jax_orchidee/stomate/modelout.py`
  - Implements paper modelout formula for `AGB_model`, `BGB_model`,
    `GPP_model`, `NPP_model`.
  - Source provenance: `stomate_lpj.f90` history writes and paper script lines
    50, 60-76, 115-118.

Independent carbon kernels:

- `set_lai_from_biomass`
  - Source provenance: `src_stomate/stomate_lai.f90::setlai` lines 58-88 and
    inline non-crop update in `stomate.f90` lines 4200-4208.
- `root_temperature`, `maintenance_respiration`
  - Source provenance: `stomate_resp.f90::maint_respiration` lines 122-170,
    203-232, 234-303, 319-376.
- `agr_allocation_split`
  - Source provenance: closed AGR split in `stomate_alloc.f90::alloc` lines
    760-834. It requires caller-supplied `LtoLSR`, `StoLSR`, `RtoLSR`, and
    `alloc_sap_above`.
- `npp_closed_update`
  - Source provenance: `stomate_npp.f90::npp_calc` lines 116-180, 230-247,
    280-295, 299-386, 447-531. It stops before leaf-age/SLA bookkeeping.

Daily scheduling helpers:

- `stomate_accumulate_daily`
  - Source provenance: `stomate.f90` lines 3198-3208 and `stomate_accu`
    variants lines 9341-9411.
- `sum_resp_maint_radia`, `accumulate_resp_maint_part`
  - Source provenance: `stomate.f90` lines 3254-3267.
- `reset_daily_on_slow`
  - Source provenance: `stomate.f90` lines 4950-5000.
- `prepare_daily_carbon_inputs`, `require_explicit_f_alloc`
  - Source provenance: `StomateLpj` process boundary lines 1093-1131.
  - The 2026-06-23 server trace can now supply explicit PFT14 `f_alloc` from
    `outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_lpj_trace.txt`,
    tag `after_alloc`. This does not implement or validate full allocation.

Reference/history readers:

- `jax_orchidee/stomate/reference.py`
  - Inventories local STOMATE files.
  - Reads restart `biomass`, `maint_resp`, `gpp_daily`, `npp_daily`,
    `resp_maint`, `resp_growth`, `PFTpresent`.
  - Packs history modelout fields for formula validation.
  - Boundary only: readers do not infer processes or parameters.

Trace schema:

- `jax_orchidee/stomate/trace_schema.py`
  - Defines grouped required columns for `gpp_accumulation`,
    `maintenance_respiration`, `resp_maint_part_accumulation`,
    `alloc_input_output`, `npp_calc_input_output`,
    `biomass_agr_before_after`, and `modelout_history`.
  - Validates supplied CSV headers or mapping-like data and reports missing
    columns by group.

Explicit integration adapter:

- `jax_orchidee/stomate/integration.py`
  - `stomate_daily_carbon_explicit` composes explicit
    `gpp_daily + biomass_before + resp_maint_part + f_alloc + PFTpresent +
    frac_growthresp` into `npp_closed_update`.
  - It never computes `f_alloc`, phenology, season stress, turnover, mortality,
    or full `StomateLpj`.

## Local Reference Truth Found

Files under the paper reference run:

- `stomate_start.nc`
- `stomate_restart.nc`
- `stomate_history_1961.nc` through `stomate_history_2010.nc`

Restart is state/readiness truth for:

- `biomass`, `maint_resp`, `resp_maint`, `resp_growth`, `gpp_daily`,
  `npp_daily`, `PFTpresent`, `sla_calc`, `t2m_daily`, `tsoil_daily`,
  `t2m_longterm`, `moiavail_week`, `tsoil_month`, `soilhum_month`,
  `leaf_age`, `leaf_frac`, `age`.

History is output/formula validation truth for:

- biomass pool outputs, AGR pool outputs, `GPP`, `NPP`, `MAINT_RESP`,
  `GROWTH_RESP`, `BM_ALLOC_*`, `NPP_ABOVE`, `NPP_BELOW`.

Important boundary:

- Yearly history fields are not process before/after traces and are not
  pointwise equal to final restart biomass. They must not be used to fabricate
  missing daily state.

## Trace-Backed Slice Added 2026-06-23

New local trace package:
`outputs/server_1961_trace_full_20260623/MANIFEST.txt`.

Lightweight parity now covers:

1. First PFT14 daily GPP accumulation record:
   `orchjax_stomate_daily_trace.txt:gpp_before_accu/gpp_after_accu` closes
   exactly with `stomate_accumulate_daily`.
2. First nonzero PFT14 maintenance group:
   `orchjax_stomate_maint_trace.txt:maint_after` closes exactly for
   `sum_resp_maint_radia` and `accumulate_resp_maint_part`.
3. First PFT14 allocation boundary group:
   `orchjax_stomate_lpj_trace.txt:after_alloc` exposes explicit `f_alloc` and
   clears only that boundary field in `prepare_daily_carbon_inputs`.

The local reader for maint/alloc trace groups is
`jax_orchidee/stomate/trace.py`; it streams fixed-format records and stops at
the first complete PFT14 group needed by tests.

## Hard Blocks

Full allocation is still blocked by `stomate_alloc.f90` lines 335-759 and
600-755. The new trace exposes `f_alloc`, but not enough upstream allocation
state to validate the process that produced it:

- `alloc_sap_above` depends on age and allocation parameters.
- `limit_L` depends on `lai_around`.
- `limit_W` depends on `moiavail_week`.
- `limit_N` depends on `t_nitrogen`, `h_nitrogen`, `tsoil_month`,
  `soilhum_month`, and root-profile integration.
- `LtoLSR`, `StoLSR`, `RtoLSR` are internal allocation ratios and are not
  stored in local restart/history.
- `f_alloc` is not present in restart/history.

Full `npp_calc` process parity is blocked by missing before/after trace:

- Current JAX can run closed algebra only with explicit `f_alloc`.
- Restart contains final state, not the process cut before/after `npp_calc`.
- History exposes partial `BM_ALLOC_*` diagnostics, not full-pool `bm_alloc`.
- `orchjax_stomate_npp_trace.txt:after_npp` exposes after-NPP records by part,
  but the current audited field set is insufficient for an end-to-end
  `npp_closed_update` parity test without guessing pre-state or `bm_alloc`.

Full `StomateLpj`/STOMATE loop is blocked by upstream processes:

- phenology, season, allocation stress, turnover, mortality, fire, light,
  establishment, land-cover change, vmax, and crop branches are not locally
  closed for this MVP.

## Completion Estimate

For the paper-case PFT14 STOMATE carbon-output MVP before exact trace:

- Parameter/modelout/readers: complete for local validation.
- Independent closed carbon kernels: mostly complete for maintenance, AGR split
  helper, NPP algebra, and setlai.
- Daily scheduling skeleton: complete for accumulation/reset/boundary packing.
- Process parity against Fortran active path: blocked at allocation and
  before/after `npp_calc` trace.

Estimated local STOMATE readiness before trace: about 60-65% of the
days-level carbon MVP scaffolding, but 0% claim of full process parity.

## First Work After Trace Exists

The first task after exact Fortran trace exists is a one-day, one-point PFT14
adapter parity test:

1. Extend or audit `orchjax_stomate_npp_trace.txt` so the field names and
   before/after values cover `biomass_before_npp`, `gpp_daily`, `f_alloc`,
   `resp_maint_part`, `PFTpresent`, `bm_alloc_after`, `resp_maint_after`,
   `resp_growth_after`, and `npp_daily_after`.
2. Use traced `f_alloc` in `stomate_daily_carbon_explicit`.
3. Compare `bm_alloc`, `resp_growth`, `npp_daily`, `biomass_after_npp`, and AGR
   pools before/after to trace.

If any of these fail, do not tune parameters. Re-audit the exact Fortran span
for that process cut.
