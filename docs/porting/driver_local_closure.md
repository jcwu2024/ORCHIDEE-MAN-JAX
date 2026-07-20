# Driver Local Closure

This document summarizes the local Driver/Grid closure after replacing the
paper-case static trace dependency with source-backed readers and validating
those readers against the 2026-06-23 fixed-format server trace package under
`outputs/server_1961_trace_full_20260623`.

## Locally Closed

The driver layer can now provide a verified first-step boundary bundle for the
paper 1961 active path:

- domain selection from `configs/orchidee_man_250919.yaml` and
  `data/forcing/cruncep_twodeg_1961.nc`
- `nbindex=1`, `kindex=[1]`, `lalo=[[21,109]]`, selected
  `nav_lon=109`, `nav_lat=21`
- first forcing timestep fields:
  `Tair`, `PSurf`, `Qair`, `Wind_E`, `Wind_N`, `Rainf`, `Snowf`, `SWdown`,
  `LWdown`, `Height_Lev1`, `Height_Levuv`
- annual CO2 and land-point `ccanopy`
- water-table positive/differential text sequences as driver inputs
- run scalars for active PFT14 setup
- imposed `veget_max[npts,nvm]` and `soiltile[npts,nstm]`
- restart validation anchors for `njsc`, clay/sand fraction, bulk density,
  pH, poor soils, `wtp`, and `wt_ab_tide`
- `IntersurfFirstStepPayload` with Fortran-facing names for ready fields
- fixed-format trace readers for the 2026-06-23 server package:
  `orchjax_driver_forcing_trace.txt`, `orchjax_intersurf_main_trace.txt`,
  `orchjax_grid_trace.txt`, `orchjax_slowproc_soilt_trace.txt`,
  `orchjax_slowproc_read_annual_trace.txt`, and
  `orchjax_slowproc_read_data_trace.txt`
- source-backed grid truth for `resolution`, `neighbours`, and final grid
  `area`: the paper one-point path uses `haversine_singlepointploy`, while
  regular lon/lat multi-point domains use
  `haversine_reglatlontoploy`/`haversine_laloseglen`/`haversine_laloarea`
  followed by `grid_scatter`
- source-backed slowproc static truth for `soilclass`, `njsc`, clay/sand/silt
  fractions, bulk density, soil pH, poor-soils fraction, `salinity`, and
  `tide_height`
- trace-free first-day scaffold from forcing through SECHIBA/STOMATE daily
  carbon, OK_LEAK, modelout diagnostics, daily reset, day-end SLOWPROC
  surface update, and day-end state packet
- trace-free later-day scaffold from a model-produced day-end packet through
  the next 48 SECHIBA half-hour steps, STOMATE daily accumulation fold,
  STOMATE daily carbon, OK_LEAK, modelout diagnostics, daily reset, and the
  next day-end state packet without reusing the initial restart as prognostic
  state
- strict multi-day scaffold that chains each later day from the previous
  model-produced day-end state; this is regression-closed through day 2 for
  the current restart-backed scaffold, but it must not be treated as cold-start
  parity until the first-day initialization path is separated from local
  `driver_start.nc`/`sechiba_start.nc`/`stomate_start.nc` readers

The local code also includes:

- source-backed raw, yearly-cached raw, and cached/interpolated model-step
  forcing readers for multiple selected land points, preserving
  `readdim2.f90::forcing_landind` land order and returning each forcing
  variable as an `npts x 1` column; the model-step path keeps the Fortran
  ordering of full zoom-grid `solarang` shortwave redistribution followed by
  land-vector compression
- regular lon/lat multi-point `grid_stuff` geometry with sorted neighbours,
  segment-derived `resolution`, and final grid `area`; a real CRUNCEP 3x3
  domain micro-case now closes `domain -> model-step forcing -> intersurf
  payload -> local slowproc static soil/salinity/tide` without server trace
- geometry and soil-overlap payload fields for `corners`, `seglength`,
  `soilclass_sub_index`, and `soilclass_sub_area`, populated either from
  source-backed local readers or explicit fixed-format trace adapters
- explicit-geometry static helpers that require caller-supplied
  `lalo/resolution` or `sub_index/sub_area`
- trace schema constants for required CSV columns
- trace validators/readers for caller-supplied CSV, mapping rows, or the
  audited 2026-06-23 fixed-format server rows
- explicit trace-to-domain/static adapters that package trace truth without
  deriving geometry

## Source-Backed Static Closure

These previous missing fields are now read from local source inputs and
validated against the fixed-format server trace:

- `resolution`, `neighbours`, and final grid `area` from
  `data/forcing/cruncep_twodeg_1961.nc` lon/lat plus the active
  single-point and regular lon/lat multi-point paths in
  `src_global/haversine.f90` and `src_global/grid.f90`
- `soilclass`, `njsc`, `clay_frac`, `sand_frac`, `silt_frac`, `bulk_dens`,
  `soil_ph`, and `poor_soils` from `data/INPUTDIR_ZZ/test.nc` through
  `slowproc_soilt` USDA `aggregate_2d` overlap semantics
- `salinity` and `salinity_bbox` from `data/INPUTDIR_ZZ/salinity_05.nc`
  through `slowproc_read_annual` bbox averaging
- `tide_height` and `tide_bbox` from `data/INPUTDIR_ZZ/tide_05_584.nc`
  through `slowproc_read_data` bbox averaging

The bundle no longer needs `fixed_format_trace_dir` for those fields.
`fixed_format_trace_dir` remains supported as an explicit trace adapter and as
regression validation truth.

The same trace package also verifies, without changing the no-trace default:

- first forcing/domain values from `orchjax_driver_forcing_trace.txt`
- first `intersurf_main_2d` boundary row from
  `orchjax_intersurf_main_trace.txt`

## Still Blocked

These fields remain blocked because exact usable Fortran source-backed inputs
are not yet present in a boundary payload shape:

- full final `StomateLpj` tail-state LAI when future optional branches beyond
  the currently wired turnover/final-setlai boundary are enabled
- cold-start versus restart-backed first-day separation: the paper
  `configs/orchidee_man_250919.yaml` minimal case and the copied server traces
  use `RESTART_FILEIN=NONE`, `SECHIBA_restart_in=NONE`, and
  `STOMATE_RESTART_FILEIN=NONE`, while the current day scaffold still consumes
  local `reference/case_001_071` start restart files as first-step dynamic
  state. Cold-start ENERBIL/HYDROL/THERMOSOIL parity must be audited on the
  cold-start path, not by comparing restart-backed state to `RESTART=NONE`
  traces. `paper_1961_driver_cold_start_first_step_coverage` now exposes the
  audited cold-start subset directly: driver/domain/run scalars, imposed
  vegetation cover, DIFFUCO `biomass=0` readstart fallback, ENERBIL
  `ENERBIL_TSURF`/`qair`/zero-evaporation fallbacks, THERMOSOIL
  `THERMOSOIL_TPRO` constant-profile fallback, no-LCC/fire-off/dynamic-peat-off
  SLOWPROC inputs, static `fc_grazing/humcste_use`, and thermal-entry
  `tdeep/hsdeep/heat_Zimov`. It deliberately remains not ready for a
  cold-start first step until the remaining non-restart HYDROL/THERMOSOIL and
  STOMATE readstart fallback state is implemented, especially the
  `thermosoil_coef` initialization path that must produce
  `cgrnd/dgrnd/lambda_snow/soilcap*`/`soilflx*` without `sechiba_start.nc`.

No history diagnostic is used as a replacement for `grid_stuff` truth. In
particular, `RESOLUTION_X/Y` from `stomate_history_1961.nc` is not sufficient:
it lacks `neighbours`, `corners`, `seglength`, and static overlap/bbox
intermediates.

## Trace Files Needed

The next Fortran run or trace normalization pass should convert the
fixed-format rows into the schema groups from
`docs/source_audits/driver_static_trace_instrumentation_plan.md`:

- `grid_geometry_after_grid_stuff` normalization for `corners` and `seglength`
  if those become payload fields
- `soil_aggregation` normalization into full `sub_index/sub_area` matrix
  payload fields if downstream tests need overlap truth directly
- `intersurf_first_step_boundary` with `resolution_x/y` and `zlev` columns if
  it should replace the current partial fixed-format boundary trace
- optional STOMATE tail-state traces when enabling currently branch-gated
  `StomateLpj` sections such as land-cover change or peat-cover updates
- later-day OK_LEAK/modelout/end-state parity traces if generalized local
  boundary wiring exposes a mismatch not diagnosable from the first-day trace

The required columns are encoded in `jax_orchidee.driver.trace_schema`.

## Tests To Add After Trace Exists

After the remaining trace payload fields are added or normalized:

- validate every trace CSV with `read_trace_csv`
- compare explicit `model_bbox_from_lalo_resolution` results to traced bbox
  limits
- add payload fields and tests for `corners`, `seglength`,
  `soilclass_sub_index`, and `soilclass_sub_area`

## Stop Condition

Local Driver/Grid work is closed for the source- and trace-supported boundary
fields above. The imposed cold-start static vegetation field (`veget`) is now
derived through the audited `slowproc_init`/`slowproc_veget` source path using
materialized `EXT_COEFF_VEGETFRAC` defaults; it is not filled from history
diagnostics or guessed geometry. Unsupported vegetation branches such as
`READ_LAI=y` still require separate source-driven coverage before use.
