# DIFFUCO PFT14 Inundation Input Chain Audit

Scope: pre-integration audit for the PFT14 `control_inudate` input chain.
This document records only sources found in this repository. It does not use
`after_diffuco_main` boundary outputs as internal mangrove-control truth.

## Fortran Contract

`src_sechiba/sechiba.f90::sechiba_main` builds `rprof` immediately before
DIFFUCO:

- lines 984-990: `rprof(:,jv) = 1./humcste_use(:,jv)`.
- lines 997-1005: calls `diffuco_main(... salinity, rprof, tide_height,
  biomass, ...)`.

`src_sechiba/diffuco.f90::diffuco_main` consumes the four audited inputs:

- lines 383-387 declare `rprof`, `tide_height`, and `biomass` as inputs.
- lines 470-474 construct `z_soil(0:nslm)` from `diaglev(1:nslm)`.
- lines 477-522 use `z_soil`, `rprof`, and `tide_height` to compute root
  fractions and inundated root fraction for every tide slot.
- lines 525-576 use `biomass` AGR pools and `tide_height` to compute root
  ventilation and anoxia.
- lines 578-584 compute `control_inudate` from PFT14 anoxia over all
  `itimetide` slots.

`diaglev` provenance:

- `src_parameters/control.f90::control_initialize` lines 584-603 allocates and
  fills `diaglev`.
- `src_parameters/vertical_soil_var.f90` lines 57-61 declare the shared
  `diaglev` state.
- In the active CWRR branch, `control_initialize` lines 593-595 set
  `diaglev=znt(1:nslm)`. The `znt` thermal-node vector is built in
  `src_parameters/vertical_soil.f90::vertical_soil_init`: lines 175-263 read
  vertical grid parameters, lines 268-281 build hydrology node candidates,
  lines 331-363 select `nslm`, and lines 374-444 build `znt`.

`humcste_use` provenance:

- `src_sechiba/hydrol.f90::hydrol_init` lines 4093-4107 assigns
  `humcste_use(ji,jv)=humcste(jv)`.
- `src_parameters/pft_parameters.f90` lines 265-273 initialize `humcste` from
  reference MTC values; lines 3604-3610 expose the `HYDROL_HUMCSTE` config
  key.

`biomass` provenance at the DIFFUCO call boundary:

- `src_sechiba/sechiba.f90::sechiba_init` lines 2383-2388 allocates
  `biomass(kjpindex,nvm,nparts,nelements)` and initializes it to zero.
- `src_sechiba/sechiba.f90::sechiba_initialize` lines 602-617 calls
  `slowproc_initialize` before any time-step `sechiba_main` call.
- `src_sechiba/slowproc.f90::slowproc_initialize` lines 257-293 calls
  `slowproc_init` and then `stomate_initialize`, passing the same `biomass`
  array.
- `src_stomate/stomate.f90::stomate_initialize` lines 1309-1368 declares
  `biomass` as `INTENT(out)` and passes it to `readstart`.
- `src_stomate/stomate_io.f90::readstart` lines 919-922 reads restart variable
  `biomass`; if the read returns only `val_exp`, it sets the whole array to
  zero.
- `src_sechiba/sechiba.f90::sechiba_main` lines 984-1005 builds `rprof` and
  calls `diffuco_main` before the same-step `slowproc_main`/`stomate_main`
  call at lines 1184-1216.
- `src_stomate/stomate_io.f90` lines 2631-2632 writes restart `biomass` at
  finalize. That saved output can become a next-run start file only if the run
  protocol moves it to `STOMATE_RESTART_FILEIN`.

## Current Local Coverage

Covered:

- `tide_height`: full `tide_height(kjpindex,itimetide)` is available through
  the local fixed-format trace adapter from
  `outputs/server_1961_bridge_trace_20260624/traces/orchjax_slowproc_read_data_trace.txt`.
  The Fortran read path is `src_sechiba/slowproc.f90` lines 2531-2534 and
  `slowproc_read_data` lines 6029-6042. Unit tests confirm shape `(1,584)`.
- `diaglev -> z_soil`: locally constructible for the paper-case CWRR branch.
  `outputs/server_1961_trace_full_20260623/run/used_run.def` records
  `DEPTH_MAX_T=38`, `DEPTH_MAX_H=2`, `DEPTH_TOPTHICK=9.77517107e-4`,
  `DEPTH_CSTTHICK=2`, `DEPTH_GEOM=2`, and `RATIO_GEOM_BELOW=1.05`. Applying
  `vertical_soil_init` and then `diaglev=znt(1:nslm)` gives 11 thermal-node
  diagnostic depths. `z_soil_from_diaglev` then prepends zero exactly as
  `diffuco_main` lines 470-474 do. Important: these are not the HYDROL
  `znh` node depths used by HYDROL table tests.
- `humcste_use -> rprof`: locally constructible for the first-step paper case.
  `outputs/server_1961_trace_full_20260623/run/used_run.def` records
  `PFT_TO_MTC__00014=2` and `HYDROL_HUMCSTE__00014=0.8`. This agrees with
  `constantes_mtc.f90` lines 209-213 for `zmaxh=2` and
  `pft_parameters.f90` lines 265-273. Since the near-surface permafrost
  rewrite in `hydrol.f90` lines 4107-4110 is commented out, `hydrol_init`
  lines 4093-4107 broadcast `humcste` to `humcste_use`; `sechiba_main` lines
  984-990 then give PFT14 `rprof=1/0.8=1.25`.

Boundary-qualified coverage:

- `biomass` for first-step only, under explicit source conditions:
  - cold-start run with `STOMATE_RESTART_FILEIN=NONE`: `readstart` lines
    919-922 falls back to an all-zero `biomass` array before the first
    `sechiba_main` call.
  - restart run whose exact `STOMATE_RESTART_FILEIN` is locally available:
    the normalized `biomass` read from that start file is the initialization
    state passed to first-step DIFFUCO, because same-step STOMATE has not run
    yet.

Missing or not sufficient:

- `biomass` beyond the first-step initialization boundary: current local bridge
  traces do not record `biomass(kjpindex,nvm,nparts,nelements)` immediately
  before `diffuco_main`. Same-step `stomate_main` output is later in the
  `sechiba_main` order and cannot supply the DIFFUCO call just made.
- `stomate_restart.nc` as an output file: readable but not by itself
  call-boundary truth for the current run. It is end-of-run saved state unless
  the run script has moved it to `stomate_start.nc` and set
  `STOMATE_RESTART_FILEIN` for the next run.
- `after_diffuco_main tide_height_1`: useful for boundary context only. It is
  not enough for `control_inudate`, which averages anoxia across every
  `itimetide` slot.

## Implemented Helpers

`jax_orchidee/sechiba/diffuco.py` now includes:

- `z_soil_from_diaglev(diaglev)`: direct Fortran construction only.
- `cwrr_diaglev_from_vertical_soil_params(...)`: direct CWRR
  `vertical_soil_init`/`control_initialize` construction from explicit local
  runtime parameters.
- `z_soil_from_cwrr_vertical_soil_params(...)`: CWRR `diaglev` plus DIFFUCO
  zero-prepend.
- `humcste_from_pft_to_mtc(...)`: source-table `humcste` construction, with an
  optional explicit `HYDROL_HUMCSTE` vector override.
- `humcste_use_from_humcste(...)`: HYDROL broadcast to
  `humcste_use(kjpindex,nvm)`.
- `rprof_from_humcste_use(humcste_use)`: direct Fortran construction only.
- `diffuco_control_inundation_input_coverage(...)`: read-only coverage report
  that marks current trace-covered and missing inputs without fabricating
  defaults.

Current result for the copied bridge trace package:

- covered: `tide_height`.
- constructible from local source/runtime truth: `diaglev -> z_soil` and
  `humcste_use -> rprof` for the first-step paper case.
- first-step `biomass` can now be marked covered only for an explicit cold
  start zero fallback or an exact `STOMATE_RESTART_FILEIN` start file.
- still missing for full multi-step DIFFUCO small-kernel integration: traced
  DIFFUCO call-boundary `biomass` for each step.
