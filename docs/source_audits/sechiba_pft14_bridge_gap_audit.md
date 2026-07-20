# SECHIBA PFT14 Bridge Gap Audit

Scope: minimal bridge from the locally closed driver/slowproc/HYDROL slices to
STOMATE/modelout for the paper-case single point, PFT14, first year. This is a
source audit and scaffold contract only; it does not implement missing SECHIBA
process formulas.

## Paper-Case Call Order

Active Fortran path:

1. `src_sechiba/intersurf.f90::intersurf_main_2d` calls
   `sechiba_main`: lines 458-761 and 640-648.
2. `src_sechiba/sechiba.f90::sechiba_main` calls, in order:
   `diffuco_main` lines 997-1005, `enerbil_main` lines 1013-1019,
   CWRR `hydrol_main` lines 1049-1072, `condveg_main` lines 1085-1091,
   `thermosoil_main` lines 1109-1118, and `slowproc_main` lines 1184-1216.
3. `src_sechiba/slowproc.f90::slowproc_main` calls `stomate_main` when
   `ok_stomate`: lines 936-973.

Implication: `diffuco` and `enerbil` are not optional pre-STOMATE processes.
`diffuco` produces same-step `gpp`; `enerbil` produces same-step evaporation,
transpiration, potential evaporation, and surface temperature that HYDROL and
STOMATE daily forcing depend on. HYDROL then produces water stress, soil
humidity, water flux, and canopy/ground water fields consumed by STOMATE and
by the next SECHIBA timestep.

## Minimum First-Year Variables

For PFT14 modelout, the hard minimum is not only final STOMATE biomass history.
The year run must reproduce the daily path that creates those fields.

### DIFFUCO -> STOMATE

- `gpp(npts,nvm)`: required for daily `gpp_daily` and modelout GPP/NPP path.
  Provenance: `diffuco.f90::diffuco_main` output declaration lines 389-405;
  `diffuco_trans_co2` call lines 665-671; GPP assignment lines 2890-2897;
  `stomate.f90::stomate_main` daily accumulation lines 3198-3208;
  `stomate_lpj.f90` history writes lines 1678 and 2203.
- `gsmean`, `rveget`, `rstruct`, `cimean`: minimum diagnostic/trace fields to
  audit CO2/transpiration coupling. Provenance: `diffuco.f90` lines 391-405
  and `diffuco_trans_co2` interface lines 2018-2088.

### ENERBIL -> HYDROL/SLOWPROC

- `transpir`, `transpot`, `vevapnu`, `vevapnu_pft`, `vevapwet`, `vevapsno`,
  `vevapflo`: required by HYDROL water balance and stress. Provenance:
  `enerbil.f90::enerbil_main` outputs lines 447-455; `enerbil_evapveg` call
  lines 541-545; `sechiba.f90` passes them to `hydrol_main` lines 1049-1055.
- `temp_sol`, `temp_sol_new`, `temp_sol_pft`, `qsurf`, `t2mdiag`,
  `evapot_corr`: required by HYDROL snow/soil update, slowproc/STOMATE daily
  forcing, and next `diffuco`. Provenance: `enerbil.f90` lines 456-471 and
  507-545; `sechiba.f90` lines 1049-1055 and 1184-1198.

### HYDROL -> STOMATE/SOILCARBON

- `humrel(npts,nvm)`, `vegstress(npts,nvm)`, `shumdiag(npts,nslm)`,
  `litterhumdiag(npts)`: required for STOMATE daily moisture, allocation
  stress, and soil humidity state. Provenance: `hydrol.f90::hydrol_main`
  declarations lines 1041-1063; `sechiba.f90` passes these to slowproc lines
  1184-1188; `stomate.f90` accumulates `humrel`, `litterhumdiag`,
  `stempdiag`, `shumdiag`, and `gpp_d` lines 3198-3208.
- `soil_mc(npts,nslm,nstm)`, `wat_flux(npts,nslm,nstm)`,
  `runoff_per_soil(npts,nstm)`, `runoff2peat(npts,nstm)`,
  `drainage_per_soil(npts,nstm)`: required by MICT-leak soil carbon/DOC.
  Provenance: `hydrol.f90` output declarations lines 1030-1038 and
  `hydrol_soil` call lines 1278-1296; `stomate_soilcarbon.f90` consumes them
  lines 651-705, uses `wat_flux` for DOC flux lines 1869-1888, runoff for DOC
  export lines 2146-2182, and drainage lines 2205-2210.
- `precip2canopy`, `precip2ground`, `canopy2ground`: required by MICT-leak
  wet deposition/canopy DOC transfer. Provenance: `hydrol.f90` declarations
  lines 1030-1035 and `hydrol_canop` call lines 1232-1235;
  `stomate_soilcarbon.f90` consumes them lines 699-704 and computes
  `DOC_canopy2ground` lines 1497-1512.

### HYDROL -> THERMOSOIL -> STOMATE

- `mc_layh`, `mcl_layh`, `soilmoist`, `mc_layh_s`, `mcl_layh_s`: needed
  before thermosoil can update `stempdiag`, which feeds STOMATE maintenance
  respiration and daily soil temperature. Provenance: `hydrol.f90` lines
  1085-1089; `sechiba.f90` maps per-tile moisture to PFT arrays lines
  1093-1102 and calls `thermosoil_main` lines 1109-1118.

### HYDROL -> Next SECHIBA Timestep

- `qsintveg`, snow state (`snow`, `snow_nobio`, `snowdz`, `snowrho`),
  `evap_bare_lim`, `drysoil_frac`, `k_litt`: needed for next `diffuco`,
  `condveg`, and HYDROL. Provenance: `hydrol.f90` INTENT/state declarations
  lines 1057-1083; `hydrol_canop` lines 1232-1235; `condveg_main` call
  lines 1085-1091; `diffuco.f90` consumes `qsintveg`, `snow`, and
  `evap_bare_lim` lines 345-368, 647-710.

## Is Current server_1961 Trace Enough?

No, not for the bridge.

Covered:

- Driver/intersurf/static boundary and salinity/tide static reads are covered
  by `outputs/server_1961_trace_full_20260623` as documented in
  `docs/porting/driver_local_closure.md`.
- HYDROL internal solve/post/update slices are covered for selected fixed
  records by `orchjax_hydrol_main_trace.txt`, `orchjax_hydrol_post_trace.txt`,
  and `orchjax_hydrol_update_trace.txt`.
- STOMATE `gpp_daily` accumulation and allocation/NPP partial traces are
  covered by `orchjax_stomate_daily_trace.txt`,
  `orchjax_stomate_lpj_trace.txt`, and `orchjax_stomate_npp_trace.txt`.

Missing as bridge truth:

- A named `after_diffuco` record for `gpp`, `gsmean`, `rveget`, `rstruct`,
  `cimean`, `vbeta*`, `q_cdrag`, and mangrove salinity/inundation controls.
  STOMATE daily trace contains GPP accumulation after the boundary, but it is
  not a complete `diffuco_main` before/after contract.
- A named `after_enerbil` record for `transpir`, `transpot`, `vevapnu`,
  `vevapnu_pft`, `vevapwet`, `vevapsno`, `vevapflo`, `temp_sol`,
  `temp_sol_new`, `qsurf`, `t2mdiag`, `evapot`, and `evapot_corr`.
- A named `after_hydrol_main` or `before_slowproc_main` record for
  `humrel`, `vegstress`, `shumdiag`, `litterhumdiag`, `soil_mc`,
  `wat_flux`, `runoff_per_soil`, `runoff2peat`, `drainage_per_soil`,
  `precip2canopy`, `precip2ground`, `canopy2ground`, `mc_layh`,
  `mcl_layh`, `soilmoist`, `qsintveg`, snow state, `evap_bare_lim`,
  `drysoil_frac`, and `k_litt`.
- A named `after_thermosoil` record for `stempdiag` before `slowproc_main`.

Minimum extra trace points, when a future server run is explicitly requested:

1. `after_diffuco_main`: one row for land point 1, PFT14 where applicable,
   fields listed above.
2. `after_enerbil_main`: one row for land point 1 plus PFT14 arrays where
   applicable.
3. `after_hydrol_main_before_condveg`: HYDROL output/state fields listed
   above, including tile 4/PFT14-relevant soil-tile arrays.
4. `after_thermosoil_before_slowproc`: `stempdiag` and PFT-mean thermal fields
   used by STOMATE.
5. `before_stomate_main`: final slowproc boundary bundle, to confirm the
   exact values passed into STOMATE after all SECHIBA updates.

## Scaffold Added

`jax_orchidee/sechiba/bridge.py` now defines a field contract:

- bridge groups:
  `hydrol_to_stomate`, `hydrol_to_thermosoil`, `hydrol_next_sechiba`,
  `diffuco_to_stomate`, and `enerbil_to_hydrol_slowproc`.
- each field carries shape, producer, consumers, required use, Fortran
  provenance, and current `server_1961_trace` status.
- `validate_bridge_payload` checks declared fields only and never derives
  missing process values.

Tests: `tests/unit/test_sechiba_bridge.py`.

## Next Development Order

1. Implement or trace-validate `diffuco_main` minimal PFT14 path:
   photosynthesis/GPP, mangrove salinity and inundation control, and
   transpiration resistance outputs. Blocker: no complete `after_diffuco`
   trace.
2. Implement or trace-validate `enerbil_main` minimal path:
   surface temperature, potential evaporation, evaporation components, and
   transpiration. Blocker: no `after_enerbil` trace.
3. Extend HYDROL local bridge after full HYDROL solve closes:
   final `humrel`, `shumdiag`, `vegstress`, MICT-leak water fields, canopy
   water split, and next-step state. Blocker: current HYDROL trace is internal
   solve/post state, not the exported `hydrol_main` boundary.
4. Implement minimal `thermosoil_main` bridge or require traced `stempdiag`:
   STOMATE maintenance respiration cannot be fully source-exact without the
   soil temperature profile after HYDROL.
5. Only then wire STOMATE daily carbon adapter toward modelout for the first
   year. Existing STOMATE carbon kernels can consume explicit fields, but
   allocation/NPP full process parity remains blocked as documented in
   `docs/porting/stomate_local_closure.md`.

## Blocker Table

| Bridge item | Status | Blocker |
| --- | --- | --- |
| `gpp` from DIFFUCO | partial trace via STOMATE daily | Need `after_diffuco_main` for process parity. |
| ENERBIL evap/transp/temp fields | not traced as boundary | Need `after_enerbil_main`. |
| HYDROL moisture stress fields | not traced as boundary | Need `after_hydrol_main` or `before_slowproc_main`. |
| HYDROL MICT-leak water fields | not traced as boundary | Need `soil_mc`, `wat_flux`, runoff/drainage, canopy water split. |
| HYDROL -> thermosoil moisture | partial internal HYDROL trace | Need complete exported moisture arrays and `after_thermosoil` stempdiag. |
| STOMATE allocation/NPP | partial trace | Existing trace gives `f_alloc`/after-NPP samples, but not full before/after process contract. |
