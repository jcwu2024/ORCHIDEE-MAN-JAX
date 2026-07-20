# ENERBIL PFT14 Minimal Closure Audit

Scope: source audit plus minimal JAX boundary scaffold for the paper-case
single land point, active PFT14 path. This document does not implement
`enerbil_main`; it identifies the minimum fields and trace points needed to do
that without guesses.

## Source Anchors

- `src_sechiba/sechiba.f90::sechiba_main` lines 1013-1019 call
  `enerbil_main` immediately after `diffuco_main`, passing `swnet`,
  DIFFUCO beta/drag fields, `emis`, `soilflx*`, and `soilcap*` into the
  ENERBIL boundary.
- `src_sechiba/sechiba.f90::sechiba_main` lines 1109-1118 call
  `thermosoil_main` after HYDROL moisture mapping, passing
  `temp_sol_new*`, `soilcap*`, and `soilflx*`.
- `src_sechiba/enerbil.f90::enerbil_main` lines 390-598 is the energy-balance
  boundary.
- `src_sechiba/enerbil.f90::enerbil_main` lines 433-438 declare
  `soilflx_pft`, `soilflx`, `soilcap`, `soilcap_pft`, `q_cdrag`, and
  `q_cdrag_pft` as `INTENT(in)`.
- `src_sechiba/enerbil.f90::enerbil_main` lines 485-545 defines the internal
  order: `enerbil_begin`, `enerbil_surftemp`, `enerbil_pottemp`,
  `enerbil_flux`, `enerbil_evapveg`, then `enerbil_t2mdiag`.
- `src_sechiba/enerbil.f90::enerbil_begin` lines 735-873 computes old-state
  surface static energy, old saturated humidity, humidity derivative, absorbed
  longwave radiation, and old-state net radiation.
- `src_sechiba/qsat_moisture.f90::qsatcalc` lines 79-179,
  `dev_qsatcalc` lines 311-408, and `qsfrict_init` lines 547-589 define the
  exact table interpolation used by `enerbil_begin`.
- `src_sechiba/hydrol.f90::hydrol_main` lines 938-1664 consumes the ENERBIL
  evaporation, transpiration, temperature, and potential evaporation fields.
- `src_sechiba/thermosoil.f90::thermosoil_main` lines 788-1054 consumes
  `temp_sol_new` and `temp_sol_new_pft` after HYDROL moisture mapping.
- `src_sechiba/thermosoil.f90::thermosoil_main` lines 995-1010 call
  `thermosoil_coef` after ENERBIL; comments at lines 997-1000 state that
  these coefficients are for the next timestep. `thermosoil_coef` declares
  `soilcap`, `soilcap_pft`, `soilflx`, and `soilflx_pft` as outputs at
  lines 1423-1427.
- `src_sechiba/slowproc.f90::slowproc_main` lines 343-377 declares the daily
  forcing boundary; lines 973-1009 call `stomate_main`.

## Position In SECHIBA

`enerbil_main` is not a diagnostic side path. It is the second required
same-step physical process after `diffuco_main`.

1. `diffuco_main` produces resistance/aerodynamic fields and GPP:
   `sechiba.f90` lines 997-1005.
2. `enerbil_main` consumes those fields and computes energy, evaporation,
   transpiration, potential evaporation, surface humidity, and diagnostic
   temperature: `sechiba.f90` lines 1013-1019.
3. CWRR `hydrol_main` immediately consumes ENERBIL water/energy fields:
   `sechiba.f90` lines 1049-1072.
4. `thermosoil_main` consumes `temp_sol_new`/`temp_sol_new_pft` after HYDROL
   moisture fields are mapped to PFT arrays: `sechiba.f90` lines 1093-1118.
5. `slowproc_main` receives `t2mdiag`, `temp_sol`, `stempdiag`,
   `evapot_corr`, and HYDROL/STOMATE water fields: `sechiba.f90` lines
   1184-1216; `slowproc.f90` lines 343-377 and 973-1009.

## Minimum ENERBIL Inputs

For the PFT14 single-point path, the minimum input bundle is exactly the
`enerbil_main` interface fields needed to produce downstream-visible state:

- Forcing and atmospheric fields: `zlev`, `lwdown`, `swnet`, `epot_air`,
  `temp_air`, `u`, `v`, `qair`, `pb`, `rau`, `precip_rain`.
  Provenance: `enerbil.f90::enerbil_main` lines 410-423 and 444.
- Coupler linearization fields: `petAcoef`, `petBcoef`, `peqAcoef`,
  `peqBcoef`. Provenance: lines 417-421.
- DIFFUCO resistance/aerodynamic fields: `vbeta`, `vbeta_pft`, `valpha`,
  `vbeta1`, `vbeta2`, `vbeta3`, `vbeta3pot`, `vbeta4`, `vbeta4_pft`,
  `vbeta5`, `q_cdrag`, `q_cdrag_pft`, `humrel`. Provenance: lines 424-431,
  437-443; DIFFUCO produces these in `sechiba.f90` lines 997-1005.
- Surface/thermal state from previous SECHIBA timestep: `emis`, `soilflx`,
  `soilflx_pft`, `soilcap`, `soilcap_pft`, `temp_sol`, `temp_sol_pft`,
  `qsurf`, `tsol_rad`, `pgflux`. Provenance: `enerbil.f90` lines 432-438
  and 463-472.
- Vegetation and snow context: `veget_max`, `snowdz`. Provenance:
  `enerbil.f90` lines 426 and 445.

These are input requirements, not a permission to infer missing values.

## `enerbil_begin` Local Closure

`enerbil_begin` is the first internal ENERBIL step. It is called by
`enerbil_main` at `enerbil.f90` lines 485-489 and feeds
`enerbil_surftemp` at lines 507-510. The following pieces are now closed by
source-backed local kernels in `jax_orchidee/sechiba/enerbil.py`:

- `psold = temp_sol * cp_air`: `enerbil_begin` line 780.
- `psold_pft(:,jv) = temp_sol_pft(:,jv) * cp_air` for one-based `jv=2..nvm`
  when `ok_LAIdev(jv)` is true, otherwise `psold_pft(:,jv)=psold`:
  lines 781-787. The Fortran loop does not assign one-based PFT1 in this
  block; the JAX helper leaves column 0 as `nan` for unassigned PFT-only
  diagnostics rather than inventing a value.
- `qsol_sat` and `qsol_sat_pft`: `enerbil_begin` lines 790-796 calling
  `qsat_moisture.f90::qsatcalc` lines 79-179. `qsatcalc` uses the
  `qsfrict` table initialized by modified Goff-Gratch equations at
  `qsat_moisture.f90` lines 547-589; molecular-weight constants are in
  `constantes_var.f90` lines 382-383.
- `pdqsold` and `pdqsold_pft`: `enerbil_begin` lines 817-832 calling
  `qsat_moisture.f90::dev_qsatcalc` lines 311-408, using the same table.
- `lwabs = emis * lwdown`: `enerbil_begin` line 861.
- `netrad = lwdown + swnet - (emis*c_stefan*temp_sol**4 +
  (1-emis)*lwdown)`: `enerbil_begin` line 867.
- `netrad_pft(:,jv)` for one-based `jv=2..nvm`:
  `enerbil_begin` lines 868-870.

These kernels are still upstream-limited. They need exact `temp_sol`,
`temp_sol_pft`, `lwdown`, `swnet`, `pb`, `emis`, and `ok_LAIdev`.
The existing driver/intersurf trace covers `lwdown`, but `swdown` is not
`swnet`. No shortwave-net approximation is made.

The surface-state input chain is audited in
`docs/source_audits/enerbil_surface_state_input_chain.md`. Current local
coverage can use `after_diffuco_main` for pre-ENERBIL `temp_sol` and
`temp_sol_pft`, because that trace is emitted after `diffuco_main` and before
`enerbil_main`. It does not use `after_enerbil_main` to close `soilcap*`,
`soilflx*`, or any other same-call pre-ENERBIL input.

## `enerbil_surftemp` Local Closure

`enerbil_surftemp` is now locally closed as a source-backed linearized solve in
`jax_orchidee/sechiba/enerbil.py::enerbil_surftemp_explicit_solve`.
This is not a tuned approximation or iterative substitute: the Fortran source
computes old fluxes and sensitivities, then evaluates one explicit `dtheta`
formula.

Closed source pieces:

- wind speed and transfer resistances: `enerbil_surftemp` lines 999-1012.
- old sensible, sublimation latent, and evaporation latent fluxes:
  lines 1027-1068.
- net-radiation, sensible, and latent sensitivity terms: lines 1080-1135.
- `sum_old`, `sum_sns`, and `dtheta`/`dtheta_pft`: lines 1144-1179.
- `psnew`, `qsol_sat_new`, `temp_sol_new` and PFT equivalents:
  lines 1188-1212.
- `epot_air_new`, `fevap`, and `qair_new`: lines 1220-1239.

The local solve still does not manufacture upstream inputs. It requires exact
inputs from `enerbil_begin`, DIFFUCO, surface thermal state, and coupler
linearization:

- `psold*`, `qsol_sat*`, `pdqsold*`, `netrad*` from `enerbil_begin`.
- `emis`, `swnet`, `epot_air`, `petAcoef`, `petBcoef`, `peqAcoef`,
  `peqBcoef`, `soilflx*`, `rau`, `soilcap*`.
- DIFFUCO/aerodynamic fields `q_cdrag*`, `vbeta*`, `valpha`, `vbeta1`,
  `vbeta5`, and `veget_max`.

Therefore the formula is closed, but first-step pipeline coverage remains
upstream-limited until those inputs are traced or source-constructed exactly.
`swdown` alone is still not `swnet`, and after-ENERBIL outputs are still not
pre-call inputs.

The offline driver formula for `swnet` is exact but conditional:
`orchideedriver.f90` lines 598 and 659 compute
`swnet(:) = (1.-(albedo(:,1)+albedo(:,2))/2.)*swdown(:)`. The JAX helper
`driver_swnet_from_swdown_albedo` implements this formula, but coverage stays
missing when only `swdown` is available and the same-step two-band `albedo` is
not traced or otherwise audited.

The `albedo -> swnet` source chain is audited in
`docs/source_audits/enerbil_albedo_swnet_input_chain.md`. For this local
reference case, `run.def` reads `driver_start.nc`, which contains exact
`albedo_vis/albedo_nir`. `dim2_driver.f90` lines 1139-1164 read those two
driver restart bands after `intersurf_initialize_2d` and recompute
`for_swnet` before the first `intersurf_main_2d`/`sechiba_main` call.
Therefore the archived local first-step `swnet` can be closed from
driver/intersurf `swdown` plus `driver_start.nc` two-band albedo. `sechiba_start.nc`
contains `soilalbedo_bg`, not final driver albedo, and same-step
`condveg_main` remains after ENERBIL.

The CONDVEG initialization formula for `emis` is exact for the current local
first step. `condveg.f90::condveg_initialize` lines 260-269 assigns
`emis(:)=emis_scal` when `impaze` is true, otherwise assigns
`emis_scal=un` and then `emis(:)=emis_scal`. `constantes_var.f90` lines
670-707 initializes `impaze=.FALSE.` and `emis_scal=1.0`; `constantes.f90`
lines 622-678 reads `IMPOSE_AZE` and reads `CONDVEG_EMIS` only inside the
true branch. The expanded local run configuration records
`IMPOSE_AZE = FALSE` at
`outputs/server_1961_trace_full_20260623/run/used_run.def` line 2448, so the
first-step ENERBIL input `emis` is exactly `1.0` from initialization. The
same-step `condveg_main` call remains after ENERBIL (`sechiba.f90` lines
1084-1090) and is not a current-step source.

The atmospheric/coupling input chain is audited separately in
`docs/source_audits/enerbil_atmospheric_input_chain.md`. `rau` is source-closed
by `sechiba.f90::sechiba_var_init` lines 3069-3092 from exact `pb` and
`temp_air`. `epot_air` and PET/PEQ coefficients are source-closed only for an
explicitly audited non-WATCHOUT, non-relaxation driver branch
(`dim2_driver.f90` lines 914-920 and 995-1003), or read from WATCHOUT forcing
fields (`readdim2.f90` lines 1833-1857). The current local driver/intersurf
traces do not contain WATCHOUT `Eair`/PET/PEQ columns. `zlev` is an
`enerbil_surftemp` interface input but the current local solve body does not
consume it.

## Explicit Local Step

`jax_orchidee/sechiba/enerbil.py::enerbil_explicit_local_step` now wires the
closed local kernels in the same source order as `enerbil_main`:

1. `enerbil_begin`: `enerbil.f90` lines 485-489.
2. `enerbil_surftemp`: lines 507-510.
3. `enerbil_pottemp`: lines 523-526, when `q_sol_pot`/`temp_sol_pot` inputs
   are supplied to the adapter.
4. `enerbil_flux`: lines 534-538, including the Milly `evapot_corr`
   correction at lines 1592-1640.
5. `enerbil_evapveg`: lines 541-545.
6. `enerbil_t2mdiag`: lines 547-551, when `temp_air` is supplied.

This adapter is deliberately explicit-input only. It does not read traces,
default missing fields, infer `swnet` from `swdown`, or reverse-use
after-ENERBIL outputs as pre-call inputs. Potential-temperature diagnostics
remain optional because the active trace package does not include
`q_sol_pot`/`temp_sol_pot`.

## Active Trace Closure Update

An active 1961 server trace package now closes the same-call ENERBIL input and
output boundary for the first active PFT14 row:

- Local package:
  `outputs/server_1961_enerbil_active_trace_20260625_0054`.
- Server work root:
  `/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_enerbil_trace_20260624_2355`.
- Trace file:
  `orchjax_sechiba_bridge_enerbil_active_trace.txt`.
- Tags:
  `before_enerbil_main_active_pft14` and
  `after_enerbil_main_active_pft14`.
- First active row used by tests: `kjit=49`, `ji=1`, `jv=14`,
  `lai=0.779116943429268`.

The pre-call trace includes the previously missing same-call inputs
`swnet`, `soilflx`, `soilflx_pft`, `soilcap`, `soilcap_pft`, `pgflux`,
the driver/PET/PEQ coefficients, DIFFUCO beta/drag fields, and the active
predicate diagnostics `lai`, `gpp`, and `veget_max`. The after-call trace adds
the HYDROL explicit-snow boundary fields `pgflux` and `temp_sol_add`.

Executable status:

- `validate_enerbil_active_module_closure` confirms the active pre-call and
  after-call payloads are field-complete.
- `assemble_enerbil_explicit_local_step_from_server_bridge` now uses trace
  values for all active pre-call inputs when they are present; it does not
  overwrite `rau`, PET/PEQ coefficients, `emis`, or `valpha` with local
  source-kernel recomputation.
- `tests/unit/test_enerbil.py::test_active_enerbil_server_bridge_assembly_matches_fortran_active_trace_outputs`
  matches Fortran active outputs for `temp_sol_new`, `temp_sol_new_pft`,
  `qsurf`, `fluxsens`, `fluxlat`, `evapot`, `evapot_corr`, `vevapnu`,
  `vevapnu_pft`, `vevapsno`, `vevapflo`, `vevapwet`, `transpir`, `transpot`,
  `pgflux`, `temp_sol_add`, and `t2mdiag`.
- The active run has `OK_LAIDEV__00014 = FALSE` in
  `outputs/server_1961_enerbil_active_trace_20260625_0054/run/used_run.def`,
  so PFT14 evaporation/transpiration uses the grid `qsol_sat_new` branch in
  `enerbil_evapveg` lines 1807-1833.

Still not claimed as closed:

- `enerbil_pottemp` is closed for the current source body: the Fortran routine
  initializes `dtheta` and `fevap` to zero, so `q_sol_pot`/`temp_sol_pot` pass
  through unchanged, and the active trace package now includes before/after
  potential-temperature records. This remains a diagnostic branch rather than a
  modelout blocker.
- Kernel-by-kernel internal trace for `psold*`, `qsol_sat*`, `dtheta*`,
  `psnew*`, `qair_new`, and `epot_air_new` has not been collected; current
  parity is module-boundary plus source-kernel output parity for the fields
  listed above.

## Minimum ENERBIL Outputs

The minimum closed output/state bundle for PFT14 paper-case parity is:

| Field | Shape | Producer | Immediate consumer |
| --- | --- | --- | --- |
| `transpir` | `(npts,nvm)` | `enerbil_evapveg` | `hydrol_main`/`hydrol_soil` |
| `transpot` | `(npts,nvm)` | `enerbil_evapveg` | `hydrol_main`; routing only if active |
| `vevapnu` | `(npts)` | `enerbil_evapveg` | `hydrol_main`/`hydrol_split_soil` |
| `vevapnu_pft` | `(npts,nvm)` | `enerbil_evapveg` | `hydrol_main`/`hydrol_split_soil` |
| `vevapwet` | `(npts,nvm)` | `enerbil_evapveg` | `hydrol_main`/`hydrol_canop` |
| `vevapsno` | `(npts)` | `enerbil_evapveg` | explicit snow or bucket snow branch in HYDROL |
| `vevapflo` | `(npts)` | `enerbil_evapveg` | `hydrol_flood` |
| `temp_sol_new` | `(npts)` | `enerbil_surftemp` | `hydrol_main`, `thermosoil_main` |
| `temp_sol_new_pft` | `(npts,nvm)` | `enerbil_surftemp` | `thermosoil_main` |
| `qsurf` | `(npts)` | `enerbil_flux` | next `diffuco_main`, restart/driver diagnostics |
| `evapot` | `(npts)` | `enerbil_flux` | `hydrol_main`, history/modelout |
| `evapot_corr` | `(npts)` | `enerbil_flux` | `hydrol_main`, `slowproc_main`, STOMATE forcing |
| `pgflux` | `(npts)` | `enerbil_flux` explicit-snow branch | `hydrol_main` explicit snow |
| `temp_sol_add` | `(npts)` | `enerbil_flux` explicit-snow branch | `hydrol_main` explicit snow |
| `temp_sol` | `(npts)` | modified/used by ENERBIL state | `slowproc_main`, next DIFFUCO/ENERBIL |
| `temp_sol_pft` | `(npts,nvm)` | modified/used by ENERBIL state | next DIFFUCO/ENERBIL |
| `t2mdiag` | `(npts)` | `enerbil_t2mdiag` | `slowproc_main`/STOMATE forcing |

Fortran provenance:

- `enerbil_main` declares evaporation/transpiration outputs at
  `enerbil.f90` lines 447-455; declares `t2mdiag`, `temp_sol_new`,
  `temp_sol_new_pft`, and `temp_sol_add` at lines 456-459; declares modified
  `evapot`, `evapot_corr`, `temp_sol`, `temp_sol_pft`, `qsurf`, `tsol_rad`,
  and `pgflux` at lines 461-472.
- `temp_sol_new` and `temp_sol_new_pft` are produced by `enerbil_surftemp`,
  called at `enerbil.f90` lines 507-510; assignments are at lines 1203 and
  1204-1212.
- `qsurf`, `evapot`, and `evapot_corr` are produced by `enerbil_flux`, called
  at lines 534-538. `qsurf` is computed at lines 1481-1492, `evapot` at line
  1545, and `evapot_corr` at lines 1613-1640.
- `pgflux` and `temp_sol_add` are updated by the explicit-snow branch inside
  `enerbil_flux`: `PHPSNOW` and the first `pgflux` update are at lines
  1554-1556; the snow-ablation cap at `tp_00`, `zgflux`, `temp_sol_add`, and
  final `pgflux` update are at lines 1561-1588. The JAX source kernel is
  `jax_orchidee/sechiba/enerbil.py::enerbil_flux_explicit_snow_diagnostics`.
- `transpir`, `transpot`, `vevapnu`, `vevapnu_pft`, `vevapwet`, `vevapsno`,
  and `vevapflo` are produced by `enerbil_evapveg`, called at lines 541-545.
  The corresponding formulas are at lines 1756-1787 and 1807-1833.
- `t2mdiag` is assigned by `enerbil_t2mdiag`, called at lines 547-551; the
  subroutine assigns `t2mdiag(:) = temp_air(:)` at lines 1993-2021.

## Bridge Role

### ENERBIL -> HYDROL

These fields enter HYDROL immediately in the same timestep:

- `temp_sol_new`: `sechiba.f90` lines 1049-1055; HYDROL snow code consumes it
  through `explicitsnow_main` when `ok_explicitsnow` is true at
  `hydrol.f90` lines 1177-1190, or through `hydrol_snow` when false at lines
  1192-1197.
- `vevapwet`: consumed by `hydrol_canop`, `hydrol.f90` lines 1232-1234.
- `vevapflo`: consumed by `hydrol_flood`, `hydrol.f90` lines 1237-1238.
- `transpir`, `vevapnu`, `vevapnu_pft`, `evapot`, `evapot_corr`: passed into
  `hydrol_soil`, `hydrol.f90` lines 1278-1284.
- `vevapsno`: passed into snow update, `hydrol.f90` lines 1177-1197.
- `pgflux` and `temp_sol_add`: passed through the same explicit-snow handling
  path as `temp_sol_new` and `vevapsno`, `hydrol.f90` lines 1177-1197. They
  are source-computable only after exact ENERBIL flux inputs are available;
  current server bridge traces do not record them directly.
- `transpot`: used by HYDROL irrigation-demand logic at `hydrol.f90` lines
  1241-1274 and passed by `sechiba.f90` lines 1049-1055.

The water-budget check in `hydrol_main` explicitly combines `runoff`,
`drainage`, `SUM(vevapwet)`, `SUM(transpir)`, `vevapnu`, `vevapsno`, and
`vevapflo`: `hydrol.f90` lines 1359-1360.

### ENERBIL -> THERMOSOIL

`temp_sol_new` and `temp_sol_new_pft` are required before `thermosoil_main`.
SECHIBA maps HYDROL moisture fields to PFT arrays at `sechiba.f90` lines
1093-1102, then calls `thermosoil_main` with `temp_sol_new`,
`temp_sol_new_pft`, snow, heat capacity/flux, and moisture fields at lines
1109-1118. `thermosoil_main` declares those fields at `thermosoil.f90` lines
788-840, then calls `thermosoil_profile` and `thermosoil_energy` at lines
906-913.

### ENERBIL -> SLOWPROC/STOMATE Daily Forcing

`slowproc_main` receives `t2m`, `t2m_min`, `temp_sol`, `stempdiag`, `swdown`,
`t2mdiag`, and `evapot_corr`: `slowproc.f90` lines 343-377 and declarations
at lines 399-433. It calls `stomate_main` at lines 973-1009 with these daily
forcing fields.

Inside STOMATE, `evapot_corr` is accumulated into `evapot_daily` at
`stomate.f90` lines 3188-3195. `temp_sol` and the HYDROL/THERMOSOIL outputs
are accumulated into daily forcing at lines 3198-3208:
`humrel_daily`, `litterhum_daily`, `t2m_daily`, `tsurf_daily`,
`tsoil_daily`, `soilhum_daily`, precipitation, and `gpp_daily`.

### ENERBIL -> Next DIFFUCO

`qsurf`, `temp_sol`, `temp_sol_pft`, `evapot`, and `evapot_corr` re-enter
`diffuco_main` in the next timestep. The call in `sechiba.f90` passes them at
lines 997-1002. `diffuco_main` declares `temp_sol`, `temp_sol_pft`, `qsurf`,
`evapot`, and `evapot_corr` at `diffuco.f90` lines 336-356; aerodynamic and
surface humidity use occurs in `diffuco_aero`, lines 940-1018, and flood
resistance uses `evapot_corr/evapot` at lines 1301-1331.

## Active Paper-Case Branches

Active and must be preserved:

- CWRR hydrology and thermosoil path. `used_run.def` has `HYDROL_CWRR =
  TRUE` at lines 125-127, and `sechiba.f90` calls `hydrol_main` at lines
  1049-1072 and `thermosoil_main` at lines 1109-1118.
- Explicit snow. `used_run.def` has `OK_EXPLICITSNOW = TRUE` at lines
  177-179. HYDROL therefore calls `explicitsnow_main`, not the bucket snow
  branch, at `hydrol.f90` lines 1177-1197. `enerbil_fusion` is skipped because
  `sechiba.f90` calls it only when `.NOT. ok_explicitsnow`, lines 1077-1081.
- Canopy interception water update. `hydrol_canop` is called unconditionally
  within `hydrol_main` at `hydrol.f90` lines 1232-1234, and consumes
  `vevapwet`.
- Soil heat path. `thermosoil_main` is called in the CWRR branch,
  `sechiba.f90` lines 1093-1118.
- STOMATE path. `used_run.def` has `STOMATE_OK_STOMATE = TRUE` at lines
  201-203; `slowproc.f90` calls `stomate_main` when `ok_stomate`, lines
  936-973.
- MICT-leak water bridge. `OK_LEAK = TRUE` in `used_run.def` lines 197-199;
  `slowproc_main` passes `soil_mc`, `wat_flux`, runoff/drainage, and canopy
  water fields to STOMATE at `sechiba.f90` lines 1211-1213 and
  `slowproc.f90` lines 1006-1009.

Inactive or non-minimal for the single-point paper case:

- `hydrolc_main` and `thermosoilc_main` are inactive because `HYDROL_CWRR =
  TRUE`; see `sechiba.f90` lines 1027-1046 and 1119-1128.
- `enerbil_fusion` is inactive because `OK_EXPLICITSNOW = TRUE`; see
  `sechiba.f90` lines 1077-1081 and `used_run.def` lines 177-179.
- Routing has `RIVER_ROUTING = TRUE` in `used_run.def` lines 109-111, but
  `sechiba.f90` calls `routing_main` only if `river_routing .AND. nbp_glo
  .GT. 1`, lines 1227-1234. The current trace grid is one land point, so the
  no-routing branch sets river/routing fields to zero at lines 1235-1240.
- Irrigation forcing is not an active closure requirement for PFT14:
  `DO_IRRIGATION = FALSE` at `used_run.def` lines 129-131 and
  `IRRIG_DRIP = FALSE` at lines 2316-2318. HYDROL still carries irrigation
  arrays and demand logic, but the minimal ENERBIL boundary should not invent
  irrigation input values.
- Canopy multilayer/extinction complexity is inactive: `CANOPY_EXTINCTION =
  FALSE` and `CANOPY_MULTILAYER = FALSE` at `used_run.def` lines 217-223.
- DGVM, peat carbon expansion branches, and erosion are not ENERBIL closure
  blockers: `STOMATE_OK_DGVM = FALSE`, `OK_PC = FALSE`, `OK_PEAT = FALSE`,
  `PEAT_OCCUR = FALSE`, and `EROSION_MODULE = FALSE` in `used_run.def` lines
  113-115 and 181-207.

## Current Trace Coverage

`outputs/server_1961_trace_full_20260623/MANIFEST.txt` lists driver,
intersurf, HYDROL internal traces, slowproc data traces, and STOMATE traces.
That package has no ENERBIL boundary record.

`outputs/server_1961_bridge_trace_20260624` adds
`after_diffuco_main`, `after_enerbil_main`, HYDROL, THERMOSOIL, and SLOWPROC
bridge traces. The `after_enerbil_main` record is useful for verifying
ENERBIL outputs, but it is after the ENERBIL call. It cannot be used as a
pre-call source for `enerbil_begin` or `enerbil_surftemp` inputs. The focused
surface-state audit keeps `soilcap`, `soilcap_pft`, `soilflx`, `soilflx_pft`,
`swnet`, and `emis` missing unless they are supplied by an audited source
kernel or a compatible pre-call trace.

Conclusion: formula slices are source-backed. The early bridge package was
upstream-limited because it mixed cold-start server traces with local restart
state; later source-backed and active trace coverage supersedes that early
diagnostic limitation for the current active paper-case ENERBIL boundary.

## Minimum Trace Contract

Future Fortran instrumentation point:

- Insert immediately after `CALL enerbil_main` in
  `src_sechiba/sechiba.f90::sechiba_main`, after line 1019 and before the
  HYDROL branch at line 1027.

Minimum record label:

- `after_enerbil_main`

Minimum scalar metadata:

- `kjit`, local land index, global `index(ji)`, `nvm`, active PFT id `14`,
  `dt_sechiba`, `ok_explicitsnow`, `hydrol_cwrr`, `river_routing`, `nbp_glo`.

Minimum grid-cell fields:

- `temp_sol`, `temp_sol_new`, `qsurf`, `evapot`, `evapot_corr`, `t2mdiag`,
  `vevapnu`, `vevapsno`, `vevapflo`, `vevapp`, `fluxsens`, `fluxlat`,
  `tsol_rad`, `pgflux`, `temp_sol_add`.

Minimum PFT14 fields:

- `temp_sol_pft(ji,14)`, `temp_sol_new_pft(ji,14)`,
  `transpir(ji,14)`, `transpot(ji,14)`, `vevapnu_pft(ji,14)`,
  `vevapwet(ji,14)`, `veget_max(ji,14)`.

Minimum upstream diagnostic fields to make the trace auditable:

- `vbeta`, `vbeta_pft(ji,14)`, `valpha`, `vbeta1`, `vbeta2(ji,14)`,
  `vbeta3(ji,14)`, `vbeta3pot(ji,14)`, `vbeta4`, `vbeta4_pft(ji,14)`,
  `vbeta5`, `q_cdrag`, `q_cdrag_pft(ji,14)`, `humrel(ji,14)`,
  `qair`, `temp_air`, `pb`, `rau`, `lwdown`, `swnet`, `soilcap`,
  `soilcap_pft(ji,14)`, `soilflx`, `soilflx_pft(ji,14)`, `snowdz(ji,:)`.

These are enough to verify the ENERBIL boundary without tracing every local
work array. If formula-level debugging is needed later, add inside-ENERBIL
records after `enerbil_surftemp`, `enerbil_flux`, and `enerbil_evapveg` calls
at `enerbil.f90` lines 507-545.

## Scaffold Added

`jax_orchidee/sechiba/enerbil_bridge.py` declares the field-level ENERBIL
boundary contract. Each field records shape, direction, role, consumers,
Fortran provenance, and trace status. The validator checks only advertised
field names and does not derive missing process values.

## 2026-06-24 First-Step Assembly Check

`tests/unit/test_enerbil.py::test_enerbil_first_step_local_restart_assembly_remains_non_parity_for_server_bridge`
assembles the current source-backed local ENERBIL sequence for `kjit=1`,
`ji=1`, `jv=14`:

- DIFFUCO pre-call fields come from `after_diffuco_main`.
- `emis` comes from the audited `condveg_initialize` branch
  `IMPOSE_AZE=FALSE`, i.e. `emis=1`.
- `rau` and non-WATCHOUT driver energy coefficients come from the audited
  `sechiba_var_init`/`dim2_driver` formulas.
- `swnet` is computed from the currently audited `intersurf_main.swdown` and
  two-band `driver_start.nc` albedo. This is valid for the local restart case,
  not automatically for the server cold-start bridge run.
- `soilcap*` and `soilflx*` come from the local `sechiba_start.nc`
  THERMOSOIL restart fields. This intentionally tests that the local restart
  case must not be mixed with the server cold-start bridge trace.

This local assembly reproduces the zero PFT14 evaporation split fields
(`transpir`, `transpot`, `vevapnu_pft`) and `qsurf` at the
`after_enerbil_main` boundary, but it does not reproduce the Fortran
`temp_sol_new`/`temp_sol_new_pft` or `evapot` values. With the audited inputs
above, local `temp_sol_new` is `284.40084613976836 K` and local `evapot` is
`0.0`; the bridge trace has `temp_sol_new=295.744633030835 K` and
`evapot=0.05719239060617407`.

This is an upstream-input closure gap, not a license to tune ENERBIL. The
server bridge trace is a cold-start 1961 run (`SECHIBA_restart_in=NONE`,
`RESTART_FILEIN=NONE`), whereas `reference/case_001_071` is a restart-driven
case (`SECHIBA_restart_in=sechiba_start.nc`, `RESTART_FILEIN=driver_start.nc`).
The local restart contains `soilcap_pft(1,14)=45865.2880547002`; the bridge
trace records `soilcap_pft(1,14)=59764.3072135468`. Because
`enerbil.f90::enerbil_main` declares `soilcap*` and `soilflx*` as
`INTENT(in)` at lines 433-438, the bridge value is a same-call diagnostic input
from that cold-start run, not an ENERBIL output. The lagged source is
THERMOSOIL: `thermosoil_main` computes coefficients for the following ENERBIL
timestep near lines 995-1010 and in `thermosoil_coef`.

Therefore, first-step ENERBIL parity must use one coherent runtime truth:
either the server cold-start bridge input tuple, including traced/lagged
THERMOSOIL coefficients, or a new local trace generated with the same restart,
forcing year, and domain as `reference/case_001_071`. Mixing server
`after_enerbil_main` outputs with local restart inputs is explicitly forbidden
by the new assembly test.

`jax_orchidee/sechiba/enerbil.py::assemble_enerbil_explicit_local_step_from_server_bridge`
and
`tests/unit/test_enerbil.py::test_enerbil_server_bridge_assembly_reports_missing_trace_inputs_without_restart_mixing`
now encode the cold-start server boundary directly. The helper consumes one
coherent runtime truth:

- DIFFUCO pre-call fields from the server `after_diffuco_main` bridge trace.
- Driver/intersurf fields from the server bridge `intersurf_main` trace.
- `soilcap` and `soilcap_pft` from the same-call server
  `after_enerbil_main` record only as trace input diagnostics. They are
  accepted under this label because `enerbil_main` declares them `INTENT(in)`
  at `enerbil.f90` lines 433-438; this is not a THERMOSOIL source-kernel
  port.

The copied bridge traces still do not contain exact pre-ENERBIL `swnet`,
`soilflx`, or `soilflx_pft` for the cold-start server run. The server
assembler therefore returns no local step and reports exactly those missing
inputs. It does not substitute `reference/case_001_071/sechiba_start.nc`, does
not derive server `swnet` from local `driver_start.nc`, and does not claim
`thermosoil_main` or `thermosoil_coef` has been ported. The lag remains
source-audited only: `thermosoil_main` lines 995-1010 call `thermosoil_coef`
after ENERBIL to prepare `soilcap*`/`soilflx*` for the next timestep, with
`thermosoil_coef` output declarations at lines 1423-1427.

Tests: `tests/unit/test_enerbil.py` and `tests/unit/test_enerbil_bridge.py`.

## Closure Status And Next Order

Closed by source audit:

- The call position and downstream role of `enerbil_main`.
- The minimum ENERBIL input/output/state field list for PFT14 single-point
  closure.
- Active/inactive branch classification for snow, hydrology, thermosoil,
  slowproc/STOMATE, routing, irrigation, and canopy complexity.
- The exact `after_enerbil_main` trace fields needed before implementation
  parity can be claimed.

Current closure boundary:

- Numerical ENERBIL parity for the active PFT14 module boundary is covered by
  `tests/unit/test_enerbil.py::test_active_enerbil_server_bridge_assembly_matches_fortran_active_trace_outputs`,
  which compares the active pre-call and after-call trace pair without mixing
  restart truths.
- `swnet` is closed for the archived local first step when
  driver/intersurf `swdown` is paired with exact two-band albedo from
  `driver_start.nc`; trace `swdown` alone remains insufficient. For the local
  first-step case, `emis` is closed from `condveg_initialize` with audited
  `IMPOSE_AZE=FALSE`; `IMPOSE_AZE=TRUE` scenarios still require audited
  `CONDVEG_EMIS` or a pre-call trace.
- Non-restart `soilcap*`/`soilflx*` initialization is covered through the
  source-backed `thermosoil_coef` path when its coherent input state is
  supplied. For the archived local reference first step, the four fields are
  still closed directly from `sechiba_start.nc`; see
  `docs/source_audits/enerbil_soil_thermal_state_input_chain.md`.
- Full ENERBIL call-order packaging for the active downstream state is covered
  at module boundary. Source-backed helpers exist for the current
  `enerbil_pottemp` body, the full
  `evapot_corr` correction path, and `enerbil_t2mdiag`; `evapot_corr`,
  `t2mdiag`, and potential-temperature diagnostics are wired into the active
  module-boundary comparison.
- Kernel-by-kernel internal ENERBIL work-array traces are not required for the
  current closure claim. They remain optional diagnostics if a future mismatch
  needs line-level localization inside `enerbil_main`.

Minimum implementation order after trace support:

1. Use the audited `driver_start.nc` `albedo_vis/albedo_nir` plus
   driver/intersurf `swdown` chain for current local first-step `swnet`, or add
   a direct pre-call `swnet/albedo` trace for future cases.
2. Keep `emis` closed from the audited CONDVEG initialization path for the
   current first step; require same-case `CONDVEG_EMIS` or a pre-call trace
   before claiming `IMPOSE_AZE=TRUE` coverage.
3. Keep `soilcap*` and `soilflx*` closed from the same-case THERMOSOIL
   restart for the archived first step; port or trace `thermosoil_coef` before
   claiming non-restart initialization coverage.
4. Extend remaining ENERBIL slices only with Fortran line-level provenance.
5. Wire traced ENERBIL outputs into HYDROL/thermosoil/slowproc bridge tests
   without filling untraced values with calibrated constants.
6. Wire only the traced ENERBIL outputs into HYDROL/thermosoil/slowproc bridge
   tests; do not fill untraced values with calibrated constants.
