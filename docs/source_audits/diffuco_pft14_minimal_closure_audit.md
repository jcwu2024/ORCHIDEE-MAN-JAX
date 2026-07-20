# DIFFUCO PFT14 Minimal Closure Audit

Scope: source audit and minimal schema for the paper-case single point, PFT14
path through `diffuco_main`. This does not implement DIFFUCO formulas and does
not infer missing values from calibration or traces.

## Source Path

The active offline path enters `sechiba_main` from
`src_sechiba/intersurf.f90::intersurf_main_2d`, which calls `sechiba_main` at
lines 640-648. Inside `src_sechiba/sechiba.f90::sechiba_main`, the process
order is:

1. `sechiba_var_init`: line 982.
2. `diffuco_main`: lines 997-1005.
3. `enerbil_main`: lines 1013-1019.
4. CWRR `hydrol_main`: lines 1046-1072.
5. `condveg_main`: lines 1085-1091.
6. `thermosoil_main`: lines 1093-1118.
7. `slowproc_main`: lines 1184-1216.

`slowproc_main` calls `stomate_main` when `ok_stomate` at
`src_sechiba/slowproc.f90` lines 936-973. Therefore DIFFUCO is upstream of
same-step ENERBIL, HYDROL, SLOWPROC, and STOMATE.

## Minimal PFT14 DIFFUCO Boundary

`src_sechiba/sechiba.f90::sechiba_main` passes these relevant inputs into
`diffuco_main` at lines 997-1005:

- Meteorology and surface state:
  `u`, `v`, `zlev`, `z0m`, `z0h`, `roughheight`, `roughheight_pft`,
  `temp_sol`, `temp_sol_pft`, `temp_air`, `temp_growth`, `rau`, `qsurf`,
  `qair`, `q2m`, `t2m`, `pb`, `swnet`, `swdown`, `coszang`, `ptnlev1`,
  `precip_rain`.
- Previous or upstream hydrology/surface state:
  `rsol`, `evap_bare_lim`, `evapot`, `evapot_corr`, `snow`, `flood_frac`,
  `flood_res`, `frac_nobio`, `snow_nobio`, `totfrac_nobio`, `humrel`,
  `qsintveg`, `qsintmax`, `tot_bare_soil`, `frac_snow_veg`,
  `frac_snow_nobio`.
- PFT/carbon/mangrove state:
  `ccanopy`, `veget`, `veget_max`, `lai`, `assim_param`, `salinity`,
  `rprof`, `tide_height`, `biomass`, `frac_age`, coordinates and grid
  metadata.
- Inout drag state:
  `tq_cdrag`, `tq_cdrag_pft`.

`src_sechiba/diffuco.f90::diffuco_main` declares the minimum outputs at lines
391-405 and inout variables at lines 407-411:

- Required for STOMATE carbon: `gpp(npts,nvm)`.
- Required for photosynthesis/stomatal audit: `gsmean`, `rveget`, `rstruct`,
  `cimean`.
- Required for ENERBIL evapotranspiration: `vbeta`, `vbeta_pft`, `valpha`,
  `vbeta1`, `vbeta2`, `vbeta3`, `vbeta3pot`, `vbeta4`, `vbeta4_pft`,
  `vbeta5`.
- Required for ENERBIL resistance context: updated `q_cdrag`,
  `q_cdrag_pft`.
- Required to audit PFT14 mangrove GPP: internal diagnostics
  `control_salinity` and `control_inudate`, declared at `diffuco.f90` lines
  81-83 and sent to XIOS at lines 722-723.
- Required to preserve downstream water-stress state: `humrel`, because
  `diffuco_comb` receives it as inout at `diffuco.f90` lines 3070-3108 and
  may zero it in high-humidity/no-transpiration cases at lines 3230-3239.

## GPP and Stomatal Coupling

The active paper case has `STOMATE_OK_CO2=y` in
`outputs/server_1961_trace_full_20260623/run/run.def` lines 61-62. The flag is
read in `src_sechiba/intersurf.f90` lines 2152-2161, and if STOMATE is active
then `ok_co2` is forced true at lines 2272-2273. Therefore
`diffuco_main` takes the `diffuco_trans_co2` branch at
`src_sechiba/diffuco.f90` lines 665-671, not the Jarvis-only
`diffuco_trans` branch at lines 673-678.

Within `diffuco_trans_co2`, the output contract is declared at
`diffuco.f90` lines 2018-2088. For PFT14, mangrove salinity and inundation
controls are not optional:

- `READ_SALINITY=TRUE`, `READ_TIDE=TRUE`, and `tides=y` in
  `outputs/server_1961_trace_full_20260623/run/run.def` lines 14-17 and
  190-191.
- `diffuco_main` computes `control_salinity` and `control_inudate` only on
  first call at `diffuco.f90` lines 445-622.
- `diffuco_trans_co2` applies both controls when `jv == 14` at lines
  2834-2844.
- `gpp` is then assigned from controlled `assimtot` at lines 2890-2897.

Local assembly now treats `npts` as the Fortran land-point dimension rather
than a paper-case constant: `humcste_use`/`rprof` are replicated to the first
dimension inferred from `tide_height`, `biomass`, or an explicit `npts`
argument, and mismatched land dimensions are rejected. This preserves the
single-point paper trace while allowing source-backed multi-grid micro-cases
for the PFT14 inundation control path.

The same block assigns conductance/resistance outputs:

- `gsmean`: lines 2884-2885.
- `cimean`: lines 2887-2895.
- `rveget`: lines 2915-2921.
- `rstruct`: lines 2925-2931.
- `vbeta3` and `vbeta3pot`: lines 2946-2969.

`gpp` is passed to `slowproc_main` by `sechiba_main` at lines 1184-1192.
`slowproc_main` passes it into `stomate_main` at `slowproc.f90` lines
973-982. `stomate_main` accumulates `gpp_d` into `gpp_daily` at
`src_stomate/stomate.f90` lines 3198-3208. `stomate_lpj.f90` writes GPP to
XIOS and history at lines 1677-1678 and 2200-2203.

## Evaporation and Transpiration Bridge

DIFFUCO does not directly output `transpir`, `vevapwet`, `vevapnu`,
`vevapsno`, `vevapflo`, `evapot`, or `evapot_corr`. It produces the beta and
drag fields that ENERBIL uses to diagnose them.

`src_sechiba/sechiba.f90::sechiba_main` passes DIFFUCO outputs
`vbeta`, `vbeta_pft`, `valpha`, `vbeta1`, `vbeta2`, `vbeta3`, `vbeta3pot`,
`vbeta4`, `vbeta4_pft`, `vbeta5`, `tq_cdrag`, and `tq_cdrag_pft` into
`enerbil_main` at lines 1013-1019.

`src_sechiba/enerbil.f90::enerbil_main` declares:

- input beta fields at lines 429-443;
- outputs `vevapnu`, `vevapnu_pft`, `vevapsno`, `vevapflo`, `transpir`,
  `transpot`, `vevapwet`, `t2mdiag`, and `temp_sol_new` at lines 449-458;
- inout `evapot`, `evapot_corr`, and `qsurf` at lines 463-467.

`enerbil_main` calls `enerbil_evapveg` at lines 541-545. In
`enerbil_evapveg`, `vevapsno`, `vevapnu`, `vevapnu_pft`, `vevapflo`,
`vevapwet`, `transpir`, and `transpot` are declared at lines 1719-1725 and
assigned from the beta fields at lines 1760-1832. `evapot` is diagnosed in
`enerbil_flux` at `enerbil.f90` line 1545, and `evapot_corr` at line 1639.

## Downstream Consumers

`hydrol_main` consumes ENERBIL evaporation/transpiration fields at
`sechiba.f90` lines 1049-1055. In `src_sechiba/hydrol.f90::hydrol_main`,
the downstream exported fields include:

- `soil_mc`, `wat_flux`, `precip2canopy`, `precip2ground`, `canopy2ground`,
  `runoff_per_soil`, `runoff2peat`, and `drainage_per_soil`: lines
  1031-1038.
- `vegstress`, `shumdiag`, `litterhumdiag`, and inout `humrel`: lines
  1041-1063.
- `mc_layh`, `mcl_layh`, `soilmoist_out`, `mc_layh_s`, and `mcl_layh_s`:
  lines 1085-1089.

`hydrol_main` also calls `hydrol_canop` with `precip2canopy`,
`precip2ground`, and `canopy2ground` at `hydrol.f90` lines 1232-1234 and
calls `hydrol_soil` with MICT-leak outputs at lines 1278-1296.

`slowproc_main` consumes the final SECHIBA bundle at `sechiba.f90` lines
1184-1216. For OK_LEAK, it passes `soil_mc`, `wat_flux`,
`drainage_per_soil`, `runoff_per_soil`, `runoff2peat`, `precip2canopy`,
`precip2ground`, and `canopy2ground` at lines 1211-1213.
`src_stomate/stomate_soilcarbon.f90::soilcarbon_leak` consumes those water
fields at lines 641-705. It uses `canopy2ground` to compute
`DOC_canopy2ground` at lines 1497-1512.

`condveg_main` consumes HYDROL-updated `drysoil_frac`, snow state, and canopy
surface variables at `sechiba.f90` lines 1085-1091. In
`src_sechiba/condveg.f90`, `condveg_main` recomputes snow fractions and roughness
at lines 398-425 and albedo at lines 431-435; with `ROUGH_DYN=y` in
`outputs/server_1961_trace_full_20260623/run/run.def` line 236, the dynamic
roughness branch is active at `condveg.f90` lines 420-425.

`thermosoil_main` consumes HYDROL moisture fields at `sechiba.f90` lines
1109-1118, then `stempdiag` is passed to `slowproc_main` at lines 1184-1188
and accumulated by `stomate_main` at `stomate.f90` lines 3201-3208.

## Active and Skippable Paper-Case Branches

Active, not skippable for the PFT14 single-point path:

- CO2 photosynthesis: `STOMATE_OK_CO2=y` in run.def lines 61-62; used by
  `diffuco.f90` lines 665-671.
- STOMATE: `STOMATE_OK_STOMATE=y` in run.def line 61; `slowproc.f90` calls
  `stomate_main` at lines 936-973.
- Mangrove salinity/tide GPP control: `READ_SALINITY=TRUE`,
  `READ_TIDE=TRUE`, `tides=y` in run.def lines 14-17 and 190-191;
  `diffuco.f90` lines 445-622 and 2834-2844.
- CWRR HYDROL: `HYDROL_CWRR=y` in run.def line 70; `sechiba.f90` uses
  `hydrol_main` at lines 1046-1072.
- Explicit snow: `OK_EXPLICITSNOW=y` in run.def line 69; `sechiba.f90` skips
  `enerbil_fusion` when true at lines 1077-1081.
- OK_LEAK water/carbon bridge: `OK_LEAK=y` in run.def line 220; MICT-leak
  water fields are passed to slowproc at `sechiba.f90` lines 1211-1213.
- Dynamic roughness: `ROUGH_DYN=y` in run.def line 236; `condveg.f90` lines
  420-425.

Inactive or skippable for this minimal closure:

- Jarvis-only `diffuco_trans`: inactive because `ok_co2` is true; see
  `diffuco.f90` lines 665-678.
- BVOC chemistry: `CHEMISTRY_BVOC=FALSE` in
  `outputs/server_1961_trace_full_20260623/run/used_run.def` lines 209-211;
  `diffuco.f90` calls `chemistry_bvoc` only at lines 686-692.
- Choisnel `hydrolc_main`: inactive because `HYDROL_CWRR=y`; see
  `sechiba.f90` lines 1027-1040 and 1046-1072.
- `enerbil_fusion`: inactive because `OK_EXPLICITSNOW=y`; see
  `sechiba.f90` lines 1077-1081.
- Two-layer/deep peat carbon flags `OK_PEAT` and `OK_PC`: both `n` in
  run.def lines 202-203.
- Floodplain routing/infiltration: `DO_FLOODPLAINS=n`,
  `DO_FLOODINFILT=n` in run.def lines 223-226.
- DGVM: `STOMATE_OK_DGVM=n` in run.def line 182.
- Fire module execution: `FIRE_DISABLE=y` in run.def line 189; this does not
  remove the `slowproc_main`/`stomate_main` carbon path.

## Existing Trace Coverage

`outputs/server_1961_trace_full_20260623/MANIFEST.txt` lists driver,
intersurf, HYDROL, slowproc, and STOMATE traces, but no
`orchjax_diffuco_*` or `after_diffuco_main` trace. The STOMATE daily trace
does contain `gpp_before_accu`/`gpp_after_accu` records, but these are after
the DIFFUCO boundary and do not expose beta fields, conductance/resistance,
drag, or mangrove control factors.

Therefore the original DIFFUCO trace package was not sufficient for strict
JAX-vs-Fortran numeric parity. The current bridge package does expose
`after_diffuco_main` output-boundary fields, and
`diffuco_numeric_parity_readiness` makes that coverage executable, but it
still lacks same-timestep `diffuco_trans_co2` dynamic inputs such as `qsatt`,
`Ca`, `vcmax`, `vbeta23`, and the mangrove control factors.

Minimum future Fortran instrumentation point:

- Insert an `after_diffuco_main` record immediately after
  `sechiba.f90::sechiba_main` line 1005.
- For land point 1 and PFT14 where applicable, write:
  `gpp`, `gsmean`, `rveget`, `rstruct`, `cimean`, `vbeta3`, `vbeta3pot`,
  `vbeta2`, `vbeta`, `vbeta_pft`, `valpha`, `vbeta1`, `vbeta4`,
  `vbeta4_pft`, `vbeta5`, `q_cdrag`, `q_cdrag_pft`,
  `control_salinity`, `control_inudate`, and `humrel`.
- Also write selected inputs needed to reproduce the branch without ambiguity:
  `salinity`, first/selected `tide_height`, `rprof(:,14)`,
  `biomass(:,14,:,icarbon)`, `veget(:,14)`, `veget_max(:,14)`,
  `lai(:,14)`, `qsintveg(:,14)`, `qsintmax(:,14)`, `temp_sol`,
  `temp_sol_pft(:,14)`, `temp_air`, `temp_growth`, `qsurf`, `qair`,
  `q2m`, `t2m`, `pb`, `swdown`, `swnet`, `evapot`, `evapot_corr`,
  `snow`, `frac_snow_veg`, `tot_bare_soil`, `evap_bare_lim`, and
  `humrel(:,14)` before the call.
- Add the planned `after_diffuco_trans_co2_pft14` record inside
  `diffuco.f90::diffuco_trans_co2`, after the PFT14 output layer and before
  leaving the PFT loop. The local patch draft writes
  `orchjax_diffuco_trans_co2_trace.txt` with the exact dynamic inputs and
  outputs needed by `diffuco_trans_co2_c3_pft_explicit`: `swdown`, `pb`,
  `qsurf`, `qsatt`, `t2m`, `temp_growth`, `Ca`, `vcmax`, `humrel`, `veget`,
  `veget_max`, `lai`, `qsintveg`, `qsintmax`, `vbeta23`, `q_cdrag`,
  `q_cdrag_pft`, `wind`, `control_salinity`, `control_inudate`, and the
  resulting `gpp`, `gsmean`, `rveget`, `rstruct`, `cimean`, `vbeta3`,
  `vbeta3pot`, plus canopy accumulator diagnostics.

Recommended paired records:

- `after_enerbil_main`, immediately after `sechiba.f90` line 1019, with
  `transpir`, `transpot`, `vevapnu`, `vevapnu_pft`, `vevapwet`,
  `vevapsno`, `vevapflo`, `evapot`, `evapot_corr`, `temp_sol`,
  `temp_sol_new`, `temp_sol_pft`, `temp_sol_new_pft`, `qsurf`, and
  `t2mdiag`.
- `after_hydrol_main_before_condveg`, immediately after `sechiba.f90` line
  1072, with HYDROL/STOMATE/MICT-leak bridge fields documented in
  `docs/source_audits/sechiba_pft14_bridge_gap_audit.md`.

## Scaffold

Added `jax_orchidee/sechiba/diffuco_bridge.py`:

- declares DIFFUCO minimal boundary fields and active branch facts;
- stores Fortran provenance for each field;
- exposes `required_after_diffuco_trace_fields`;
- validates payload field presence without deriving missing process values.

Tests: `tests/unit/test_diffuco_bridge.py`.

## Closure Status and Next Order

Closed by source audit:

- `diffuco_main` position in the paper-case execution chain.
- Minimal PFT14 DIFFUCO input/output boundary.
- Active CO2/mangrove/CWRR/OK_LEAK branches and inactive BVOC/Jarvis-only/
  Choisnel/OK_PEAT/OK_PC/floodplain branches.
- Source-backed CWRR `diffuco_bare` beta algebra via
  `diffuco_bare_cwrr_beta`: `diffuco.f90::diffuco_bare` lines 1670-1685 set
  grid `vbeta4 = MIN(evap_bare_lim, 1 - SUM(vbeta2+vbeta3))` and per-PFT
  `vbeta4_pft = MIN(evap_bare_lim, veget_max - (vbeta2+vbeta3))` for present
  vegetation. The inactive/commented `evap_bare_lim_pft` path is not used, so
  HYDROL `evap_bare_lim_ns` must not be fed directly to DIFFUCO.
- Full explicit `diffuco_comb` beta/humidity algebra via
  `diffuco_comb_explicit`: warm dew, freezing dew, interception coefficient,
  transpiration/humidity zeroing, bare-soil interception overrule, and final
  near-zero beta cleanup from `diffuco_comb` lines 3121-3267. The helper takes
  `qsatt` explicitly because Fortran obtains it from `qsatcalc` at line 3126.
  `diffuco_comb_final_beta_bundle_no_dew` remains available as a guarded
  shortcut only with explicit `qsatt >= qair` proof.
- The pre-FvCB `diffuco_trans_co2` activity and water-input gates via
  `diffuco_trans_co2_activity`: LAI/vegetation/radiation/humidity/growth
  temperature assimilation mask from lines 2338-2384, plus `zqsvegrap` and
  `water_lim` from lines 2393-2400. This is only a gate/input helper; it does
  not implement Yin/FvCB assimilation or conductance.
- LAI discretization and Beer light fractions via `diffuco_lai_light_table`,
  from `diffuco_trans_co2` lines 2214-2234. Runtime constants (`nlai`,
  `laimax`, `lai_level_depth`) and PFT `ext_coeff` remain explicit inputs with
  Fortran defaults available.
- VPD, `fvpd`, and boundary-layer conductance via
  `diffuco_vpd_boundary_conductance`, from `diffuco_trans_co2` lines
  2267-2276 and 2486-2497. It requires explicit `qsatt`; the qsat table itself
  is not recomputed inside DIFFUCO.
- Temperature-response inputs via `diffuco_photo_temperature_response`, with
  `diffuco_arrhenius` and `diffuco_arrhenius_modified` matching
  `diffuco.f90` helper functions lines 3356-3412. The bundle covers
  `T_KmC`, `T_KmO`, `T_Sco`, `T_gamma_star`, `T_Rd`, acclimated
  `T_Jmax/T_Vcmax`, `T_gm`, `vc`, `vj`, `gm`, `g0var`, `KmC`, `KmO`, `Sco`,
  `gamma_star`, and `low_gamma_star` from lines 2427-2484. Active PFT
  parameters and `vcmax` remain explicit inputs.
- The C3 Yin/FvCB scalar layer solver via `diffuco_c3_assimilation_yin_layer`,
  matching the Fortran state machine in `diffuco_trans_co2` lines 2705-2785.
  It preserves the two-pass Vc/J root selection, the Vc reset when the J root
  is not lower, the `A_1 == 9999`/`A_1 < -Rd` fallback to `-Rd`, and exposes
  per-branch cubic diagnostics for trace comparison.
- The C3 fixed-LAI canopy integration via `diffuco_c3_canopy_layer_integrals`,
  covering `diffuco_trans_co2` lines 2503-2558 and 2788-2872. The helper
  computes layer masks, `N_Vcmax`, `vc2`, `vj2`, `Rd`, absorbed light, `JJ`,
  layer `assimi`/`gs`/`leaf_ci`, and the canopy `assimtot`, `Rdtot`,
  `gstot`, `leaf_gs_top`, `ilai`, `laisum`, and `cim` accumulators for one
  active C3 PFT. PFT14 salinity/inundation controls remain a caller-side
  mutation of `assimtot` before output conversion, as in lines 2834-2844.
- The PFT14 `assimtot` control mutation via
  `diffuco_apply_pft14_assimtot_controls`, matching
  `diffuco_trans_co2` lines 2834-2844. This is deliberately not GPP-only:
  controlled `assimtot` must also feed `cimean` at lines 2888-2894.
- A one-PFT C3 `diffuco_trans_co2` source-order wrapper via
  `diffuco_trans_co2_c3_pft_explicit`: it wires the closed activity gate,
  LAI/light table, VPD/boundary conductance, temperature response, C3 canopy
  integration, PFT14 `assimtot` controls, and output conversion for one active
  C3 PFT. Inputs that Fortran obtains upstream (`qsatt`, `humrel`, `vbeta23`,
  mangrove controls, and PFT parameters) remain explicit and are not inferred.
- A PFT14 C3 DIFFUCO beta-chain wrapper via
  `diffuco_pft14_c3_beta_closure_explicit`, matching `diffuco_main` source
  order lines 665-710: `diffuco_trans_co2` -> `diffuco_bare` ->
  `diffuco_comb`. It inserts the closed PFT14 `vbeta3/vbeta3pot` column into
  caller-supplied full-PFT beta arrays, computes CWRR bare-soil beta from
  `evap_bare_lim`, then applies the explicit `diffuco_comb` dew/no-dew
  algebra. Other PFT columns remain caller-supplied, so this wrapper is a
  PFT14 closure without pretending to implement all PFTs.
- The post-FvCB `diffuco_trans_co2` output layer via
  `diffuco_trans_co2_outputs_from_fvcb`: it converts explicit FvCB
  intermediates into `gsmean`, `cimean`, `gpp`, `rveget`, `rstruct`,
  `vbeta3`, and `vbeta3pot` following lines 2884-2969. This helper assumes
  `assimtot`, `gstot`, `leaf_gs_top`, `Rdtot`, `gamma_star`, `fvpd`,
  `g0var`, `laisum`, and Fortran 1-based `ilai` have already been
  source-computed or traced; it includes the Fortran `speed = MAX(min_wind,
  wind)` factor in `cresist`.
- Bridge direction from DIFFUCO beta fields to ENERBIL evaporation and from
  ENERBIL/HYDROL to STOMATE/MICT-leak.
- Executable numeric-parity readiness via
  `diffuco_numeric_parity_readiness`, which confirms that the copied active
  DIFFUCO trace package has the dynamic inputs needed to rerun the strict
  PFT14 C3 wrapper.
- Active and inactive strict parity for the PFT14 C3
  `diffuco_trans_co2` kernel against
  `outputs/server_1961_diffuco_active_after_main_trace_20260624_2244`. The
  active record starts at `kjit=49` (`lai=0.779116943429268`), and the local
  wrapper matches traced `gpp`, `gsmean`, `rveget`, `rstruct`, `cimean`,
  `vbeta3/vbeta3pot`, controlled `assimtot`, `Rdtot`, converted local
  `gstot`, `leaf_gs_top`, `laisum`, `cim`, `ilai`, `gamma_star`, `fvpd`, and
  `g0var`.
- The available overlap between `diffuco_trans_co2` and the SECHIBA
  `after_diffuco_main` bridge boundary is executable via
  `validate_diffuco_trans_to_after_main_overlap`. For `kjit=1..4`, the shared
  fields (`gpp`, `gsmean`, `rveget`, `rstruct`, `cimean`,
  `vbeta3/vbeta3pot`, drag, humidity, interception, vegetation, LAI, and
  `qsurf`) match exactly for the inactive early-timestep rows.
- Current module-closure status is executable via
  `validate_diffuco_module_closure`: active `diffuco_trans_co2`, inactive
  early-step `after_diffuco_main` overlap, and active
  `after_diffuco_main_active_pft14` are all present and closed.
- Active PFT14 `diffuco_main` beta-bundle parity is covered by
  `test_server_1961_diffuco_pft14_active_main_beta_closure_matches_after_main_trace`.
  It composes the JAX chain in Fortran order
  `diffuco_trans_co2 -> diffuco_bare -> diffuco_comb` and matches the active
  `after_diffuco_main_active_pft14` row for `vbeta3/vbeta3pot`,
  `vbeta4/vbeta4_pft`, `valpha`, `vbeta`, `vbeta_pft`, `vbeta1`, and
  `vbeta2`.

Superseding closure boundary:

- Active PFT14 `diffuco_main` module-boundary beta/resistance/GPP parity is
  source-backed and executable through the active `diffuco_trans_co2 ->
  diffuco_bare -> diffuco_comb` chain. The closed claim is PFT14-specific; it
  does not imply arbitrary multi-PFT runtime wiring.
- Runtime multi-PFT `diffuco_trans_co2` wiring outside the closed PFT14 C3
  helper stack is not a current PFT14 semantic gap. Any future non-PFT14 PFT
  support must supply exact `assim_param`/`vcmax` downregulation inputs and
  call-boundary salinity/inundation state instead of inferring missing columns.
- Numeric ENERBIL evaporation/transpiration and HYDROL exported-boundary
  closure are now covered in their own module ledgers and tests. They should
  not remain counted as DIFFUCO-open work.

Minimum implementation order:

1. Wire the active PFT14 `diffuco_trans_co2` wrapper into a runtime
   `diffuco_main`-level adapter only when all same-step call-boundary arrays
   are explicit; do not fill other PFT columns by inference.
2. Use the now-closed DIFFUCO beta/resistance boundary as the input truth for
   ENERBIL, starting from `after_enerbil_main` parity.
3. Port the minimal ENERBIL
   evapotranspiration bridge.
4. Use HYDROL exported-boundary trace to close `humrel`, `vegstress`,
   MICT-leak water fields, and thermosoil moisture inputs before extending
   STOMATE/modelout parity.
