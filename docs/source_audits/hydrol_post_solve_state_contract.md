# HYDROL Post-Solve State Contract

This note records the Phase 1H source audit after the main tridiagonal solve.
It is not a full `hydrol_soil` implementation plan; it only separates closed
post-solve formulas from stateful branches that still require exact trace.

Source truth: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`.

## HYDROL Main Pre-Soil Kernels

Two source-order helpers are now closed before the full soil solve:

- `hydrol_vegupd_static_state`: `hydrol_main` calls `hydrol_vegupd` at line
  1206; after the stateful `hydrol_tmc_update` call, `hydrol_vegupd` computes
  soil/PFT masks and split inputs at lines 5191-5248. The helper covers
  `mask_soiltile`, `mask_veget`, `vegetmax_soil`, `frac_bare`, and
  `frac_bare_ns`. It deliberately excludes the preceding `hydrol_tmc_update`
  state migration and its `drain_upd`/`runoff_upd` outputs.
- `hydrol_canop_interception`: `hydrol_main` calls `hydrol_canop` at lines
  1232-1234; subroutine `hydrol_canop` lines 4992-5117 subtracts ENERBIL
  interception loss from `qsintveg`, adds rain interception using
  `throughfall_by_pft`, limits canopy storage by `qsintmax`, computes
  `precisol`, `precip2canopy`, `precip2ground`, and `canopy2ground`, then
  distributes `tot_melt` by `veget_max/vegtot`.
- `hydrol_flood_reservoir`: `hydrol_main` calls `hydrol_flood` at lines
  1237-1238; subroutine `hydrol_flood` lines 5276-5334 limits floodplain
  evaporation by `flood_res`, sends any evaporation deficit to `subsinksoil`,
  computes `floodout`, and scales `precisol` by `(1 - flood_frac)`.
- `hydrol_split_soil_fluxes`: `hydrol_soil` calls `hydrol_split_soil` at line
  5698; subroutine `hydrol_split_soil` lines 8603-8925 transforms PFT-level
  throughfall, bare-soil evaporation, transpiration, and root uptake into
  soil-tile/layer arrays. Implemented source blocks are `precisol_ns` lines
  8651-8661, `vevapnu_ns` lines 8665-8675, old `ae_ns` aggregation into
  `vevapnu_old` lines 8680-8688, new `ae_ns` update lines 8693-8723,
  `tr_ns` lines 8741-8755, and `rootsink` lines 8760-8773.
- `hydrol_irrigation_demand_ratio`: `hydrol_main` lines 1241-1274 compute the
  PFT irrigation-demand ratio from `transpot`, `evapot`, `precip_rain`,
  `vegstress_old`, `soil_deficit`, `irrig_threshold`, `irrig_fulfill`, and
  the `irrig_drip` switch. The helper preserves the Fortran `jv=2,nvm` loop,
  drip/flooding branch formulas, and row normalization by the positive demand
  sum.
- `hydrol_routing_soil_water_split`: `hydrol_soil` lines 5661-5751 map routing
  `returnflow`/`reinfiltration` and scalar `irrigation` into soil-tile inputs.
  The helper keeps `returnflow_soil=0`, divides top reinfiltration by `vegtot`,
  routes active irrigation to the one-based `pref_soil_veg` crop tile, and
  raises `ValueError` for the Fortran `STOP "hydrol irrig"` case where active
  irrigation demand targets a non-crop soil tile.
- `hydrol_soil_surface_water_setup`: early `hydrol_soil` surface-flux setup at
  lines 5800-5847. It reduces incoming surface water
  (`water2infilt`, optional crop `irrigation_soil`, `reinfiltration_soil`,
  `precisol_ns`, and negative evaporation/sublimation terms) against positive
  extraction (`ae_ns` and `subsinksoil`), then returns updated
  `water2infilt`, `water2extract`/`flux_top`, `flux_infilt`, and the
  `ae_ns += subsinksoil` update for the active tile. It stops before
  `hydrol_soil_coef`, `hydrol_soil_infilt`, and runoff reinfiltration.
- `hydrol_soil_infilt_explicit`: `hydrol_soil` calls `hydrol_soil_infilt` at
  line 5864; subroutine `hydrol_soil_infilt` lines 7432-7590 fills the first
  layer immediately, advances infiltration layer by layer using the supplied
  conductivity profile, computes `qinfilt` and `ru_infilt`, and optionally
  returns the `check_cwrr2` water-balance residual. The helper requires exact
  `mc`, `flux_infilt`, `k`, `dz`, saturation/conductivity parameters, and
  `kfact_root`. Peat and freezing branches are explicit inputs; they are not
  inferred or defaulted.
- `hydrol_soil_coef_mineral_profile_from_tables`: full-profile mineral
  coefficient rebuild from the source-built CWRR tables. It covers
  `hydrol_soil_coef` freezing mineral lines 8248-8280 and non-freezing mineral
  lines 8290-8307, including the source distinction between freeze-branch
  `mc_used` and non-freeze total-`mc` conductivity evaluation. It deliberately
  excludes the peat table branch.
- `build_peat_cwrr_tables` and `hydrol_soil_coef_peat_profile_from_tables`:
  peat CWRR table construction from `hydrol_var_init` lines 4163-4168 and
  4249-4306, with constants from `constantes_soil_var.f90` lines 163-167,
  plus peat coefficient selection/scaling from `hydrol_soil_coef` lines
  8259-8266 and 8290-8297. These helpers are explicit; the caller must route
  them only when `peat_hydro` is true and the active soil tile is 4, 5, or 6.
- `hydrol_soil_tile_explicit_step`: a longer explicit-input adapter that
  composes the closed kernels in Fortran order: surface setup lines 5800-5847,
  optional mineral or peat `hydrol_soil_coef` line 5858, infiltration line 5864
  plus `hydrol_soil_infilt` lines 7432-7590, runoff reinfiltration update
  lines 5871-5880, optional second mineral or peat `hydrol_soil_coef` line
  5891, tridiagonal setup line 5896, and the already audited
  RHS/solve/post-solve lines 5953-6075. It can still accept explicit
  coefficients, but when given coefficient tables it preserves the source order
  by using the first coefficient call's `k` for infiltration and the second
  call's `a/b/d/k` for setup, solve, and bottom drainage. Optional flags also
  expose the audited post-solve over-MCS, negative-runoff, water-table, and
  under-MCR blocks. It does not choose soil tile or compute moisture-stress
  diagnostics.
- `hydrol_soil_all_tiles_explicit_step`: explicit all-soil-tile driver for the
  audited soil chain. It runs the tile loop from surface setup through
  post-solve water-table/under-MCR/runoff-to-peat/tide routing, then applies
  the post-loop `run2peat`/`run2man`/`wt_ab` reinjection and `tmc +=
  water2infilt`, covering source spans 5800-6296 and 6757-6787. It still
  requires exact split fluxes and state arrays as inputs.
- `hydrol_module_explicit_step`: module-level explicit assembly of
  `hydrol_vegupd_static_state`, `hydrol_canop_interception`,
  `hydrol_flood_reservoir`, `hydrol_split_soil_fluxes`, and the all-tile soil
  loop. It follows the `hydrol_main` call order but deliberately excludes the
  stateful `hydrol_tmc_update` migration and later moisture-stress diagnostics.
- `hydrol_module_outputs`: packages downstream HYDROL outputs, including
  `runoff_per_soil = ru_ns`, `drainage_per_soil = dr_ns`, `runoff2peat`,
  `tmc`, `tmc_soil`, and `wtd`, with output/aggregation provenance at lines
  6806-6815 and 7361-7362.
- `hydrol_soil_layer_diagnostics`: one-soil-tile layer diagnostics from
  `hydrol_soil` lines 6474-6562. It computes `sm`, `smt`, `smw`, `smf`,
  `sms`, mineral `*_tmp` thresholds, `sm_nostress`, and `soil_wet_ns` with
  explicit mineral/peat constants and `njsc`.
- `hydrol_water_stress_diagnostics`: one-soil-tile `us`, `humrelv`, and
  `vegstressv` diagnostics from `hydrol_soil` lines 6565-6725, including PFT1
  zeroing, absent-PFT zeroing, under-MCR zeroing, old/new water-stress formulas,
  and the optional `dyn_nroot_larix` branch when its required source inputs are
  provided explicitly.
- `hydrol_soilmoist_aggregates`: tile-weighted `soilmoist`,
  `soilmoist_liquid`, `mc_layh`, `mcl_layh`, `mc_layh_s`, and `mcl_layh_s`
  from `hydrol_soil` lines 6729-6741 and `hydrol_diag_soil` lines 9159-9195.
- `hydrol_stress_aggregates`: PFT-level `humrel` and `vegstress` aggregation
  from `hydrol_diag_soil` lines 9089-9107.
- `hydrol_shumdiag`: `shumdiag`, `shumdiag_perma`, `shumdiag_peat`,
  `shumdiag_croppeat`, and `shumdiag_man` from `hydrol_diag_soil` lines
  9199-9284.
- `hydrol_litter_top_diagnostics`: `tmc_litter`, litter threshold water
  contents, `soil_wet_litter`, `tmc_trampling`, `tmc_topgrass`,
  `mc_peat_above`, `mc_croppeat_above`, and `mc_man_above` from
  `hydrol_soil` lines 6414-6470 and 6793-6795, with the matching
  initialization formulas at lines 4418-4508.
- `hydrol_grid_flux_aggregates`: `ae_ns` masking, grid-cell `runoff`,
  `drainage`, `humtot`, and adjusted `vevapnu` from `hydrol_diag_soil` lines
  9006-9082.
- `hydrol_litter_grid_diagnostics`: `k_litt`, `litterhumdiag`, and
  `drysoil_frac` from `hydrol_diag_soil` lines 9112-9155. `k_litt` remains
  table-driven and requires source-built `k_lin`/`k_lin_peat` values or
  explicit equivalent inputs; it is not approximated from the current solved
  conductivity.
- `hydrol_module_diagnostics`: explicit module-level diagnostic adapter that
  composes the layer, stress, moisture, litter/topgrass, runoff/grid, and
  `shumdiag` helpers after `hydrol_module_explicit_step`. It requires
  caller-owned Fortran state such as `nroot`, `dh`, `soiltile`, `vegtot`,
  `njsc`, thresholds, CWRR tables for `k_litt`, and optional dynamic root
  controls instead of synthesizing them. When supplied with an explicit
  `hydrol_alt_residual_solve_step` result plus `tmcint` and `evapot`, it can
  also expose the later `evap_bare_lim_ns` diagnostic without mutating
  prognostic `mc/mcl/tmc`.
- `hydrol_evap_bare_limit_diagnostic`: bare-soil evaporation limitation beta
  from the alternate residual dummy solve, `hydrol_soil` lines 7110-7163. It
  maps `tmcint - tmc_dummy - flux_bottom` through `mask_soiltile`,
  `frac_bare_ns`, `vegtot`, `evapot`, litter wilting/residual thresholds, the
  `do_rsoil` branch, clipping, and final `is_under_mcr` zeroing. It preserves
  the source restore semantics at lines 7165-7174 by returning diagnostics only.

Both helpers take Fortran module state explicitly (`vegtot`,
`throughfall_by_pft`, `precisol`, `subsinksoil`) and return updated arrays.
They do not default missing snow melt, flood, or canopy state. In particular,
`tot_melt` must come from an exact explicit-snow/bucket-snow source or trace.
`hydrol_split_soil_fluxes` likewise takes `ae_ns`, `evap_bare_lim_ns`,
`frac_bare_ns`, `humrelv`, `us`, `vegetmax_soil`, `vegtot`, and
`pref_soil_veg` explicitly. It does not synthesize `rootsink` or any later
`hydrol_soil` surface forcing.

## Audited Main Path

Main post-tridiagonal sequence in `hydrol_soil`:

- Tridiagonal call: line 6004.
- Bottom drainage before correction: lines 6007-6016.
- Liquid total after solve `tmcf`: lines 6018-6025.
- Transpiration sum and redistribution residual `check_tr_ns`: lines
  6027-6033.
- Drainage numerical correction: lines 6042-6047; optional recheck lines
  6050-6052.
- Post-solve total moisture update from liquid `mcl`: lines 6063-6075.
- Over-saturation correction call and routing: lines 6076-6092.
- Negative runoff transfer to drainage: lines 6102-6114.
- Optional water-table saturation forcing: lines 6116-6160.
- Water-table depth and under-residual smoothing call: lines 6162-6186.
- Recompute total moisture `tmc`: lines 6300-6313.
- Recompute `mcl` from updated `mc`: lines 6316-6328.
- Diagnostic moisture/stress section starts at lines 6395-6805.

Alternate residual-boundary path:

- Residual boundary solve call: line 7042.
- Alternate post-solve total moisture update from liquid `mcl`: lines
  7044-7058.
- Bottom flux for water budget: lines 7062-7066.
- Recompute `mc` and `tmc` for top-flux budget: lines 7078-7108.
- Bare-soil evaporation limit diagnostic: lines 7110-7163.

## Closed Kernels Added In Phase 1H

Implemented in `jax_orchidee/sechiba/hydrol.py`:

- `hydrol_layer_moisture_content`: source weights from `hydrol_soil`, lines
  6473-6521; also reused by `tmci/tmcf/tmc` formulas at lines 5923-5929,
  6018-6025, and 6300-6313.
- `hydrol_total_moisture_content`: column sum for `tmci`, `tmcf`, and `tmc`;
  source line spans above.
- `hydrol_mc_to_mcl`: mineral/non-peat liquid reconstruction from total
  moisture, lines 5899-5908 and 6316-6328.
- `hydrol_mcl_to_mc_after_solve`: mineral/non-peat total moisture update after
  solve, lines 6063-6075, 7044-7058, and 7078-7092.
- `hydrol_liquid_redistribution_check`: `check_tr_ns`, lines 6027-6033 and
  6050-6052.
- `hydrol_soil_clip_over_mcs2`: direct over-saturation clipping and correction
  water amount from `hydrol_soil_smooth_over_mcs2`, lines 7935-8026. Routing
  to runoff or drainage remains outside this helper.
- `hydrol_route_over_mcs2`: source-order routing around
  `hydrol_soil_smooth_over_mcs2`, lines 6076-6092. It sends the clipped excess
  to drainage only when `free_drain_coef >= 0.5` and `ok_freeze_cwrr` is false;
  otherwise it remains a runoff correction.
- `hydrol_correct_negative_runoff`: negative `ru_ns` transfer into drainage,
  lines 6102-6114.
- `hydrol_force_water_table_saturation`: optional saturation forcing below
  prescribed `zwt_force`, lines 6116-6158. It preserves the source activation
  test on `zwt_force(1,jst) <= zmaxh` and returns `dmc`, `dr_force_ns`, and
  updated `dr_ns`.
- `hydrol_effective_water_table_depth`: effective water-table depth diagnostic
  from lines 6162-6180, walking upward from the bottom through contiguous
  saturated nodes and supporting peat tile saturation.
- `hydrol_soil_smooth_under_mcr`: under-residual smoothing and
  `is_under_mcr` diagnosis from lines 7623-7770, including explicit peat tile
  threshold handling and `mask_soiltile`.
- `hydrol_after_under_mcr_runoff_peat_tide_routing`: per-soil-tile
  runoff-to-peat/mangrove and above-surface reservoir routing from lines
  6183-6296. It preserves the source's per-`jst` reset of `runoff2peat`.
- `hydrol_runoff_peat_post_loop_reinjection`: post-soiltile loop reinjection
  of `run2peat`, `run2man`, and `wt_ab` into `water2infilt`, plus `tmc_soil`
  and `tmc += water2infilt`, lines 6757-6787.
- `hydrol_soil_all_tiles_explicit_step`: all-tile explicit driver that calls
  the closed per-tile kernels and runoff/tide post-loop reinjection in source
  order. It is the first HYDROL soil-loop assembly layer; moisture diagnostics
  and `hydrol_diag_soil` grid/litter outputs are now available through the
  explicit post-step `hydrol_module_diagnostics` adapter. The later alternate
  residual evap-bare-limit re-solve is computed as same-step diagnostic state
  and does not replace or mutate the primary prognostic solve.
- `hydrol_alt_residual_solve_step`: explicit alternate residual-boundary solve
  slice from lines 6971-7108. It builds the no-rootsink RHS, runs the first
  solve, triggers the residual top boundary when `mcl(:,1) < mcr_eff` and
  `flux_top > min_sechiba`, runs the guarded second solve, reconstructs `mc`,
  and returns bottom flux and `tmc`.
- `hydrol_module_explicit_step` and `hydrol_module_outputs`: higher-level
  HYDROL assembly and output packaging for the source-backed path now covered
  by local kernels.
- `hydrol_soil_coef_mineral_profile_from_tables`: mineral profile coefficient
  selector/scaler from `hydrol_soil_coef`, lines 8248-8280 and 8290-8307.
- `build_peat_cwrr_tables` and `hydrol_soil_coef_peat_profile_from_tables`:
  peat table builder and profile coefficient selector from `hydrol_var_init`,
  lines 4163-4168 and 4249-4306, and `hydrol_soil_coef`, lines 8259-8266 and
  8290-8297.

These kernels require explicit inputs. They do not choose PFT, peat flags,
branch activation, fluxes, or restart/default values. The module assembly layer
is now available for the audited source path, but it remains an explicit
function: caller-owned state migration, restart initialization, and diagnostics
are not hidden inside it.

## Blocked Stateful Pieces

Not implemented as full state wiring:

- Full `hydrol_main` state driver with restart/state migration. The explicit
  module assembly now wires the audited kernels, but `hydrol_tmc_update`,
  restart defaults, and external forcing/state ownership remain outside it.
- The alternate residual branch is implemented as `hydrol_alt_residual_solve_step`
  and its `evap_bare_lim_ns` conversion is implemented as
  `hydrol_evap_bare_limit_diagnostic`. The all-tile driver computes the
  dummy solve and `tmcint` in source order, and `hydrol_module_diagnostics`
  refuses to expose `evap_bare_lim_ns` unless the caller supplies same-step
  `tmcint` and ENERBIL `evapot`. Source lines 6971-7163 remain a diagnostic
  re-solve, not a drop-in replacement for the primary vegetation/root water
  redistribution solve.
- Cross-step wiring of the newly diagnosed `evap_bare_lim_ns` into the next
  DIFFUCO call remains a SECHIBA driver validation item. It must consume the
  diagnosed field explicitly rather than falling back to restart-era values.

## Minimal Trace/State Needed Next

Before full `hydrol_soil` state update can be wired, the next Fortran trace
should add, per active `(kjit, ji, jst, jsl)`:

- `mc_before_solve`, `mcl_before_solve`, `profil_froz_hydro_ns`, `mcr`,
  `mcs`, `njsc`, `mask_soiltile`, `resolv`.
- `mcl_after_tridiag`, `tmci`, `tmcf`, `rootsink_by_layer`,
  `rootsink_sum`, `flux_top`, `dr_ns_before_corr`, `check_tr_ns`,
  `dr_corrnum_ns`, `dr_ns_after_corr`.
- `mc_after_mcl_update`, `mc_after_over_mcs2`, `ru_corr_ns`, `dr_corr_ns`,
  `ru_ns_after_over_mcs2`, `dr_ns_after_over_mcs2`.
- `is_under_mcr`, `check_under_ns`, `mc_after_under_mcr`, final `mc`,
  final `mcl`, and final `tmc`.
- Alternate residual branch trigger fields from lines 7011-7018 and
  post-alternate `mcl/mc` from lines 7044-7058, even when inactive.

For remaining HYDROL validation, add source trace coverage for:

- The `hydrol_module_diagnostics` adapter outputs, including `tmc_litter`,
  `soil_wet_litter`, `tmc_trampling`, `tmc_topgrass`, `mc_peat_above`,
  `mc_croppeat_above`, `mc_man_above`, `k_litt`, `litterhumdiag`,
  `drysoil_frac`, `runoff`, `drainage`, `humtot`, and adjusted `vevapnu`.
- Runtime scheduling evidence for the later `evap_bare_lim_ns` alternate
  residual diagnostic path from lines 7110-7163, especially the exact `tmcint`,
  `mcint/mclint`, and same-step `evapot` state used before restoration.
