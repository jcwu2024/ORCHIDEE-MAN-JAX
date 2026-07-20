# Routing PFT14 Branch Ledger

Scope: source-driven audit for the SECHIBA routing boundary used when
`river_routing .AND. nbp_glo > 1`. Single-point no-routing is already covered
elsewhere; the paper-scale output is currently understood as a mosaic of 669
independent single-landpoint cases, so this ledger tracks only the separate
true multi-landpoint/global-routing path.

## Closed Local Boundary

- `routing_accumulator_step` covers `routing.f90::routing_main` lines 912-1007:
  accumulation of `floodout_mean`, water `precip_mean`/`runoff_mean`/
  `drainage_mean`, `temp_sol_mean`, `transpot_mean`, `totnobio_mean`,
  `k_litt_mean`, `humrel_mean`, `vegtot_mean`, `time_counter`, and the
  pre-routing zeroing of `returnflow`, `reinfiltration`, `irrigation`,
  `riverflow`, `coastalflow`, sediment/POC deposition, stream diagnostics,
  `flood_res`, and `fastr`.
- With `ok_doc=True`, the same helper covers `routing.f90` lines 920-947:
  DOC/CO2 additions from `DOC_EXP_agg`, erosion DOC from `DOC_ERO_agg`, POC
  additions from `POC_EXP_agg`, sediment additions from `SED_EXP_agg`, zeroing
  precipitation/drainage POC+sediment slots, and NaN-to-zero cleanup.
- At the daily gate, it covers `routing.f90` lines 1014-1029: `flow_input`
  construction from accumulated runoff/drainage/precipitation and
  `flood_inp`/`stream_inp` splitting by `flood_frac/(flood_frac+streamfl_frac)`.
- `routing_lake_step` covers `routing.f90::routing_lake` lines 5390-5448:
  lake inflow addition, optional `doswamps` returnflow, proportional non-water
  species return when `ok_doc` is true, reservoir writeback, `return_lakes`,
  `lake_diag`, and remaining `lakeinflow` scaling by total routing area.
- `routing_reservoir_outflow_step` covers `routing.f90::routing_flow` lines
  3253-3330 and 3352-3388: upstream stream-water partitioning of
  `qflow_avebas`/`stream_resavebas`/`stream_damavailbas`, fast/slow/stream
  water outflow limits for active routed basins, zero flow for inactive
  `route_tobasin <= 0`, and non-water concentration outflow/writeback for
  fast/slow reservoirs plus stream DOC/CO2 slots through `ico2aq`.
- `routing_flood_pond_input_step` covers `routing.f90::routing_flow` lines
  3558-3681: floodplain/pond vertical water balance, tiny particulate
  floodplain-store deposition, and basin distribution of flood/stream DOC and
  CO2 inputs.
- `routing_daily_scaled_outputs` covers `routing.f90::routing_main` lines
  1115-1135 and 1139-1140: converting daily routing means back to
  `dt_sechiba`, converting river/coastal/hydrograph water volume by `mille`,
  scaling `slowflow_diag`, and adding river-bed-to-floodplain sediment/POC
  transfers into total deposition outputs.
- `routing_return_reinfiltration_step` covers `routing.f90::routing_flow`
  lines 4940-4970: returnflow from swamps and reinfiltration from pond/flood
  drainage are summed across basins only for water/DOC/CO2 slots through
  `ico2aq`, divided by total routing area when any floodplain/swamp/pond
  switch is active, and otherwise remain zero.
- `routing_flow_diagnostics_step` covers `routing.f90::routing_flow` lines
  5219-5329: netflow diagnostics, end-of-day reservoir diagnostics,
  hydrograph/slowflow selection by `hydrodiag`, irrigation/deposition
  aggregation, river-bed POC/DOC/CO2 clipping, lake/coastal/river outlet slot
  copies from `transport`, `flood_daily`, `flood_res`, and `fastr`.
- `routing_lake_overflow_step` covers `routing.f90::routing_flow` lines
  5331-5366: lake storage capping by `max_lake_reservoir * totarea`,
  concentration-proportional non-water overflow, lake reservoir writeback,
  global overflow summation, and uniform coastal redistribution through
  `mask_coast`/`nb_coast_gridcells`.
- `routing_reservoir_update_step` covers `routing.f90::routing_flow` lines
  4208-4372: reservoir writeback from runoff/drainage/transport/flood/pond
  terms, `streamb_inflow`, POC/sediment clearing in `return_swamp`, negative
  water-storage correction cascading from flood to stream to fast to slow
  reservoirs, explicit slow-reservoir failure, and post-update `totflood`.
- `routing_area_fractions_step` covers `routing.f90::routing_flow` lines
  4374-4500: stream swelling area, basin partitioning by square-root stream
  storage, `streamfl_frac`/`streamfl_frac_bas`, optional river-bed deposition
  limiting, `flood_frac`/`flood_frac_bas`, `flood_height`, and pond contribution
  to the combined flood fraction. It intentionally preserves the source
  behavior at lines 4438-4442 where POC river-bed transfer overwrites the
  three-column `rivbed2fld_sed` assignment and `rivbed2fld_poc` is untouched.
- `routing_pond_flux_step` covers `routing.f90::routing_flow` lines
  3924-3993: pond drainage water limits, DOC/CO2 drainage by pond
  concentration, `reinf_slope` pond inflow from fast flow, POC/sediment
  transfer to floodplain deposition, POC/sediment zeroing in pond inflow, and
  the disabled-pond reset branch.
- `routing_swamp_flood_step` covers `routing.f90::routing_flow` lines
  4074-4206: `new_flood_scheme` stream 50th-percentile partitioning and
  limited outflow, optional swamp returnflow from excess transport with
  POC/sediment deposition and POC/sediment zeroing in `return_swamp`, floodplain
  inflow proportional to the floodplain/stream area split, and the old-scheme
  zeroing of `floods`/`flood_reservoir`.
- `routing_floodplain_flux_step` covers `routing.f90::routing_flow` lines
  3686-3915: floodplain drainage water limits, DOC/CO2 drainage by floodplain
  concentration, POC/sediment deposition during drainage, lateral floodplain
  outflow by `flood_tcst`, source-order POC/sediment deposition and flow,
  10%-of-water particulate caps, inactive-route particulate clearing, disabled
  floodplain resets, and adding `flood_flow` to `stream_reservoir`.
- `routing_transport_between_basins_step` covers `routing.f90::routing_flow`
  lines 4003-4061: downstream `transport(route_togrid, route_tobasin, :)`
  accumulation from `fast_flow + slow_flow + stream_flow` and optional
  `check_riverbal` water/carbon balance diagnostics.
- `routing_stream_erosion_step` covers `routing.f90::routing_flow` lines
  3390-3539, with `nlybiotur` setup from lines 3202-3206: Method-3 sediment
  transport capacity, excess suspended sediment deposition, POC following the
  clay fraction, re-erosion from stored river-bed deposits, channel/bank
  erosion when deposits are insufficient, `carbon_32l` writeback for vegetated
  PFTs, stream dam-availability update, and end-of-block stream-water
  subtraction.
- `routing_poc_decomposition` covers `routing.f90::poc_decomposition` lines
  11688-11737: POC decay, DOC/CO2 partitioning, CUE-retained cross-pool
  transfers, and POC reservoir writeback.
- `routing_co2_chemistry_step` covers `routing.f90::routing_flow` lines
  4502-4929: water temperature, CO2 solubility, Schmitt number, DOC
  temperature-dependent decay, POC decay in flood/stream/fast/pond reservoirs,
  river-bed POC decay into stream DOC/CO2, segmented flood/stream/pond CO2
  evasion, instantaneous fast-reservoir CO2 evasion, slow-reservoir DOC decay
  without evasion, inactive flood/stream return/deposition branches, basin pCO2
  water-weighting, and atm-to-micro-atm conversion.
- `routing_irrigation_step` covers `routing.f90::routing_flow` lines
  4975-5205: net irrigation requirement, basin-level stream/fast/slow
  withdrawals, transported-matter scaling after water withdrawal, irrigation
  deficit tracking, same-grid adduction from the fullest stream basin,
  source-order same-grid adduction assignment, and sub-100 km neighbor-grid
  adduction with transported-matter transfer.
- `routing_flow_step` covers a conservative source-order wrapper around the
  audited `routing_flow` kernels above. It accepts Fortran 1-based route arrays
  and outlet codes, converts them only at the local transport boundary, and is
  locally covered for the disabled-surface-switch outlet path, active
  floodplain/pond composition, lake overflow/coastal redistribution, and
  irrigation demand using the Fortran `precip` input rather than `flood_inp`,
  new-flood-scheme swamp/floodplain routing, active stream DOC/CO2 chemistry,
  neighbor irrigation adduction, river-balance diagnostics, and broader
  combined active-state writeback through the daily boundary. Remaining
  wrapper work is production coupling to complete routing map assets, not a
  source-kernel gap.
- `routing_daily_boundary_step` covers the `routing_main` daily gate in
  `routing.f90` lines 1014-1140: accumulator due gating, source-order
  `routing_flow` call, `routing_lake` call, addition of lake returnflow,
  daily accumulator reset, and `dt_sechiba/dt_routing` scaling of routing
  outputs back to the SECHIBA timestep.
- `routing_setvar_no_keyword` covers `sechiba_io.f90` lines 244-441 for the
  `NO_KEYWORD` restart fallback behavior used by `routing_init`: arrays are
  defaulted only when all elements equal `val_exp`; partial exceptional values
  are preserved.
- `routing_init_restart_defaults` covers `routing.f90::routing_init` lines
  2163-2459 and 2530-2710 for restart-field ownership after I/O: reservoir and
  daily-mean fallback defaults, `doirrigation`-conditioned
  `irrigation_mean`, `stream_seddep(:,:,ih2o:ico2aq)=0`,
  `drainage_mean`/`precip_mean` zeroing above `ico2aq`, map/climatology fields
  defaulting to `undef_sechiba`, diagnostic reservoir scaling by total routing
  area, and zero initialization of CO2/input diagnostics.
- `routing_initialize_map_flags` covers `routing.f90::routing_initialize`
  lines 594-621: `do_irrigation`/`do_floodplains`/`doswamps`/`ok_doc`
  conditioned map reinitialization flags, including the source rule that
  nonpositive `streamr50th` under floodplains forces both floodplain and stream
  surface initialization.
- `routing_irrigmap_preprocess` covers `routing.f90::routing_irrigmap` lines
  10885-10911: percent-to-fraction conversion and source thresholds for
  irrigation, flood-type fractions, headwaters, and stream surfaces while
  preserving missing values.
- `routing_irrigmap_aggregate` covers `routing.f90::routing_irrigmap` lines
  11012-11204 after `aggregate_p`: negative `irrsub_area` clearing, source
  order accumulation over the first `COUNT(irrsub_area > 0)` slots, grid-cell
  area caps, floodplain composition from flood+dam+saline maps, swamp
  extraction, stream-surface writeback, `floodplains=max(floodplains,
  stream_area+min_sechiba)`, stream-reservoir threshold switches,
  unweighted `qflow_ave`/`stream_resave` sums, and the
  `ok_damreservoir` fallback to `un`.
- `routing_initialize_stream_fraction` covers `routing.f90::routing_initialize`
  lines 641-647: stream fraction is `min(stream_area/sum(routing_area),1)` and
  is zero when total routing area is zero.
- `read_routing_map_fields` covers the strict NetCDF input contract in
  `routing.f90::routing_basins` lines 7108-7183: longitude/latitude fields
  and required `trip`, `basins`, `topoind`, `gravel`, and `drainage_area`
  variables are read in Fortran `(longitude, latitude)` order. Missing source
  variables are explicit errors; no gravel/drainage defaults are synthesized.
- `routing_map_fields_for_domain` covers the SECHIBA branch guard for routing
  map ownership: `sechiba.f90` lines 1227-1234 enter routing only when
  `river_routing .AND. nbp_glo > 1`. Single-landpoint paper runs therefore do
  not read `routing.nc` even when `RIVER_ROUTING=TRUE`; active multi-landpoint
  routing enforces the strict reader contract above.
- `routing_sortcoord` covers `routing.f90::routing_sortcoord` lines
  7728-7813: duplicate coordinate compression, longitude periodic sorting
  when absolute longitude exceeds 160 degrees, west-east/east-west and
  north-south/south-north ordering, and zero tail fill after compression.
- `routing_hierarchy` covers `routing.f90::routing_hierarchy` lines
  8539-8627: direction following over the high-resolution routing map,
  first-dimension wraparound, cumulative topographic-index hierarchy,
  undefined-cell preservation, large-topography/decreasing-hierarchy guards,
  and the source loop-limit failure condition.
- `routing_getgrid_from_subgrid` covers `routing.f90::routing_getgrid` lines
  7547-7709 after `aggregate`: source-cell ordering into the local coarse-box
  grid, value transfer for `trip`/basin/topographic/gravel/drainage/hierarchy
  fields, invented coastal cells for `sub_pts==0`, boundary outflow tagging
  as 101-108, and source-order corner/edge diagonal simplification.
- `routing_findbasins_simple` covers the non-splitting subset of
  `routing.f90::routing_findbasins` lines 7862-8121: initial basin grouping,
  singleton river-ocean cells converted to coastal flow, merging multiple
  singleton ocean cells into the synthetic coastal basin `-1`, source outflow
  direction retention (`trip-100` for outflow markers), size sorting, and the
  final outflow-count sanity check. Multi-cell basins with multiple outflows
  are explicitly guarded because the source then calls `routing_simplify` and
  `routing_cutbasin`, which remain separate open topology kernels.
- `routing_findrout` covers `routing.f90::routing_findrout` lines 8647-8758:
  outflow-point discovery from `trip > 9`, tracing positive trip directions
  1-8 to each outflow, upstream counts per outflow point, cycle detection, and
  the source basin-size conservation check.
- `routing_simplify` covers `routing.f90::routing_simplify` lines 8143-8311:
  isolation of a single local basin, duplicate same-border outflow detection
  for 101/103/105/107, source-order redirection of the largest-hierarchy
  outflow sub-basin toward a neighboring sub-basin, and in-place `trip`
  writeback.
- `routing_cutbasin` covers `routing.f90::routing_cutbasin` lines 8331-8519:
  local basin isolation, optional one-cell boundary outflow absorption when
  the basin count is already at `nbasmax`, split by `routing_findrout`
  outflow assignment, emitted new basin names/sizes/points, and lost-point
  guard.
- `routing_globalize_one_grid` covers `routing.f90::routing_globalize` lines
  8780-8934 for one grid cell in the normal `routing_basins` call order:
  basin ID insertion, coastal synthetic-basin member transfer, area sums,
  averaged `topoind`/gravel/drainage, OUTP hierarchy selection, negative
  outflow `min_topoind` reset, and positive outflow-direction mapping through
  `neighbours`.
- `routing_linkup` covers `routing.f90::routing_linkup` lines 8957-9413:
  direct target-grid basin matching, hierarchy and same-direction acceptance,
  +/- one-neighbor fallback, same-grid fallback, coastal fallback, inflow table
  updates, and final target-basin existence checks.
- `routing_fetch` covers `routing.f90::routing_fetch` lines 9435-9555: basin
  area normalization to grid land area, upstream fetch accumulation along
  outflow chains, conversion of existing river outlets to coastal outlets, and
  marking the `num_largest` coastal outlets as river outlets.
- `routing_killbas` covers `routing.f90::routing_killbas` lines 10091-10256:
  killed-basin merge into a takeover basin, area/topo/gravel/drainage/fetch
  updates, downstream fetch reallocation, inflow redirection, downstream inflow
  list removal, basin-field shifting, and basin-count decrement.
- `routing_truncate_reduce_to_nbasmax` covers the source selection order in
  `routing.f90::routing_truncate` lines 9652-9912 for reducing basin count:
  coastal/return outlet merges, same-outflow-grid merges, same-basin-ID merges,
  and the fallback hammer path via `routing_killbas`.
- `routing_truncate_finalize_no_reduction` covers `routing.f90::routing_truncate`
  lines 9917-10067 after basin counts are already within `nbasmax`: definitive
  route-array writeback, negative outflow convention conversion to
  `nbasmax+1/2`, route consistency checks, grid-area correction of
  `routing_area`, and largest-outlet river marking. It preserves the source
  condition that any outlet code `> nbasmax`, including return flow, can be
  selected by the largest-outlet pass.
- `routing_basins_post_aggregate` covers the post-`aggregate` topology chain
  in `routing.f90::routing_basins` lines 7277-7501 by composing
  `routing_hierarchy`, `routing_getgrid`, the guarded non-splitting
  `routing_findbasins` subset, `routing_globalize`, `routing_linkup`,
  `routing_fetch`, and `routing_truncate`. It is locally covered for an
  invented coastal cell and a real two-grid neighbor-to-coast chain with
  internally computed hierarchy. It intentionally does not claim `aggregate_p`
  geometric ownership, NetCDF routing/floodplain readers, or the full
  production `routing_basins` I/O wrapper.
- `routing_basins_from_map_fields` covers the complete in-memory
  `routing_basins` regular-lon/lat handoff after strict NetCDF field loading:
  source mask construction from finite `trip`, `aggregate_2d` overlap
  ownership via `driver.static.aggregate_2d_overlap`, source-derived
  `min_topoind`/`invented_basins`, and delegation into the post-aggregate
  topology chain. It is locally covered on a complete synthetic routing map.
  It still depends on complete routing map fields and therefore does not
  bypass the missing `gravel`/`drainage_area` variables in the local
  `routing.nc`.

Tests:

- `tests/unit/test_routing.py::test_routing_accumulator_before_daily_gate_updates_means_and_keeps_outputs_zero`
- `tests/unit/test_routing.py::test_routing_accumulator_daily_gate_builds_flow_input_and_doc_splits`
- `tests/unit/test_routing.py::test_routing_accumulator_ok_doc_refuses_short_flow_axis`
- `tests/unit/test_routing.py::test_routing_lake_step_returns_water_and_doc_by_lake_concentration`
- `tests/unit/test_routing.py::test_routing_lake_step_without_swamps_only_updates_reservoir_and_scaled_diagnostics`
- `tests/unit/test_routing.py::test_routing_reservoir_outflow_step_matches_source_water_and_concentration_rules`
- `tests/unit/test_routing.py::test_routing_reservoir_outflow_step_zeroes_inactive_route_basins`
- `tests/unit/test_routing.py::test_routing_flood_pond_input_step_applies_vertical_flux_and_basin_inputs`
- `tests/unit/test_routing.py::test_routing_daily_scaled_outputs_match_fortran_unit_conversion`
- `tests/unit/test_routing.py::test_routing_return_reinfiltration_step_sums_only_water_doc_co2_slots`
- `tests/unit/test_routing.py::test_routing_return_reinfiltration_step_zero_when_all_surface_switches_disabled`
- `tests/unit/test_routing.py::test_routing_flow_diagnostics_step_matches_fortran_gridcell_aggregation`
- `tests/unit/test_routing.py::test_routing_lake_overflow_step_caps_lakes_and_adds_global_overflow_to_coast`
- `tests/unit/test_routing.py::test_routing_lake_overflow_step_keeps_coastalflow_when_no_coastal_cell`
- `tests/unit/test_routing.py::test_routing_reservoir_update_step_applies_source_writeback_and_totflood`
- `tests/unit/test_routing.py::test_routing_reservoir_update_step_cascades_negative_water_like_fortran`
- `tests/unit/test_routing.py::test_routing_reservoir_update_step_refuses_materially_negative_slow_reservoir`
- `tests/unit/test_routing.py::test_routing_area_fractions_step_matches_stream_swell_and_flood_formulas`
- `tests/unit/test_routing.py::test_routing_area_fractions_step_preserves_source_rivbed_limit_assignment`
- `tests/unit/test_routing.py::test_routing_pond_flux_step_matches_drainage_inflow_and_deposition_rules`
- `tests/unit/test_routing.py::test_routing_pond_flux_step_zeroes_pond_state_when_disabled`
- `tests/unit/test_routing.py::test_routing_swamp_flood_step_matches_new_scheme_return_and_flooding`
- `tests/unit/test_routing.py::test_routing_swamp_flood_step_old_scheme_zeroes_floods_and_reservoir`
- `tests/unit/test_routing.py::test_routing_floodplain_flux_step_matches_drainage_lateral_flow_and_caps`
- `tests/unit/test_routing.py::test_routing_floodplain_flux_step_zeroes_when_floodplains_disabled`
- `tests/unit/test_routing.py::test_routing_floodplain_flux_step_inactive_route_zeroes_particulate_store_like_source`
- `tests/unit/test_routing.py::test_routing_transport_between_basins_step_routes_combined_outflow_to_targets`
- `tests/unit/test_routing.py::test_routing_transport_between_basins_step_updates_optional_balance_diagnostics`
- `tests/unit/test_routing.py::test_routing_transport_between_basins_step_refuses_out_of_range_target`
- `tests/unit/test_routing.py::test_routing_flow_step_wires_disabled_surface_sequence_and_fortran_outlet_route`
- `tests/unit/test_routing.py::test_routing_flow_step_composes_active_floodplain_and_pond_branches`
- `tests/unit/test_routing.py::test_routing_flow_step_applies_lake_overflow_after_diagnostics`
- `tests/unit/test_routing.py::test_routing_flow_step_requires_fortran_one_based_routes`
- `tests/unit/test_routing.py::test_routing_flow_step_irrigation_uses_precip_not_flood_input`
- `tests/unit/test_routing.py::test_routing_flow_step_composes_new_flood_scheme_swamp_and_flood_branches`
- `tests/unit/test_routing.py::test_routing_flow_step_composes_doc_chemistry_active_stream_path`
- `tests/unit/test_routing.py::test_routing_flow_step_composes_neighbor_irrigation_adduction`
- `tests/unit/test_routing.py::test_routing_flow_step_wires_river_balance_diagnostics`
- `tests/unit/test_routing.py::test_routing_daily_boundary_before_gate_only_accumulates_and_returns_zero_outputs`
- `tests/unit/test_routing.py::test_routing_daily_boundary_due_runs_flow_lake_scales_outputs_and_resets_accumulators`
- `tests/unit/test_routing.py::test_routing_daily_boundary_combined_active_state_writebacks_remain_coherent`
- `tests/unit/test_routing.py::test_routing_stream_erosion_step_deposits_excess_sediment_and_poc_like_source`
- `tests/unit/test_routing.py::test_routing_stream_erosion_step_reerodes_bed_deposits_with_poc_following_clay`
- `tests/unit/test_routing.py::test_routing_stream_erosion_step_erodes_banks_and_updates_soil_carbon`
- `tests/unit/test_routing.py::test_routing_poc_decomposition_matches_source_pool_transfers`
- `tests/unit/test_routing.py::test_routing_co2_chemistry_fast_slow_and_riverbed_branches`
- `tests/unit/test_routing.py::test_routing_co2_chemistry_flood_and_pond_segmented_evasion`
- `tests/unit/test_routing.py::test_routing_co2_chemistry_inactive_flood_and_stream_return_stores`
- `tests/unit/test_routing.py::test_routing_irrigation_step_disabled_leaves_reservoirs_and_fluxes_zero`
- `tests/unit/test_routing.py::test_routing_irrigation_step_withdraws_stream_fast_slow_and_scales_matter`
- `tests/unit/test_routing.py::test_routing_irrigation_step_same_grid_adduction_uses_fullest_stream_basin`
- `tests/unit/test_routing.py::test_routing_irrigation_step_neighbor_adduction_adds_remote_stream_water`
- `tests/unit/test_routing.py::test_routing_setvar_no_keyword_replaces_only_all_missing_arrays`
- `tests/unit/test_routing.py::test_routing_init_restart_defaults_match_source_cold_start_groups`
- `tests/unit/test_routing.py::test_routing_init_restart_defaults_preserve_restart_values_and_derive_diagnostics`
- `tests/unit/test_routing.py::test_routing_initialize_map_flags_match_missing_restart_rules`
- `tests/unit/test_routing.py::test_routing_irrigmap_preprocess_thresholds_percent_maps_and_preserves_missing`
- `tests/unit/test_routing.py::test_routing_irrigmap_aggregate_matches_source_flags_and_area_rules`
- `tests/unit/test_routing.py::test_routing_irrigmap_aggregate_zeroes_negative_subarea_before_source_order_count`
- `tests/unit/test_routing.py::test_routing_irrigmap_aggregate_preserves_noninitialized_outputs_and_dam_switch`
- `tests/unit/test_routing.py::test_routing_initialize_stream_fraction_uses_stream_area_only_and_caps`
- `tests/unit/test_routing.py::test_read_routing_map_fields_requires_source_variables_and_transposes_to_fortran_order`
- `tests/unit/test_routing.py::test_read_routing_map_fields_refuses_missing_fortran_variables`
- `tests/unit/test_routing.py::test_routing_map_fields_for_domain_reads_only_when_global_routing_branch_active`
- `tests/unit/test_routing.py::test_routing_sortcoord_compresses_duplicates_and_uses_periodic_longitudes`
- `tests/unit/test_routing.py::test_routing_hierarchy_accumulates_topo_and_wraps_longitude_like_source`
- `tests/unit/test_routing.py::test_routing_hierarchy_refuses_unrouted_cycle`
- `tests/unit/test_routing.py::test_routing_getgrid_from_subgrid_orders_cells_and_tags_boundary_outflows`
- `tests/unit/test_routing.py::test_routing_getgrid_from_subgrid_invents_coastal_cell_when_no_source_points`
- `tests/unit/test_routing.py::test_routing_findbasins_simple_collects_single_outflow_basins_by_size`
- `tests/unit/test_routing.py::test_routing_findbasins_simple_merges_singleton_ocean_points_into_coastal_basin`
- `tests/unit/test_routing.py::test_routing_findbasins_simple_refuses_unimplemented_multi_outflow_topology`
- `tests/unit/test_routing.py::test_routing_findrout_follows_trip_paths_to_each_outflow`
- `tests/unit/test_routing.py::test_routing_findrout_refuses_cycles_and_size_mismatch`
- `tests/unit/test_routing.py::test_routing_simplify_redirects_duplicate_border_outflow_to_neighbor_subbasin`
- `tests/unit/test_routing.py::test_routing_cutbasin_splits_local_basin_by_outflow_points`
- `tests/unit/test_routing.py::test_routing_globalize_one_grid_sums_basin_fields_and_maps_positive_outflow`
- `tests/unit/test_routing.py::test_routing_globalize_one_grid_transfers_coastal_basin_and_resets_negative_hierarchy`
- `tests/unit/test_routing.py::test_routing_linkup_direct_target_basin_updates_outflow_and_inflow_tables`
- `tests/unit/test_routing.py::test_routing_linkup_allows_equal_hierarchy_when_directions_are_adjacent`
- `tests/unit/test_routing.py::test_routing_linkup_falls_back_to_coastal_when_neighbor_is_ocean`
- `tests/unit/test_routing.py::test_routing_fetch_normalizes_area_accumulates_upstream_and_marks_largest_river`
- `tests/unit/test_routing.py::test_routing_fetch_refuses_outflow_cycles`
- `tests/unit/test_routing.py::test_routing_truncate_finalize_no_reduction_writes_route_arrays_and_scales_area`
- `tests/unit/test_routing.py::test_routing_truncate_finalize_no_reduction_refuses_unreduced_basin_counts`
- `tests/unit/test_routing.py::test_routing_killbas_merges_last_basin_into_takeover_and_reduces_count`
- `tests/unit/test_routing.py::test_routing_killbas_middle_shift_updates_downstream_inflow_references`
- `tests/unit/test_routing.py::test_routing_killbas_redirects_inflows_to_takeover_basin`
- `tests/unit/test_routing.py::test_routing_killbas_updates_fetch_on_new_and_old_downstream_paths`
- `tests/unit/test_routing.py::test_routing_killbas_redirects_sources_into_shifted_basin_number`
- `tests/unit/test_routing.py::test_routing_truncate_reduce_to_nbasmax_merges_smallest_coastal_into_largest`
- `tests/unit/test_routing.py::test_routing_basins_post_aggregate_invented_coastal_cell_runs_to_final_route_arrays`
- `tests/unit/test_routing.py::test_routing_basins_post_aggregate_links_real_source_cells_to_coastal_outlet`
- `tests/unit/test_routing.py::test_routing_basins_from_map_fields_aggregates_complete_map_then_links_topology`

## Still Open

- `routing_flow` reservoir network (`routing.f90` roughly lines 1079 and
  3180-5368): source kernels are locally covered and a conservative
  source-order `routing_flow_step` wrapper exists for disabled-surface outlet,
  floodplain/pond, lake-overflow, same-grid irrigation-demand,
  swamp/new-flood, active stream DOC chemistry, and neighbor irrigation
  adduction paths, plus river-balance diagnostics and broader active-state
  writeback through the daily boundary. The daily `routing_main` gate through
  `routing_lake` and unit scaling is locally covered. The remaining gap is
  production coupling to complete routing map assets. The source-order status
  is tracked in `routing_flow_sequence_ledger.md`.
- `routing_initialize`/`routing_init` restart and map ownership:
  `routing.f90` lines 442-722 and 1662 onward read/build basin topology,
  routing reservoirs, floodplain/irrigation/swamp/stream maps, and restart
  means. Restart fallback/default semantics, map reinitialization flags, and
  the post-`aggregate_p` irrigation/flood/swamp/stream map aggregation are
  locally covered. The strict routing-map reader contract and first
  post-`aggregate` topology extraction steps (`routing_hierarchy`/
  `routing_getgrid`) are also locally covered, and the non-splitting subset of
  `routing_findbasins` is covered with guards for the splitting cases. The
  downstream topology stack from `routing_findrout` through simplify/cut,
  linkup/fetch, and truncate reduction/finalization is now covered by local
  micro-cases, and the pure post-`aggregate` wrapper is covered for invented
  coastal and real neighbor-outlet cases. The in-memory complete-map wrapper
  now covers the regular-lon/lat `aggregate_2d` handoff. The remaining
  initialization gap before global routing is claimed is complete production
  file availability/wiring: the local `data/MICT_BIOE/Input/routing.nc`
  currently lacks the source-required `gravel` and `drainage_area` variables,
  so it is not silently usable for the production routing topology path.
- Optional routing switches are now locally represented in the routing ledger
  where their `routing_flow` internals are source-backed. Scenario ownership
  still requires explicit initialization/map/restart state before global
  routing parity can be claimed.

This is an asset/production-wiring gap, not a blocker for the paper workflow
that runs the 669 target cells as independent `nbp_glo=1` cases. In that
workflow, `sechiba.f90` takes the no-routing zero branch even when
`RIVER_ROUTING=TRUE`, because `routing_main` is guarded by
`river_routing .AND. nbp_glo > 1`.

## Rule For Future Work

Do not use the 669 independent single-landpoint mosaic as proof of a true
`nbp_glo>1` routing run. For multi-landpoint/global-routing runs, first close
the restart/map state and reservoir network with micro-cases, then use annual
modelout only as a validation gate.
