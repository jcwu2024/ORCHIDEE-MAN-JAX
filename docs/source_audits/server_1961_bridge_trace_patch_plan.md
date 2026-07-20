# Server 1961 Bridge Trace Patch Plan

Scope: one-shot supplemental server trace plan for the paper-case 1961 run. This
is a local planning/schema document only. Do not patch the local Fortran source
truth under `fortran_source/ORCHIDEE`, do not log in to the server for this
task, do not submit jobs, and never write under the server `zzhao` area.

Source truth:

- `fortran_source/ORCHIDEE/src_sechiba/sechiba.f90`
- `fortran_source/ORCHIDEE/src_sechiba/diffuco.f90`
- `fortran_source/ORCHIDEE/src_sechiba/enerbil.f90`
- `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`
- `fortran_source/ORCHIDEE/src_sechiba/slowproc.f90`
- `fortran_source/ORCHIDEE/src_stomate/stomate.f90`

Trace truth already available:

- `outputs/server_1961_trace_full_20260623/MANIFEST.txt`
- `docs/source_audits/server_1961_trace_contract.md`
- `docs/source_audits/sechiba_pft14_bridge_gap_audit.md`

Local machine-readable draft:

- `outputs/server_trace_patch/bridge_trace_patch_plan.py`
- `outputs/server_trace_patch/apply_bridge_trace_patch.py`

`apply_bridge_trace_patch.py` is an inert local draft for a future server fresh
copy. It targets only `modipsl_250919_dev/modeles/ORCHIDEE` under a private
server work root and must not be run against `fortran_source/ORCHIDEE`. The
first draft patches `sechiba.f90` and `slowproc.f90` bridge boundaries only;
optional modelout trace is intentionally left out until NetCDF/history output
inspection proves it is needed.

## Objective

The next server trace should close the SECHIBA bridge gaps that block exact
DIFFUCO, ENERBIL, exported HYDROL, THERMOSOIL-to-STOMATE, and STOMATE daily
boundary parity. The trace must be narrow enough to run once without producing
another multi-GB package, but broad enough that workers do not need a second
server round trip.

Minimum selected slice:

- land point: `ji = 1` unless the run owner confirms a different reference
  land point is the paper-case target.
- PFT: `jv = 14`.
- soil tile: `jst = pref_soil_veg(14)`, expected tile 4 for the current PFT14
  mangrove path; include the emitted `jst` key in every tile record rather than
  assuming it downstream.
- hydrology layers: all `jsl = 1:nslm` for layer fields, because
  `shumdiag`, `soil_mc`, `wat_flux`, and thermosoil moisture cannot be
  validated from only the bottom layer.
- snow layers: all `ksnow = 1:nsnow` only for `snowdz/snowrho/snowtemp` rows.
- timestep: first four SECHIBA calls plus the first daily boundary where
  `do_slow = .TRUE.`. The run owner may extend to the first 48 half-hourly
  steps if the first day is needed for daily accumulator closure.

All records should use the same fixed-format style as
`jax_orchidee.trace.fixed_format`: a textual tag first, followed by numeric or
logical list-directed values, with stable field order and no headers inside the
trace file.

## Output Files And Registry Names

Use new files to avoid changing the semantics of the existing 2026-06-23
package:

| Registry group | File | Tag(s) |
| --- | --- | --- |
| `sechiba_bridge_diffuco` | `orchjax_sechiba_bridge_diffuco_trace.txt` | `after_diffuco_main` |
| `sechiba_bridge_enerbil` | `orchjax_sechiba_bridge_enerbil_trace.txt` | `after_enerbil_main` |
| `sechiba_bridge_hydrol` | `orchjax_sechiba_bridge_hydrol_trace.txt` | `after_hydrol_main_pft`, `after_hydrol_main_layer`, `after_hydrol_main_tile_layer`, `after_hydrol_main_tile` |
| `sechiba_bridge_thermosoil` | `orchjax_sechiba_bridge_thermosoil_trace.txt` | `after_thermosoil_before_slowproc_pft`, `after_thermosoil_before_slowproc_layer` |
| `sechiba_bridge_slowproc` | `orchjax_sechiba_bridge_slowproc_trace.txt` | `before_slowproc_main`, `before_stomate_main` |
| `sechiba_bridge_modelout` | `orchjax_sechiba_bridge_modelout_trace.txt` | `before_modelout`, `after_modelout` only if modelout remains ambiguous after local NetCDF inspection |

The local reader can register each file in `jax_orchidee.trace.server_1961`
without new parser machinery. The field list in
`outputs/server_trace_patch/bridge_trace_patch_plan.py` is intentionally shaped
like a future `FixedTraceSchema` addition.

## Common Record Keys

Every record starts with enough keys for streaming lookup:

- `kjit`: SECHIBA timestep.
- `ji`: local land-point index.
- `jv`: PFT index when the record is PFT-specific; emit `14`.
- `jst`: soil tile index for tile records; emit `pref_soil_veg(14)` in bridge
  records from `sechiba.f90`, and the hydrol tile loop index in
  `hydrol.f90`-internal records if used.
- `jsl`: hydrology layer index for layer records.
- `ksnow`: snow layer index for snow records.
- `do_slow`: logical boundary flag for slowproc/STOMATE records.
- `dt_sechiba`, `dt_stomate`, `dt_days`: include at daily boundaries so
  accumulator workers can validate units without opening unrelated traces.

## Record Groups

### `after_diffuco_main`

Fortran location:

- File: `src_sechiba/sechiba.f90`
- Subroutine: `sechiba_main`
- Insert immediately after the `CALL diffuco_main` ending near line 1005.
- Producer source: `src_sechiba/diffuco.f90::diffuco_main` lines 306-410;
  `diffuco_trans_co2` call lines 665-671; GPP/resistance assignment lines
  2885-2931 and 2948-2969.

Record:

- File: `orchjax_sechiba_bridge_diffuco_trace.txt`
- Tag: `after_diffuco_main`
- Rows: one row for `ji=1, jv=14` for selected `kjit`.

Fields:

`kjit, ji, jv, jst_pref, gpp, gsmean, rveget, rstruct, cimean, vbeta,
vbeta_pft, vbeta1, vbeta2, vbeta3, vbeta3pot, vbeta4, vbeta4_pft, vbeta5,
q_cdrag, q_cdrag_pft, humrel, qsintveg, qsintmax, salinity, tide_height_1,
veget, veget_max, lai, temp_sol, temp_sol_pft, qsurf, evapot, evapot_corr`

Shape/index notes:

- Scalar by point: `vbeta(ji)`, `vbeta1(ji)`, `vbeta4(ji)`, `vbeta5(ji)`,
  `q_cdrag(ji)`, `temp_sol(ji)`, `qsurf(ji)`, `evapot(ji)`,
  `evapot_corr(ji)`.
- PFT fields: `(...)(ji,14)`.
- `jst_pref = pref_soil_veg(14)` lets HYDROL and ENERBIL workers connect PFT14
  to its soil tile.
- `tide_height_1 = tide_height(ji,1)` is enough for the first-step mangrove
  branch unless the server pre-check shows that DIFFUCO indexes another tide
  slot; if so, emit `tide_height(ji,1:MIN(4,SIZE(tide_height,2)))` as separate
  fixed fields.

Consumers:

- DIFFUCO worker: photosynthesis and resistance parity.
- ENERBIL worker: verifies `vbeta*`, `q_cdrag*`, and resistance inputs.
- STOMATE worker: verifies `gpp` entering daily accumulation.

Why necessary:

- Existing `orchjax_stomate_daily_trace.txt` proves `gpp_d` at the STOMATE
  accumulator but not the DIFFUCO process boundary, conductance/resistance
  state, or mangrove salinity/tide controls.

### `after_enerbil_main`

Fortran location:

- File: `src_sechiba/sechiba.f90`
- Subroutine: `sechiba_main`
- Insert immediately after the `CALL enerbil_main` ending near line 1019.
- Producer source: `src_sechiba/enerbil.f90::enerbil_main` lines 390-471;
  surface state calculations lines 507-545; evaporation split in
  `enerbil_evapveg` lines 1692-1832.

Record:

- File: `orchjax_sechiba_bridge_enerbil_trace.txt`
- Tag: `after_enerbil_main`
- Rows: one row for `ji=1, jv=14` for selected `kjit`.

Fields:

`kjit, ji, jv, jst_pref, transpir, transpot, vevapwet, vevapnu, vevapnu_pft,
vevapsno, vevapflo, vevapp, evapot, evapot_corr, temp_sol, temp_sol_pft,
temp_sol_new, temp_sol_new_pft, qsurf, t2mdiag, fluxsens, fluxlat, soilcap,
soilcap_pft, snowdz_1, precip_rain`

Shape/index notes:

- PFT fields: `transpir(ji,14)`, `transpot(ji,14)`, `vevapwet(ji,14)`,
  `vevapnu_pft(ji,14)`, `temp_sol_pft(ji,14)`,
  `temp_sol_new_pft(ji,14)`, `soilcap_pft(ji,14)`.
- Point fields: all other one-dimensional variables.
- `snowdz_1` is a first-layer sentinel to confirm whether explicit snow is
  active in the traced timestep; full snow profile is captured in HYDROL or
  thermosoil groups.

Consumers:

- ENERBIL worker: surface energy and evaporation split parity.
- HYDROL worker: exact water-balance forcing into `hydrol_main`.
- STOMATE/daily worker: `t2mdiag`, `temp_sol`, and `evapot_corr` daily forcing.

Why necessary:

- Existing traces do not expose `transpir`, `transpot`, `vevap*`,
  `temp_sol_new`, `qsurf`, or `evapot_corr` at the process boundary. HYDROL
  and STOMATE cannot be made exact by inspecting history output alone.

### `after_hydrol_main_pft`

Fortran location:

- File: `src_sechiba/sechiba.f90`
- Subroutine: `sechiba_main`
- Insert inside the `hydrol_cwrr` branch immediately after the `CALL hydrol_main`
  ending near line 1072, before `rsol(:) = -un`.
- Producer source: `src_sechiba/hydrol.f90::hydrol_main` lines 938-1090;
  `hydrol_canop` call lines 1232-1235; `hydrol_soil` call lines 1278-1296;
  final exported assignments lines 7360-7363.

Record:

- File: `orchjax_sechiba_bridge_hydrol_trace.txt`
- Tag: `after_hydrol_main_pft`
- Rows: one row for `ji=1, jv=14` for selected `kjit`.

Fields:

`kjit, ji, jv, jst_pref, humrel, vegstress, qsintveg, precip2canopy,
precip2ground, canopy2ground, runoff, drainage, drysoil_frac, evap_bare_lim,
k_litt, litterhumdiag, snow, snow_age, snow_nobio_1, snow_nobio_age_1,
tot_melt, floodout, fwet_out, wtp, fwet_new, mc_peat_above, liqwt_ratio,
mc_man_above`

Shape/index notes:

- PFT fields use `jv=14`.
- `snow_nobio_1` is the first non-biological class only. Add all `nnobio`
  classes only if the next-step surface worker requires them; otherwise this
  bridge focuses on PFT14.
- `runoff` and `drainage` are whole-point exports; tile-specific values are in
  `after_hydrol_main_tile`.

Consumers:

- HYDROL exported-boundary worker.
- THERMOSOIL worker through post-HYDROL moisture state.
- STOMATE daily and MICT-leak soilcarbon workers.
- Next-step DIFFUCO/CONDVEG workers for `qsintveg`, snow, `drysoil_frac`,
  `evap_bare_lim`, and `k_litt`.

Why necessary:

- Existing HYDROL traces cover internal solve/post/update records, not the
  exported `hydrol_main` boundary that `sechiba_main` passes forward.

### `after_hydrol_main_layer`

Fortran location:

- Same as `after_hydrol_main_pft`, immediately after `CALL hydrol_main`.

Record:

- File: `orchjax_sechiba_bridge_hydrol_trace.txt`
- Tag: `after_hydrol_main_layer`
- Rows: `ji=1`, `jv=14`, `jsl=1:nslm` for selected `kjit`.

Fields:

`kjit, ji, jv, jst_pref, jsl, shumdiag, shumdiag_perma, shumdiag_peat,
shumdiag_croppeat, shumdiag_man, mc_layh, mcl_layh, soilmoist, mc_layh_pft,
mcl_layh_pft, soilmoist_pft`

Shape/index notes:

- `mc_layh_pft/mcl_layh_pft/soilmoist_pft` are available only after
  `sechiba.f90` maps `mc_layh_s/mcl_layh_s/soilmoist` to PFT arrays at lines
  1093-1102. Therefore either emit this record twice or place it after the
  mapping block. Recommended: emit layer moisture after the mapping block and
  keep the tag name `after_hydrol_main_layer`.
- `mc_layh_pft(ji,jsl,14) = mc_layh_s(ji,jsl,pref_soil_veg(14))`.

Consumers:

- THERMOSOIL worker.
- STOMATE daily soil humidity worker.
- HYDROL exported-state worker.

Why necessary:

- Soil temperature and STOMATE maintenance respiration depend on the full
  layer profile, not only one HYDROL internal layer sample.

### `after_hydrol_main_tile_layer`

Fortran location:

- Same `sechiba_main` block after `CALL hydrol_main`; emit before or after the
  PFT moisture mapping because these fields are already exported from HYDROL.

Record:

- File: `orchjax_sechiba_bridge_hydrol_trace.txt`
- Tag: `after_hydrol_main_tile_layer`
- Rows: `ji=1`, `jst=pref_soil_veg(14)`, `jsl=1:nslm` for selected `kjit`.

Fields:

`kjit, ji, jv, jst, jsl, soil_mc, wat_flux, mc_layh_s, mcl_layh_s`

Shape/index notes:

- `soil_mc(ji,jsl,jst)`, `wat_flux(ji,jsl,jst)`,
  `mc_layh_s(ji,jsl,jst)`, `mcl_layh_s(ji,jsl,jst)`.

Consumers:

- HYDROL exported-state worker.
- STOMATE MICT-leak/DOC worker.
- THERMOSOIL PFT moisture mapping audit.

Why necessary:

- `soil_mc` and `wat_flux` are direct MICT-leak inputs and are not available in
  existing internal solve traces as exported arrays.

### `after_hydrol_main_tile`

Fortran location:

- Same `sechiba_main` block after `CALL hydrol_main`.

Record:

- File: `orchjax_sechiba_bridge_hydrol_trace.txt`
- Tag: `after_hydrol_main_tile`
- Rows: `ji=1`, `jst=pref_soil_veg(14)` for selected `kjit`.

Fields:

`kjit, ji, jv, jst, runoff_per_soil, runoff2peat, drainage_per_soil, soiltile,
reinf_slope, drunoff_tot`

Shape/index notes:

- Tile fields use `(...)(ji,jst)`.
- Include `soiltile(ji,jst)` and `reinf_slope(ji)` to diagnose masked or
  near-zero tile behavior without tracing all six tiles.

Consumers:

- HYDROL water-budget worker.
- STOMATE MICT-leak runoff/drainage DOC export worker.

Why necessary:

- Whole-point `runoff/drainage` cannot validate tile-specific MICT-leak water
  exports.

### `after_thermosoil_before_slowproc_pft`

Fortran location:

- File: `src_sechiba/sechiba.f90`
- Subroutine: `sechiba_main`
- Insert immediately after the `CALL thermosoil_main` ending near line 1118
  and before the wind-speed loop near lines 1131-1136.
- Producer source: `thermosoil_main` call receives `stempdiag`,
  `temp_sol_new`, `mc_layh`, `mcl_layh`, `soilmoist`, and PFT moisture arrays
  at `sechiba.f90` lines 1109-1118.

Record:

- File: `orchjax_sechiba_bridge_thermosoil_trace.txt`
- Tag: `after_thermosoil_before_slowproc_pft`
- Rows: one row for `ji=1, jv=14` for selected `kjit`.

Fields:

`kjit, ji, jv, jst_pref, temp_sol, temp_sol_new, temp_sol_pft,
temp_sol_new_pft, t2mdiag, qsurf, soilflx, soilflx_pft, soilcap, soilcap_pft,
grndflux, gtemp`

Shape/index notes:

- PFT fields use `jv=14`.
- `temp_sol` is the value later passed to `slowproc_main`; `temp_sol_new` is
  retained for ENERBIL/HYDROL/THERMOSOIL debugging.

Consumers:

- THERMOSOIL worker.
- STOMATE daily temperature and maintenance-respiration workers.

Why necessary:

- Existing traces include STOMATE maintenance samples, but not the SECHIBA soil
  thermal boundary that generated `stempdiag`.

### `after_thermosoil_before_slowproc_layer`

Fortran location:

- Same as `after_thermosoil_before_slowproc_pft`.

Record:

- File: `orchjax_sechiba_bridge_thermosoil_trace.txt`
- Tag: `after_thermosoil_before_slowproc_layer`
- Rows: `ji=1`, `jsl=1:nslm` for selected `kjit`.

Fields:

`kjit, ji, jv, jst_pref, jsl, stempdiag, shumdiag, shumdiag_perma, mc_layh,
mcl_layh, soilmoist`

Shape/index notes:

- `stempdiag(ji,jsl)` is the critical output.
- Include moisture fields again to confirm the exact inputs seen by
  `thermosoil_main` around the update.

Consumers:

- THERMOSOIL worker.
- STOMATE daily `tsoil_daily` and maintenance respiration workers.

Why necessary:

- STOMATE accumulates `stempdiag` at `stomate.f90` lines 3201-3208 and uses
  soil temperature in maintenance respiration; existing bridge traces do not
  expose the profile before slowproc.

### `before_slowproc_main`

Fortran location:

- File: `src_sechiba/sechiba.f90`
- Subroutine: `sechiba_main`
- Insert immediately before `CALL slowproc_main` near lines 1184-1216.

Record:

- File: `orchjax_sechiba_bridge_slowproc_trace.txt`
- Tag: `before_slowproc_main`
- Rows: one row for `ji=1, jv=14`, plus layer companion rows only if the
  `after_thermosoil` records are omitted. Recommended: keep this row scalar/PFT
  only and rely on `after_thermosoil_before_slowproc_layer` for layers.

Fields:

`kjit, ji, jv, jst_pref, t2mdiag, temp_sol, gpp, humrel, vegstress,
litterhumdiag, precip_rain, precip_snow, swdown, evapot_corr, snow,
snowdz_1, snowrho_1, tot_bare_soil, veget, veget_max, lai, frac_age_1,
height, qsintmax, wspeed, wtp, fwet_new, fpeat, mc_peat_above, liqwt_ratio,
mc_man_above, flood_frac_stream`

Shape/index notes:

- `flood_frac_stream = flood_frac(ji) + streamfl_frac(ji)` to match the actual
  slowproc argument at lines 1211-1213.
- `frac_age_1 = frac_age(ji,14,1)` is only a sentinel. If age-class workers
  need full `nagec`, add separate `before_slowproc_age` rows.

Consumers:

- Bridge adapter that connects SECHIBA to slowproc.
- STOMATE worker if `before_stomate_main` is absent.

Why necessary:

- It proves the exact SECHIBA-side state after DIFFUCO, ENERBIL, HYDROL,
  CONDVEG, and THERMOSOIL have all run.

### `before_stomate_main`

Fortran location:

- File: `src_sechiba/slowproc.f90`
- Subroutine: `slowproc_main`
- Insert inside `IF (ok_stomate)` immediately before `CALL stomate_main` near
  line 973.

Record:

- File: `orchjax_sechiba_bridge_slowproc_trace.txt`
- Tag: `before_stomate_main`
- Rows: one row for `ji=1, jv=14` for selected `kjit`; emit only when
  `ok_stomate` is true.

Fields:

`kjit, ji, jv, do_slow, end_of_year, dt_sechiba, dt_stomate, dt_days, t2m,
t2m_min, temp_sol, humrel, litterhumdiag, precip_rain, precip_snow, wspeed,
lightn, popd, gpp, lai, veget, veget_max, veget_max_new, t2mdiag,
evapot_corr, tdeep, hsdeep_long, snow, snowdz_1, snowrho_1, wtp, fwet_new,
fpeat, shumdiag_peat_1, mc_peat_above, liqwt_ratio, shumdiag_croppeat_1,
mc_croppeat_above, shumdiag_man_1, mc_man_above, soil_mc_top_tile,
wat_flux_top_tile, drainage_per_soil_tile, runoff_per_soil_tile,
runoff2peat_tile, flood_frac, precip2canopy, precip2ground, canopy2ground`

Shape/index notes:

- `soil_mc_top_tile = soil_mc(ji,1,pref_soil_veg(14))` and equivalent tile
  fields are sentinels here. Full layer/tile arrays are already in
  `after_hydrol_main_tile_layer`.
- Use `dt_*` names already understood by current STOMATE trace readers.

Consumers:

- STOMATE daily carbon, daily water, maintenance, NPP/allocation, and MICT-leak
  workers.

Why necessary:

- It verifies that `slowproc_main` did not alter the SECHIBA bridge fields and
  captures the monthly/static fire and population values that are added between
  `before_slowproc_main` and `stomate_main`.

### Optional `before_modelout` / `after_modelout`

Fortran location:

- Prefer not to add this unless local NetCDF/modelout comparison remains
  ambiguous.
- Candidate file: `src_sechiba/sechiba.f90`, around the history writes starting
  near line 1529, or the project-specific modelout call site if audited before
  the server run.

Record:

- File: `orchjax_sechiba_bridge_modelout_trace.txt`
- Tags: `before_modelout`, `after_modelout`.

Fields:

`kjit, ji, jv, gpp, transpir, vevapnu, vevapwet, vevapsno, vevapflo, humrel,
vegstress, temp_sol, t2mdiag, runoff, drainage, npp, co2_flux`

Why optional:

- History/modelout values are consumers, not process boundaries. Add this only
  if the next worker must distinguish a process mismatch from a write/units
  mismatch.

## Output Limiting Rules

The patch should define constants in `sechiba.f90`/`slowproc.f90` or a small
included helper block:

- `orchjax_trace_point = 1`
- `orchjax_trace_pft = 14`
- `orchjax_trace_steps = 4`
- `orchjax_trace_daily = .TRUE.`

Recommended write guards:

```fortran
IF (ji == orchjax_trace_point .AND. kjit <= orchjax_trace_steps) THEN
   ...
ENDIF
IF (ji == orchjax_trace_point .AND. jv == orchjax_trace_pft .AND. &
    (kjit <= orchjax_trace_steps .OR. do_slow)) THEN
   ...
ENDIF
```

Layer rows are acceptable because the selected slice is small:

- `after_hydrol_main_layer`: at most `nslm` rows per selected timestep.
- `after_hydrol_main_tile_layer`: at most `nslm` rows per selected timestep.
- `after_thermosoil_before_slowproc_layer`: at most `nslm` rows per selected
  timestep.

Avoid:

- loops over all land points;
- loops over all PFTs except where a worker explicitly needs an all-PFT sum;
- writing full `nstm` or `nsnow` cubes unless the record is tile/layer limited;
- tracing every timestep for the full year.

The expected bridge trace package should be kilobytes to a few megabytes, not
hundreds of megabytes.

## Server-Run Preconditions

Before anyone goes to the server, confirm:

1. The selected land-point index is still `ji=1` for the paper-case local slice.
2. `pref_soil_veg(14)` is tile 4 in the exact server build/run configuration.
3. The run should cover either first four SECHIBA timesteps only or first day
   plus the first `do_slow=.TRUE.` daily boundary.
4. Whether DIFFUCO mangrove control needs more than `tide_height(ji,1)`.
5. Whether modelout before/after records are necessary, or NetCDF output plus
   process boundaries are sufficient.
6. The server patch will be applied only to a private copy under
   `/public/share/qhcess/qhcess2/User/jcwu/...`, never to `zzhao`.

## Expected Run Outputs

The next run should return:

- `traces/orchjax_sechiba_bridge_diffuco_trace.txt`
- `traces/orchjax_sechiba_bridge_enerbil_trace.txt`
- `traces/orchjax_sechiba_bridge_hydrol_trace.txt`
- `traces/orchjax_sechiba_bridge_thermosoil_trace.txt`
- `traces/orchjax_sechiba_bridge_slowproc_trace.txt`
- optional `traces/orchjax_sechiba_bridge_modelout_trace.txt`
- run log and manifest with source root, run script, patched build tree, created
  UTC timestamp, and line counts.

## Success Validation

Local validation after the trace package is copied back:

1. Register the new fixed-format schemas in `jax_orchidee.trace.server_1961`
   using the field order from `outputs/server_trace_patch/bridge_trace_patch_plan.py`.
2. Add streaming parser tests that read one synthetic record per tag and reject
   arity drift.
3. Assert that `find_server_record(..., criteria={"itime": first_kjit,
   "ji": 1, "jv": 14})` works for PFT records without materializing the files.
4. Check DIFFUCO:
   `after_diffuco_main.gpp == before_stomate_main.gpp` for matching `kjit`,
   allowing only expected unit/name differences already documented in STOMATE.
5. Check ENERBIL to HYDROL:
   `after_enerbil_main.transpir/transpot/vevap*` match the fields entering the
   HYDROL boundary for the same `kjit`.
6. Check HYDROL export:
   `after_hydrol_main_tile_layer.soil_mc/wat_flux` and
   `after_hydrol_main_tile.runoff_per_soil/drainage_per_soil` match the
   sentinels in `before_stomate_main`.
7. Check THERMOSOIL to STOMATE:
   `after_thermosoil_before_slowproc_layer.stempdiag` is the profile passed to
   STOMATE and accumulated at `stomate.f90` lines 3201-3208.
8. Check daily boundary:
   the first `before_stomate_main` record with `do_slow=.TRUE.` brackets the
   existing `orchjax_stomate_daily_trace.txt:gpp_before_accu/gpp_after_accu`
   daily accumulator records.

If any of these fail, prefer local schema/field-order fixes first. A new server
run should be needed only if a required variable was not emitted or the selected
slice missed the first daily boundary.
