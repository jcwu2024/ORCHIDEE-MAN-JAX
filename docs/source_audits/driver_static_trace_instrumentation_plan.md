# Driver Static Trace Instrumentation Plan

Phase 1G goal: unlock exact paper-case parity for driver geometry and static
inputs without guessing `resolution`, `neighbours`, bboxes, or area overlaps.
This is a practical trace patch plan only; no Fortran source is modified here.

## 1. Domain And First Forcing Point

Insertion points:

- `src_driver/dim2_driver.f90`, lines 415-435: immediately after the first
  `forcing_READ`, before and after `ilandindex/jlandindex` are derived.
- `src_driver/readdim2.f90`, `forcing_info`, lines 49-495: forcing dimensions,
  nav lon/lat, `Areas`, `contfrac`.
- `readdim2.f90`, `forcing_read_interpol`, lines 660-1687, and
  `forcing_just_read`, lines 1691-1869: first timestep fields.
- `readdim2.f90`, `forcing_landind`, lines 1899-1968: final `nbindex` and
  `kindex`.
- `readdim2.f90`, `forcing_grid`/`forcing_zoom`, lines 1972-2051, and
  `domain_size`, lines 2263-2341: zoomed domain and selected forcing cells.

Trace group: `domain_forcing_selected_point`

Required columns:

`year,tstep,ik,i_fortran,j_fortran,kindex,nav_lon,nav_lat,lalo_lat,lalo_lon,contfrac,area,Tair,PSurf,Qair,Wind_E,Wind_N,Rainf,Snowf,SWdown,LWdown,Height_Lev1,Height_Levuv`

JAX consumer:

- `jax_orchidee.driver.domain.read_domain_grid`
- `jax_orchidee.driver.domain.read_forcing_first_step`

Post-trace tests:

- Assert zero-based Python forcing indices map to Fortran `i_fortran/j_fortran`.
- Assert `kindex`, `lalo`, `contfrac`, `area`, and first timestep forcing
  match trace exactly.

## 2. Grid Geometry After `grid_stuff`

Insertion points:

- `src_global/grid.f90`, `grid_stuff`, lines 416-470: after
  `grid_topolylist` and after `grid_scatter`.
- `grid.f90`, `grid_topolylist`, lines 490-631: after polygon construction,
  sorting, segment length, and area computation.
- `src_global/haversine.f90`, `haversine_singlepointploy`, lines 394-488:
  single-point synthetic polygon used by this paper case.
- `haversine.f90`, `haversine_laloseglen`, lines 667-736, and
  `haversine_laloarea`, lines 863-888: segment lengths and area.
- `grid.f90`, `grid_scatter`, lines 663-718: after `resolution_g` is computed
  from `seglength_g` and scattered.

Trace group: `grid_geometry_after_grid_stuff`

Required columns:

`ik,kindex,lalo_lat,lalo_lon,contfrac,area,resolution_x,resolution_y,neighbour_1,neighbour_2,neighbour_3,neighbour_4,neighbour_5,neighbour_6,neighbour_7,neighbour_8,corner_1_lon,corner_1_lat,corner_2_lon,corner_2_lat,corner_3_lon,corner_3_lat,corner_4_lon,corner_4_lat,seglength_1,seglength_2,seglength_3,seglength_4`

JAX consumer:

- `jax_orchidee.driver.domain.DomainGrid` can populate `resolution` and
  `neighbours` only after this trace exists.
- `jax_orchidee.driver.static.model_bbox_from_lalo_resolution` can consume
  traced `lalo/resolution`.

Post-trace tests:

- Assert `DomainGrid.resolution` and `DomainGrid.neighbours` match trace.
- Assert traced `area` resolves the current mismatch between forcing `Areas`
  and history `Areas`.
- Use traced `resolution` to build salinity/tide bbox tests.

## 3. Soil Aggregation

Insertion points:

- `src_sechiba/slowproc.f90`, `slowproc_init`, lines 1997-2010 and
  2174-2192: before/after calls to `slowproc_soilt`, and after
  `njsc = MAXLOC(soilclass)`.
- `slowproc.f90`, `slowproc_soilt`, lines 4284-4925: before file close,
  after `aggregate_p`, and after each final soil field is normalized.
- `slowproc.f90`, lines 4536-4546: immediately after `sub_index/sub_area`
  allocation and `aggregate_p` call.
- `slowproc.f90`, lines 4612-4664 and 4686-4692: Zobler branch source values,
  accumulated `soilclass`, `clayfraction`, `sandfraction`, `bulk_density`,
  `soil_ph`, `poor_soils`, and normalization.
- `src_global/interpol_help.f90`, `aggregate_2d`, lines 46-466, and
  `aggregate_2d_p`, lines 830-889: exact `sub_index/sub_area` generation.

Trace group: `soil_aggregation`

Required columns:

`ik,fopt,source_i,source_j,sub_area,soiltext,soil_ph_source,poor_soils_source,bulk_dens_source,soilclass_1,soilclass_2,soilclass_3,njsc,clay_frac,sand_frac,bulk_dens,soil_ph,poor_soils`

JAX consumer:

- `jax_orchidee.driver.static.zobler_soilclass_from_explicit_overlap`
- `jax_orchidee.driver.static.weighted_mean_from_explicit_overlap`
- `jax_orchidee.driver.static.njsc_from_soilclass`

Post-trace tests:

- Feed traced `sub_index/sub_area/soiltext` to JAX and compare
  `soilclass/njsc`.
- Feed traced source scalar values to weighted overlap helper and compare
  clay, sand, bulk density, pH, and poor soils.
- Only after this passes should `soilclass/njsc` become generated bundle
  fields instead of restart anchors.

## 4. Salinity And Tide BBox Means

Insertion points:

- `slowproc.f90`, lines 2494-2534: immediately before and after
  `slowproc_read_annual(...,'salinity')` and
  `slowproc_read_data(...,'tide')`.
- `slowproc_read_annual`, lines 6300-6570:
  - after `lon_up/lon_low/lat_up/lat_low` are computed at lines 6399-6418;
  - inside count/accumulation branches at lines 6461-6512;
  - after mean or fallback assignment at lines 6533-6549.
- `slowproc_read_data`, lines 6029-6297:
  - after bbox arrays are computed at lines 6121-6140;
  - inside count/accumulation branches at lines 6182-6239;
  - after mean or fallback assignment at lines 6261-6275.
- `slowproc_nearest`, lines 4209-4263: nearest index when fallback branch is
  taken.

Trace groups: `salinity_bbox`, `tide_bbox`

Required columns:

`field_name,ik,time_index,lon_low,lon_up,lat_low,lat_up,source_count,fallback_nearest,nearest_source_i,nearest_source_j,mean_value,final_value`

JAX consumer:

- `jax_orchidee.driver.static.bbox_center_mean`

Post-trace tests:

- Use traced `resolution/lalo` from grid geometry to reproduce bbox limits.
- Assert source-point counts and means match Fortran for salinity.
- Assert tide time-axis means match Fortran for selected tide timesteps.
- If fallback occurs, add a separate exact nearest-neighbor helper/test using
  traced fallback rows; do not use fallback before it is observed.

## 5. First-Step `intersurf` Boundary

Insertion points:

- `dim2_driver.f90`, lines 1108-1125: just before
  `CALL intersurf_initialize_2d`.
- `dim2_driver.f90`, lines 1293-1305: just before first
  `CALL intersurf_main_2d`.
- `src_sechiba/intersurf.f90`, `intersurf_initialize_2d`, lines 113-152 and
  377-387: after gathering land-point inputs, before `sechiba_initialize`.
- `intersurf.f90`, `intersurf_main_2d`, lines 458-500 and 640-648: after
  gathering land-point inputs, before `sechiba_main`.

Trace group: `intersurf_first_step_boundary`

Required columns:

`call_name,year,tstep,ik,kindex,lalo_lat,lalo_lon,contfrac,resolution_x,resolution_y,temp_air,pb,qair,u,v,precip_rain,precip_snow,swdown,lwdown,zlev,ccanopy`

JAX consumer:

- `jax_orchidee.driver.bundle.DriverFirstStepBundle`

Post-trace tests:

- Compare bundle `domain`, `forcing`, and `ccanopy` to traced boundary rows.
- After geometry trace is available, require `resolution_x/y` and
  `neighbours` instead of `missing_geometry_fields`.
- Confirm `intersurf_initialize_2d` and first `intersurf_main_2d` see the same
  expected first-step driver state unless Fortran modifies a field between the
  two calls.

## Schema Helper

`jax_orchidee.driver.trace_schema` contains the trace group names and required
columns above. It is metadata only: it does not read traces, compute geometry,
or infer static fields.
