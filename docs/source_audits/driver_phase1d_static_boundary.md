# Driver Phase 1D/1E Static Boundary Audit

This note is for the Driver/Grid implementation boundary only. It records why
Phase 1D implements restart validation reads but does not yet implement
soilclass, salinity, or tide static readers. Phase 1E adds explicit-geometry
generic helpers but still does not claim paper-case static parity.

## Restart Truth Inventory

Local reference directory:

`reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658`

`driver_start.nc` and `driver_restart.nc` contain driver state fields only:
`fluxsens`, `vevapp`, `zlev_old`, `qair_old`, `eair_old`, `rau_old`, PET/PEQ
coefficients, albedo, and `z0`. They do not contain static SECHIBA fields.

`sechiba_start.nc` and `sechiba_restart.nc` contain these Phase 1D validation
fields:

- `veget`, `veget_max`, `frac_nobio`
- `njsc`, `clay_frac`, `sand_frac`, `bulk_dens`, `soil_ph`, `poor_soils`
- `wtp`, `wt_ab_tide`

They do not contain `soiltile`, `soilclass`, `salinity`, or `tide_height`.
Therefore these absent fields cannot be used as restart validation truth.

## Fortran Provenance

Restart reads in `fortran_source/ORCHIDEE/src_sechiba/slowproc.f90`:

- `slowproc_init`, lines 1581-1601: `veget`, `veget_max`, `frac_nobio`
- `slowproc_init`, lines 1655-1674: `njsc`, `clay_frac`, `sand_frac`
- `slowproc_init`, lines 1791-1804: `bulk_dens`, `soil_ph`, `poor_soils`

Restart writes in `slowproc.f90`:

- lines 1222-1239: `veget`, `veget_max`, `frac_nobio`, `njsc`,
  `clay_frac`, `sand_frac`
- lines 1269-1271: `soil_ph`, `poor_soils`, `bulk_dens`

Soilclass/static soil source path:

- `slowproc_init`, lines 1997-2010 and 2174-2192: call
  `slowproc_soilt` when restart soil fields are absent, then set
  `njsc = MAXLOC(soilclass)`
- `slowproc_soilt`, lines 4284-4925: reads `SOILCLASS_FILE`, `soiltext`,
  `soil_ph`, `poor_soils`, builds `soilclass`, `clayfraction`,
  `sandfraction`, `bulk_density`, `soil_ph`, and `poor_soils`
- `slowproc_soilt`, lines 4544-4545: calls `aggregate_p`
- `src_global/interpol_help.f90`, `aggregate_2d`, lines 46-466: exact
  area-overlap calculation using `lalo`, `neighbours`, `resolution`, and
  `contfrac`

Salinity/tide source path:

- `slowproc_init`, lines 2494-2534: reads `READ_SALINITY`, `READ_TIDE`, then
  calls `slowproc_read_annual(..., 'salinity')` and
  `slowproc_read_data(..., 'tide')`
- `slowproc_read_data`, lines 6029-6297: computes model-cell bbox from
  `lalo` and `resolution`, averages source points inside the bbox, and only
  falls back to nearest source point when no source point is found
- `slowproc_read_annual`, lines 6300-6570: same bbox/average/fallback pattern
  for annual scalar fields

Grid geometry source path:

- `src_global/grid.f90`, `grid_stuff`, lines 416-470: calls
  `grid_topolylist`, stores `contfrac_g`, scatters `neighbours_g`,
  `resolution_g`, and other geometry fields
- `grid_topolylist`, lines 490-631: detects `RegLonLat`, calls
  `haversine_singlepointploy` when `nland == 1`, stores polygon corners,
  computes headings, segment lengths, and area
- `src_global/haversine.f90`, `haversine_singlepointploy`, lines 394-488:
  for an `iim=jjm=1` domain, constructs a synthetic one-degree-like polygon
  from `area = 111111.0 * 111111.0` and great-circle radial offsets
- `src_global/haversine.f90`, `haversine_laloseglen`, lines 667-736:
  computes segment lengths in meters along lat/lon-aligned polygon sides
- `src_global/haversine.f90`, `haversine_laloarea`, lines 863-888:
  computes polygon area from averaged opposite segment lengths
- `grid.f90`, `grid_scatter`, lines 688-707: scatters `neighbours`,
  `contfrac`, `area`, and computes `resolution_g(:,1)` from segments 1/3 and
  `resolution_g(:,2)` from segments 2/4 before scattering `resolution`

## Implementation Boundary

`DomainGrid.resolution` and `DomainGrid.neighbours` are still intentionally
unset in the Python driver. Local `stomate_history_1961.nc` exposes diagnostic
`RESOLUTION_X=111106.453125` and `RESOLUTION_Y=111111.0`, but it does not
expose `neighbours`, `corners`, `seglength`, or the `slowproc_init` bbox and
overlap intermediates. It also reports `CONTFRAC=0.0625` and
`Areas=12345149440.0`, while the forcing-domain reader sees
`contfrac=0.5625` and `Areas=46266785792.0`, so this history file cannot be
mixed into driver parity as complete geometry truth.

Phase 1E therefore implements only helpers that require explicit geometry:

- `model_bbox_from_lalo_resolution(lalo, resolution_m)` reproduces the
  bbox formulas from `slowproc_read_data`/`slowproc_read_annual` when the
  caller supplies exact model `lalo` and `resolution`.
- `bbox_center_mean(...)` reproduces the source center-point mean over the
  explicit bbox. It raises by default if no source point is found, and only
  follows the Fortran `slowproc_nearest` fallback when the caller explicitly
  sets `allow_nearest_fallback=True`; salinity/tide readers use that source
  branch and preserve `counts == 0` as the diagnostic marker.
- `weighted_mean_from_explicit_overlap(...)` and
  `zobler_soilclass_from_explicit_overlap(...)` consume already-known
  `sub_index/sub_area` from `aggregate_p`; they do not reconstruct overlap
  geometry.

Next exact-truth requirement:

- Trace or restart-dump `lalo`, `resolution`, `neighbours`, and `contfrac` at
  the `slowproc_init` call site for the 1961 paper case.
- For soilclass parity, also trace `sub_index`, `sub_area`, `soilclass`,
  `clayfraction`, `sandfraction`, `bulk_density`, `soil_ph`, `poor_soils`,
  and final `njsc` after `slowproc_soilt`.
- For salinity/tide parity, trace `lon_low/lon_up/lat_low/lat_up`,
  `n_origpopden` or `n_origlightn`, averaged `salinity`, and `tide_height`.
