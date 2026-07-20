# Local Data Assets Audit

Source configuration:

- `configs/orchidee_man_250919.yaml`
- `PROJECT_MANIFEST.md`

This audit confirms the local files needed for the 1961 minimal single case.

## Required Files

All `minimal_single_case.required_files` entries exist locally.

Core files:

- `data/forcing/cruncep_twodeg_1961.nc`
- `data/INPUTDIR_ZZ/global_co2_ann_1700_2017.txt`
- `data/forcing_VN/WTM_forcing_pos.txt`
- `data/forcing_VN/WTM_forcing_diff.txt`
- `data/INPUTDIR_ZZ/salinity_05.nc`
- `data/INPUTDIR_ZZ/tide_05_584.nc`
- `data/INPUTDIR_ZZ/PFT1860_mangr_025deg.nc`
- `data/INPUTDIR_ZZ/catch2_wtd_vkq_socoff_10.nc`
- `data/INPUTDIR_ZZ/test.nc`
- `data/MICT_BIOE/Input/Population/popd_1901.nc`
- `data/MICT_BIOE/Input/ratio_GFED4s_MICTRef_Opt6.nc`
- `data/MICT_BIOE/Input/soils_param.nc`
- `data/MICT_BIOE/Input/alb_bg_modisopt_2D.nc`
- `data/MICT_BIOE/Input/cartepente2d_15min.nc`
- `data/MICT_BIOE/Input/routing.nc`
- `data/MICT_BIOE/Input/floodplains.nc`
- `data/MICT_BIOE/Input/reftemp.nc`
- `data/MICT_BIOE/Input/refSOC_NCSCD_linear_05deg_v5.nc`
- `data/MICT_BIOE/Input/refSOC_NCSCD_05deg_v3_0-0.3m.nc`

Runtime XML files:

- `fortran_run_scripts/paper_250919/peat-leak_xml_zz/iodef.xml`
- `fortran_run_scripts/paper_250919/peat-leak_xml_zz/context_orchidee.xml`
- `fortran_run_scripts/paper_250919/peat-leak_xml_zz/field_def_orchidee.xml`
- `fortran_run_scripts/paper_250919/peat-leak_xml_zz/file_def_orchidee_pods_year_240405.xml`

## Forcing 1961

File:

`data/forcing/cruncep_twodeg_1961.nc`

Dimensions:

- `latitude = 90`
- `longitude = 180`
- `tstep = 1460`

Variables:

- `nav_lon(latitude, longitude)`: degrees east
- `nav_lat(latitude, longitude)`: degrees north
- `Tair(tstep, latitude, longitude)`: K
- `PSurf(tstep, latitude, longitude)`: Pa
- `Qair(tstep, latitude, longitude)`: kg/kg
- `Wind_E(tstep, latitude, longitude)`: m/s
- `Wind_N(tstep, latitude, longitude)`: m/s
- `Rainf(tstep, latitude, longitude)`: kg/m2/s
- `Snowf(tstep, latitude, longitude)`: kg/m2/s
- `SWdown(tstep, latitude, longitude)`: W/m2
- `LWdown(tstep, latitude, longitude)`: W/m2
- `Areas(latitude, longitude)`: m2
- `contfrac(latitude, longitude)`: unitless
- `time(tstep)`
- `timeplussix(tstep)`

The target forcing center from the config is `(lon=109.0, lat=21.0)`.
The file contains this center directly.

Index convention from the audit:

- zero-based: `longitude=109`, `latitude=21`
- one-based: `longitude=110`, `latitude=22`

First target-point values at timestep 0:

- `Tair = 274.36215 K`
- `PSurf = 99929.5 Pa`
- `Qair = 0.0037663`
- `Rainf = 2.27005e-05`
- `Snowf = 2.92246e-06`
- `SWdown = 2.84980`
- `LWdown = 274.05627`
- `Areas = 33798772736.0`
- `contfrac = 1.0`

## Text Drivers

- `WTM_forcing_pos.txt`: 17520 rows
  - first values: `308.33, 329.00, 349.67, 347.92, 346.17`
- `WTM_forcing_diff.txt`: 17520 rows
  - first values: `44.83, 20.67, 20.67, -1.75, -1.75`
- `global_co2_ann_1700_2017.txt`: 318 rows
  - range: 1700-2017
  - 1961 value: `317.27 ppm`

## Static Inputs

- `PFT1860_mangr_025deg.nc`
  - dims: `PFT=14`, `lat=720`, `lon=1440`
  - variable: `maxvegetfrac`
- `catch2_wtd_vkq_socoff_10.nc`
  - dims: `latitude=60`, `longitude=360`, `time_counter=1`
  - variables: `vp`, `kp`, `qp`, `fmax`, `mask`
- `test.nc`
  - dims: `y=180`, `x=360`
  - variables include `soiltext`, `soilcolor`, `soilbd`, `soil_ph`, `poor_soils`
- `soils_param.nc`
  - dims: `y=180`, `x=360`
  - variables include `soilcolor`, `soiltext`
- `salinity_05.nc`
  - dims: `time_counter=1`, `lat=360`, `lon=720`
  - variable: `salinity`
- `tide_05_584.nc`
  - dims: `time_counter=584`, `lat=360`, `lon=720`
  - variable: `tide`
- `routing.nc`
  - dims: `y=360`, `x=720`
  - variables present: `nav_lon`, `nav_lat`, `trip`, `basins`, `topoind`
  - variables required by this Fortran source but absent locally:
    `gravel`, `drainage_area`
  - no other local NetCDF routing-map candidate was found under `data/`

Server read-only check on 2026-07-08 for inspected reference/test outputs:

- The paper-scale product is represented as a mosaic of independent
  single-landpoint cases (for example
  `orc_calibrate_250919_sen/arg2_1.0/001.0-071.0`,
  `001.0-073.0`, and the rest of the 669 landpoint cases), not as one
  multi-landpoint ORCHIDEE run. Each inspected case therefore remains subject
  to the `nbp_glo=1` no-routing guard even when `RIVER_ROUTING = TRUE`.
- The actual `zzhao/OUT/test_240430_2/out1961/run.def.1961` points
  `ROUTING_FILE` to `/public/home/qhcess2/User/wli/MICT_BIOE/Input/routing.nc`.
- That server file exists, but a local header copy under
  `outputs/routing_server_check/routing_wli.nc` has the same five variables as
  the local file and is missing `gravel` and `drainage_area`.
- The same `used_run.def` has `RIVER_ROUTING = TRUE`, but the inspected
  `zzhao/OUT/test_240430_2/out1961/sechiba_restart.nc` has only `x=1`,
  `y=1` and no routing/reservoir route variables. This is consistent with the
  Fortran guard in `sechiba.f90` that skips `routing_initialize` and
  `routing_main` unless `river_routing .AND. nbp_glo .GT. 1`.
- The inspected `zzhao/OUT/orc_calibrate_250919_sen/.../stomate_history_1961.nc`
  path matching the local `reference/case_001_071` copy is also a one-cell
  modelout file (`lat=1`, `lon=1`).

Other required inputs are present and should be audited in detail only when
their process branch is implemented.

## Reference Outputs

Reference directory:

`reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658`

Present:

- `stomate_history_1961.nc` through `stomate_history_2010.nc`
- `driver_start.nc`
- `sechiba_start.nc`
- `stomate_start.nc`
- `driver_restart.nc`
- `sechiba_restart.nc`
- `stomate_restart.nc`

`stomate_history_1961.nc`:

- dims: `time_counter=1`, `lat=1`, `lon=1`, `veget=14`
- contains yearly STOMATE fields for PFT14 modelout validation

## Risk Notes

- Atmospheric forcing is on a 1-degree grid.
- Static inputs include 0.5-degree and 0.25-degree grids.
- The driver/static reader must follow the Fortran interpolation and subset
  rules rather than inventing a simplified mapping.
- Some static files have sparse metadata; process-specific audits should cite
  Fortran read/interpolation code before implementing readers.
- The local `routing.nc` is not a complete input for a true multi-landpoint
  `routing.f90::routing_basins` in this source tree because lines 7179 and
  7181 read `gravel` and `drainage_area`. JAX routing topology code must keep
  failing explicitly on this file until a complete source-equivalent routing
  map is supplied or the Fortran paper target expands from the 669 independent
  single-landpoint mosaic to an actual `nbp_glo>1` routing run.
