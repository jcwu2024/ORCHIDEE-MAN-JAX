# Driver First-Step Boundary Contract

This contract describes the Phase 1F bundle assembled by
`jax_orchidee.driver.bundle.load_paper_1961_first_step_bundle`. It is a
driver/SECHIBA boundary input bundle, not a SECHIBA, HYDROL, or STOMATE process
implementation.

Update 2026-06-23: the fixed-format server trace package under
`outputs/server_1961_trace_full_20260623/traces` is now consumed for explicit
grid geometry and slowproc static truth when `fixed_format_trace_dir` is
supplied.

## Fortran Call Order

1. `dim2_driver.f90`, lines 147-175

   The driver opens the two water-table text files, reads `FORCING_FILE`, and
   calls `forcing_info`.

2. `dim2_driver.f90`, lines 415-435

   The first real forcing read calls `forcing_READ`, fills `kindex`,
   `nbindex`, lon/lat, meteorological fields, `for_contfrac`,
   `for_resolution`, and `for_neighbours`, then derives `ilandindex` and
   `jlandindex`.

3. `dim2_driver.f90`, lines 708-719

   `ATM_CO2` is read and assigned to `for_ccanopy(:,:)`.

4. `dim2_driver.f90`, lines 1108-1125

   `intersurf_initialize_2d` is called with `iim`, `jjm`, `nbindex`,
   `kindex`, lon/lat, `for_contfrac`, `for_resolution`, first-level forcing
   fields, `for_ccanopy`, precipitation, radiation, pressure, and output
   arrays.

5. `intersurf.f90`, lines 113-152 and 377-387

   `intersurf_initialize_2d` gathers land-point fields and calls
   `sechiba_initialize` with `lalo`, `contfrac`, `neighbours`, `resolution`,
   forcing fields, and restart/history identifiers.

6. `sechiba.f90`, lines 602-605

   `sechiba_initialize` calls `sechiba_init`, then `slowproc_initialize`.

7. `slowproc.f90`, lines 1581-1601, 1655-1674, 1791-1804, 1881-1926, and
   2820-2925

   Restart fields are read, imposed vegetation can set `veget_max`, and
   `slowproc_veget` derives `veget` and `soiltile` from `veget_max`,
   `pref_soil_veg`, LAI, and `EXT_COEFF_VEGETFRAC`.

## Bundle Fields Ready

The Phase 1F bundle contains these verified fields:

- `domain.iim`, `domain.jjm`, `domain.nbindex`, `domain.kindex`
- `domain.lon`, `domain.lat`, `domain.lalo`, `domain.contfrac_land`
- first forcing fields: `Tair`, `PSurf`, `Qair`, `Wind_E`, `Wind_N`,
  `Rainf`, `Snowf`, `SWdown`, `LWdown`, `Height_Lev1`, `Height_Levuv`
- Fortran-facing forcing aliases: `temp_air`, `pb`, `qair`, `u`, `v`,
  `precip_rain`, `precip_snow`, `swdown`, `lwdown`, `zlev`, `zlevuv`
- `co2_ppm=317.27` and land-point `ccanopy[npts]`
- water-table positive and differential text sequences
- `RunScalars`: `NVM=14`, `NSTM=6`, `PREF_SOIL_VEG__00014=4`,
  `SECHIBA_VEGMAX__00014=1`, materialized `EXT_COEFF_VEGETFRAC`, and
  `SLOWPROC_HEIGHT`; MTC-backed `z0_over_height` and materialized
  `RATIO_Z0M_Z0H` for CONDVEG static roughness
- `ImposedVegetationState`: `veget_max[npts,nvm]`, `frac_nobio`,
  `totfrac_nobio`, `veget[npts,nvm]`, `soiltile[npts,nstm]`
- restart validation anchors from `sechiba_start.nc`: `njsc`, `clay_frac`,
  `sand_frac`, `bulk_dens`, `soil_ph`, `poor_soils`, `wtp`, `wt_ab_tide`
- optional trace-backed `domain.resolution`, `domain.neighbours`, and final
  grid `area` from `grid_scatter`
- optional trace-backed `StaticTraceFields.soilclass`, `njsc`,
  clay/sand/silt fractions, bulk density, soil pH, poor-soils fraction,
  salinity, and tide height from `slowproc_soilt`, `slowproc_read_annual`,
  and `slowproc_read_data`

Local validation anchors for the paper case:

- `nav_lon=109`, `nav_lat=21`, `kindex=[1]`, `nbindex=1`
- `contfrac_land=0.5625`, forcing `Areas=46266785792.0`
- first forcing step: `Tair=291.3238830566406`,
  `PSurf=98422.8671875`, `Qair=0.008104387670755386`,
  `Wind_E=-3.8484580516815186`, `Wind_N=-0.8622167110443115`,
  `SWdown=362.3620910644531`, `LWdown=338.0463562011719`
- `veget_max[0,13]=1`, cold-start `veget[0,13]=1`, `soiltile[0,3]=1`
- restart anchors: `njsc=2`, `clay_frac=0.2`, `sand_frac=0.4`,
  `bulk_dens=1650`, `soil_ph=7`, `poor_soils=0`, `wtp=0`,
  `wt_ab_tide=0`

## Blocked Fields

Without the 2026-06-23 fixed-format trace directory, the bundle intentionally
does not contain guessed values for:

- exact `resolution`, `neighbours`, `corners`, and `seglength` from
  `grid_stuff`
- `soilclass`, `soilclass_sub_index`, and `soilclass_sub_area`
- salinity bbox and averaged `salinity`
- tide bbox and averaged `tide_height`
The local cold-start imposed-vegetation path now computes exact `veget` from
`slowproc_init`/`slowproc_veget`: missing LAI is still `val_exp` during the
first `slowproc_veget` call, so present PFTs become full vegetation cover
before LAI is reset to zero for STOMATE. This closes the paper-case
`IMPOSE_VEG=y`, `READ_LAI=n`, `STOMATE_OK_STOMATE=y` path.

The `READ_LAI=y` cold-start branch is now closed only for an explicit
same-case `laimap(kjpindex,nvm,12)` payload that has already been read from
restart or produced by the source `slowproc_interlai` path. The JAX
`slowproc_lai_explicit` kernel follows `slowproc.f90::slowproc_lai` lines
2949-3062 for monthly `mean`/`inter` LAI selection, and
`slowproc_cold_start_vegetation_entry_state` uses that LAI in the following
`slowproc_veget` call. Arbitrary LAI-map file interpolation/regridding remains
an explicit external source boundary; it is not guessed from `veget_max` or
filled by defaults.

`stomate_history_1961.nc` contains diagnostic `RESOLUTION_X/Y`, but it does
not contain the complete `grid_stuff` geometry or the static overlap/bbox
intermediates. It is not enough to claim paper-case salinity, tide, or
soilclass parity.

With `fixed_format_trace_dir=outputs/server_1961_trace_full_20260623/traces`,
`resolution`, `neighbours`, `soilclass`, `njsc`, clay/sand/silt fractions,
bulk density, soil pH, poor-soils fraction, `salinity`, `salinity_bbox`,
`tide_height`, `tide_bbox`, `corners`, `seglength`, `soilclass_sub_index`,
and `soilclass_sub_area` are no longer missing fields. The intersurf main trace
is used to validate ccanopy and land-point boundary identity; its
meteorological values are already transformed driver boundary values and are
not used to overwrite the raw first forcing slab.

## Trace Contract

Exact-truth trace normalization may still dump or preserve these values at the
Fortran boundary for regression checks:

- after `slowproc_veget`: exact `veget`, `veget_max`, `frac_nobio`,
  `totfrac_nobio`, and `soiltile`
