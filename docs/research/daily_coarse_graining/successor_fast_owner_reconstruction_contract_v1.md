# Successor Fast-Owner Reconstruction Contract v1

Status: **schema implemented in the all-Teacher materializer; neural decoder is not implemented**.

Date: 2026-08-09.

Machine contract: `manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json`

Contract SHA256: `4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9`

## Purpose

The existing 101 source-backed process outputs are preserved. This contract
adds the missing canonical fast-owner reconstruction layer required to rebuild
all 160 leaves and 2,855 columns of the accepted v6 fast-day boundary before
the retained exact daily tail runs.

The new endpoints do not directly predict biomass, NPP, allocation, or
mortality. They reconstruct day-end fast state and daily auxiliary values.
The retained exact processes continue to update biomass and productivity from
the named fast GPP, maintenance respiration, water, thermal, and carbon
inventory endpoints.

This design reads no v5 data, regenerates no v6 data, reconstructs no 48-step
forcing sequence, and performs no 48-step state scan.

## Frozen Data Identity

- Dataset: `pft14-daily-teacher-669-1961-2010`
- Dataset manifest SHA256: `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`
- Markov contract SHA256: `6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442`
- SECHIBA state source-contract SHA256: `9ec8b89a9c1ff1988d7571d0ba25bb3310520f27aaf2ea0d7fea6c942cc235bd`

## Complete Ownership

| Role | Leaves | Scalar columns |
| --- | ---: | ---: |
| `existing_process_endpoint` | 21 | 1482 |
| `deterministic_source_exact_derivation` | 19 | 95 |
| `mirror_of_canonical_owner` | 30 | 369 |
| `true_guarded_or_static_carry` | 11 | 70 |
| `independent_neural_endpoint` | 79 | 839 |
| **Total** | **160** | **2855** |

Every leaf has exactly one role. Mirrors must point directly to a non-mirror
canonical owner, so mirror chains and cycles are forbidden. The JSON contract
contains the owner, source provenance, unit policy, constraint, mask policy,
shape, axes, and Teacher slice for all 160 leaves.

## Added Neural Boundary

- 79 named value heads spanning 839 scalar values.
- One `rveget` defined-logit head spanning 2 compact-PFT scalar logits.
- Total added scalar output width: 841.
- Proposed route domains: 42 water heads, 21 energy heads, 16 carbon heads.
- Candidate and matched control must use the identical added schema, masks,
  losses, normalization, and materializer.
- Fast-state endpoints use normalized day-end-minus-day-start-owner targets.
  Daily summaries and OK_LEAK auxiliaries use normalized absolute targets.
- All values remain in Teacher-native units. Constraint transforms are applied
  only after their exact decoder form is approved.

### Architecture impact requiring review

Under the current source-routed implementation, the 79 value heads plus the
mask head would add 80 field queries. They would participate in the same
source-admitted process graph: content-dependent attention in the candidate
and fixed graph mixing in the control. They do not add cross-day memory.

The decoder is coordinate-shared, so 841 scalar outputs do not imply an
`841 x d_model` dense layer. With `d_model=256`, 80 query/output embeddings
add 40,960 parameters. If the proposed eight monitoring/loss groups are
kept as separate label embeddings, the estimated total increase is
43,008 parameters, from 8,679,681 to
8,722,689 per arm (about
0.496%). This estimate is not an implementation
authorization; the query interaction, label/loss grouping, and mask decoder
must be approved first.

## New Value Heads

`unit_policy` is `teacher_native_no_conversion` for every row.

| Teacher leaf | Output key | Shape | Axes | Domain | Constraint | Mask | Teacher slice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `driver_previous_step_state.albedo` | `endpoint.fast_state.driver_previous_step_state.albedo` | `1 x 2` | `npts, spectral_band_2` | `energy` | `unit_interval` | Teacher-defined + static contract mask | `[0:2]` |
| `driver_previous_step_state.fluxsens` | `endpoint.fast_state.driver_previous_step_state.fluxsens` | `1` | `npts` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[4:5]` |
| `driver_previous_step_state.vevapp` | `endpoint.fast_state.driver_previous_step_state.vevapp` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[13:14]` |
| `diffuco_previous_step_state.cimean` | `endpoint.fast_state.diffuco_previous_step_state.cimean` | `1 x 2` | `npts, nvm` | `carbon` | `finite_teacher_native` | Teacher-defined + static contract mask | `[15:17]` |
| `diffuco_previous_step_state.control_inudate` | `endpoint.fast_state.diffuco_previous_step_state.control_inudate` | `1` | `npts` | `carbon` | `unit_interval` | Teacher-defined + static contract mask | `[17:18]` |
| `diffuco_previous_step_state.drysoil_frac` | `endpoint.fast_state.diffuco_previous_step_state.drysoil_frac` | `1` | `npts` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[19:20]` |
| `diffuco_previous_step_state.evapot` | `endpoint.fast_state.diffuco_previous_step_state.evapot` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[21:22]` |
| `diffuco_previous_step_state.evapot_corr` | `endpoint.fast_state.diffuco_previous_step_state.evapot_corr` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[22:23]` |
| `diffuco_previous_step_state.gpp` | `endpoint.fast_state.diffuco_previous_step_state.gpp` | `1 x 2` | `npts, nvm` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[24:26]` |
| `diffuco_previous_step_state.gsmean` | `endpoint.fast_state.diffuco_previous_step_state.gsmean` | `1 x 2` | `npts, nvm` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[26:28]` |
| `diffuco_previous_step_state.qsurf` | `endpoint.fast_state.diffuco_previous_step_state.qsurf` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[36:37]` |
| `diffuco_previous_step_state.roughheight` | `endpoint.fast_state.diffuco_previous_step_state.roughheight` | `1` | `npts` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[37:38]` |
| `diffuco_previous_step_state.roughheight_pft` | `endpoint.fast_state.diffuco_previous_step_state.roughheight_pft` | `1 x 2` | `npts, nvm` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[38:40]` |
| `diffuco_previous_step_state.rstruct` | `endpoint.fast_state.diffuco_previous_step_state.rstruct` | `1 x 2` | `npts, nvm` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[40:42]` |
| `diffuco_previous_step_state.rveget` | `endpoint.fast_state.diffuco_previous_step_state.rveget` | `1 x 2` | `npts, nvm` | `carbon` | `dynamic_defined_nonnegative` | dynamic value mask + eligible-slot BCE | `[42:44]` |
| `diffuco_previous_step_state.temp_sol_pft` | `endpoint.fast_state.diffuco_previous_step_state.temp_sol_pft` | `1 x 2` | `npts, nvm` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[66:68]` |
| `diffuco_previous_step_state.z0h` | `endpoint.fast_state.diffuco_previous_step_state.z0h` | `1` | `npts` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[72:73]` |
| `diffuco_previous_step_state.z0m` | `endpoint.fast_state.diffuco_previous_step_state.z0m` | `1` | `npts` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[73:74]` |
| `enerbil_previous_step_state.soilcap_pft` | `endpoint.fast_state.enerbil_previous_step_state.soilcap_pft` | `1 x 2` | `npts, nvm` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[206:208]` |
| `enerbil_previous_step_state.soilflx_pft` | `endpoint.fast_state.enerbil_previous_step_state.soilflx_pft` | `1 x 2` | `npts, nvm` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[209:211]` |
| `hydrol_previous_step_state.ae_ns` | `endpoint.fast_state.hydrol_previous_step_state.ae_ns` | `1 x 6` | `npts, nstm` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[215:221]` |
| `hydrol_previous_step_state.canopy2ground` | `endpoint.fast_state.hydrol_previous_step_state.canopy2ground` | `1 x 2` | `npts, nvm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[221:223]` |
| `hydrol_previous_step_state.drainage_per_soil` | `endpoint.fast_state.hydrol_previous_step_state.drainage_per_soil` | `1 x 6` | `npts, nstm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[229:235]` |
| `hydrol_previous_step_state.evap_bare_lim` | `endpoint.fast_state.hydrol_previous_step_state.evap_bare_lim` | `1` | `npts` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[235:236]` |
| `hydrol_previous_step_state.evap_bare_lim_ns` | `endpoint.fast_state.hydrol_previous_step_state.evap_bare_lim_ns` | `1 x 6` | `npts, nstm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[236:242]` |
| `hydrol_previous_step_state.fwet_new` | `endpoint.fast_state.hydrol_previous_step_state.fwet_new` | `1` | `npts` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[249:250]` |
| `hydrol_previous_step_state.grndflux` | `endpoint.fast_state.hydrol_previous_step_state.grndflux` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[250:251]` |
| `hydrol_previous_step_state.humrel` | `endpoint.fast_state.hydrol_previous_step_state.humrel` | `1 x 2` | `npts, nvm` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[251:253]` |
| `hydrol_previous_step_state.humrelv` | `endpoint.fast_state.hydrol_previous_step_state.humrelv` | `1 x 2 x 6` | `npts, nvm, nstm` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[253:265]` |
| `hydrol_previous_step_state.nroot` | `endpoint.fast_state.hydrol_previous_step_state.nroot` | `1 x 2 x 11` | `npts, nvm, nslm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[397:419]` |
| `hydrol_previous_step_state.precip2canopy` | `endpoint.fast_state.hydrol_previous_step_state.precip2canopy` | `1 x 2` | `npts, nvm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[419:421]` |
| `hydrol_previous_step_state.precip2ground` | `endpoint.fast_state.hydrol_previous_step_state.precip2ground` | `1 x 2` | `npts, nvm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[421:423]` |
| `hydrol_previous_step_state.profil_froz_hydro_ns` | `endpoint.fast_state.hydrol_previous_step_state.profil_froz_hydro_ns` | `1 x 11 x 6` | `npts, nslm, nstm` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[423:489]` |
| `hydrol_previous_step_state.run2man` | `endpoint.fast_state.hydrol_previous_step_state.run2man` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[491:492]` |
| `hydrol_previous_step_state.run2peat` | `endpoint.fast_state.hydrol_previous_step_state.run2peat` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[492:493]` |
| `hydrol_previous_step_state.runoff2peat` | `endpoint.fast_state.hydrol_previous_step_state.runoff2peat` | `1 x 6` | `npts, nstm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[499:505]` |
| `hydrol_previous_step_state.runoff_per_soil` | `endpoint.fast_state.hydrol_previous_step_state.runoff_per_soil` | `1 x 6` | `npts, nstm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[505:511]` |
| `hydrol_previous_step_state.shumdiag_peat` | `endpoint.fast_state.hydrol_previous_step_state.shumdiag_peat` | `1 x 11` | `npts, nslm` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[511:522]` |
| `hydrol_previous_step_state.snow_age` | `endpoint.fast_state.hydrol_previous_step_state.snow_age` | `1` | `npts` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[523:524]` |
| `hydrol_previous_step_state.snow_nobio_age` | `endpoint.fast_state.hydrol_previous_step_state.snow_nobio_age` | `1 x 1` | `npts, nnobio` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[525:526]` |
| `hydrol_previous_step_state.snowdz` | `endpoint.fast_state.hydrol_previous_step_state.snowdz` | `1 x 3` | `npts, nsnow` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[526:529]` |
| `hydrol_previous_step_state.snowgrain` | `endpoint.fast_state.hydrol_previous_step_state.snowgrain` | `1 x 3` | `npts, nsnow` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[529:532]` |
| `hydrol_previous_step_state.snowmelt` | `endpoint.fast_state.hydrol_previous_step_state.snowmelt` | `1` | `npts` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[538:539]` |
| `hydrol_previous_step_state.snowrho` | `endpoint.fast_state.hydrol_previous_step_state.snowrho` | `1 x 3` | `npts, nsnow` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[539:542]` |
| `hydrol_previous_step_state.soilflxresid` | `endpoint.fast_state.hydrol_previous_step_state.soilflxresid` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[611:612]` |
| `hydrol_previous_step_state.subsnownobio` | `endpoint.fast_state.hydrol_previous_step_state.subsnownobio` | `1 x 1` | `npts, nnobio` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[612:613]` |
| `hydrol_previous_step_state.temp_hydro` | `endpoint.fast_state.hydrol_previous_step_state.temp_hydro` | `1 x 11` | `npts, nslm` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[613:624]` |
| `hydrol_previous_step_state.temp_sol_add` | `endpoint.fast_state.hydrol_previous_step_state.temp_sol_add` | `1` | `npts` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[624:625]` |
| `hydrol_previous_step_state.us` | `endpoint.fast_state.hydrol_previous_step_state.us` | `1 x 2 x 6 x 11` | `npts, nvm, nstm, nslm` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[625:757]` |
| `hydrol_previous_step_state.wat_flux` | `endpoint.fast_state.hydrol_previous_step_state.wat_flux` | `1 x 11 x 6` | `npts, nslm, nstm` | `water` | `finite_teacher_native` | Teacher-defined + static contract mask | `[757:823]` |
| `hydrol_previous_step_state.water2infilt` | `endpoint.fast_state.hydrol_previous_step_state.water2infilt` | `1 x 6` | `npts, nstm` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[823:829]` |
| `hydrol_previous_step_state.wt_ab` | `endpoint.fast_state.hydrol_previous_step_state.wt_ab` | `1` | `npts` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[829:830]` |
| `hydrol_previous_step_state.wt_ab_tide` | `endpoint.fast_state.hydrol_previous_step_state.wt_ab_tide` | `1` | `npts` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[830:831]` |
| `thermosoil_previous_step_state.cgrnd` | `endpoint.fast_state.thermosoil_previous_step_state.cgrnd` | `1 x 31 x 2` | `npts, ngrnd, nvm` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[837:899]` |
| `thermosoil_previous_step_state.cgrnd_snow` | `endpoint.fast_state.thermosoil_previous_step_state.cgrnd_snow` | `1 x 3` | `npts, nsnow` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[899:902]` |
| `thermosoil_previous_step_state.dgrnd` | `endpoint.fast_state.thermosoil_previous_step_state.dgrnd` | `1 x 31 x 2` | `npts, ngrnd, nvm` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[902:964]` |
| `thermosoil_previous_step_state.dgrnd_snow` | `endpoint.fast_state.thermosoil_previous_step_state.dgrnd_snow` | `1 x 3` | `npts, nsnow` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[964:967]` |
| `thermosoil_previous_step_state.gtemp` | `endpoint.fast_state.thermosoil_previous_step_state.gtemp` | `1` | `npts` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[981:982]` |
| `thermosoil_previous_step_state.lambda_snow` | `endpoint.fast_state.thermosoil_previous_step_state.lambda_snow` | `1` | `npts` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[1046:1047]` |
| `thermosoil_previous_step_state.pcapa_en` | `endpoint.fast_state.thermosoil_previous_step_state.pcapa_en` | `1 x 32 x 2` | `npts, ngrnd, nvm` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[1047:1111]` |
| `thermosoil_previous_step_state.shum_ngrnd_permalong` | `endpoint.fast_state.thermosoil_previous_step_state.shum_ngrnd_permalong` | `1 x 32 x 2` | `npts, ngrnd, nvm` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[1207:1271]` |
| `thermosoil_previous_step_state.soilcap` | `endpoint.fast_state.thermosoil_previous_step_state.soilcap` | `1` | `npts` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[1271:1272]` |
| `thermosoil_previous_step_state.soilflx` | `endpoint.fast_state.thermosoil_previous_step_state.soilflx` | `1` | `npts` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[1272:1273]` |
| `daily_interface.humrel_daily` | `endpoint.daily_summary.humrel_daily` | `1 x 2` | `npts, nvm` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[1365:1367]` |
| `daily_interface.litterhum_daily` | `endpoint.daily_summary.litterhum_daily` | `1` | `npts` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[1367:1368]` |
| `daily_interface.resp_maint_radia` | `endpoint.daily_summary.resp_maint_radia` | `1 x 14` | `scalar` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[1393:1407]` |
| `daily_interface.snowmass_daily` | `endpoint.daily_summary.snowmass_daily` | `1` | `npts` | `water` | `nonnegative` | Teacher-defined + static contract mask | `[1408:1409]` |
| `daily_interface.soilhum_daily` | `endpoint.daily_summary.soilhum_daily` | `1 x 11` | `npts, nslm` | `water` | `unit_interval` | Teacher-defined + static contract mask | `[1409:1420]` |
| `daily_interface.wspeed_daily` | `endpoint.daily_summary.wspeed_daily` | `1` | `npts` | `energy` | `nonnegative` | Teacher-defined + static contract mask | `[1436:1437]` |
| `ok_leak.lignin_struc_above` | `endpoint.ok_leak_aux.lignin_struc_above` | `1 x 2` | `npts, nvm` | `carbon` | `unit_interval` | Teacher-defined + static contract mask | `[1569:1571]` |
| `ok_leak.lignin_struc_below` | `endpoint.ok_leak_aux.lignin_struc_below` | `1 x 2 x 32` | `npts, nvm, ndeep` | `carbon` | `unit_interval` | Teacher-defined + static contract mask | `[1571:1635]` |
| `ok_leak.litterpart` | `endpoint.ok_leak_aux.litterpart` | `1 x 2 x 2` | `npts, nvm, nlitt` | `carbon` | `unit_interval` | Teacher-defined + static contract mask | `[1635:1639]` |
| `ok_leak.dead_leaves` | `endpoint.ok_leak_aux.dead_leaves` | `1 x 2 x 2` | `npts, nvm, nlitt` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[1639:1643]` |
| `ok_leak.fuel_1hr` | `endpoint.ok_leak_aux.fuel_1hr` | `1 x 2 x 2 x 1` | `npts, nvm, nlitt, nelements` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[1643:1647]` |
| `ok_leak.fuel_10hr` | `endpoint.ok_leak_aux.fuel_10hr` | `1 x 2 x 2 x 1` | `npts, nvm, nlitt, nelements` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[1647:1651]` |
| `ok_leak.fuel_100hr` | `endpoint.ok_leak_aux.fuel_100hr` | `1 x 2 x 2 x 1` | `npts, nvm, nlitt, nelements` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[1651:1655]` |
| `ok_leak.fuel_1000hr` | `endpoint.ok_leak_aux.fuel_1000hr` | `1 x 2 x 2 x 1` | `npts, nvm, nlitt, nelements` | `carbon` | `nonnegative` | Teacher-defined + static contract mask | `[1655:1659]` |
| `final_diagnostics.temp_sol` | `endpoint.fast_state.final_diagnostics.temp_sol` | `1` | `scalar` | `energy` | `finite_teacher_native` | Teacher-defined + static contract mask | `[2814:2815]` |
| `sechiba_finalize_state.leaf_ci` | `endpoint.fast_state.sechiba_finalize_state.leaf_ci` | `1 x 2 x 20` | `npts, nvm, nlai` | `carbon` | `finite_teacher_native` | Teacher-defined + static contract mask | `[2815:2855]` |

## Typed/State Collision Exclusions

The following 14 instantaneous state endpoints must never be populated from a
same-named `typed.*` daily integral:

- `driver_previous_step_state.fluxsens` -> `endpoint.fast_state.driver_previous_step_state.fluxsens`
- `enerbil_previous_step_state.soilflx` -> `thermosoil_previous_step_state.soilflx`
- `enerbil_previous_step_state.soilflx_pft` -> `endpoint.fast_state.enerbil_previous_step_state.soilflx_pft`
- `hydrol_previous_step_state.canopy2ground` -> `endpoint.fast_state.hydrol_previous_step_state.canopy2ground`
- `hydrol_previous_step_state.drainage` -> `hydrol_previous_step_state.drainage_per_soil`
- `hydrol_previous_step_state.drainage_per_soil` -> `endpoint.fast_state.hydrol_previous_step_state.drainage_per_soil`
- `hydrol_previous_step_state.precip2canopy` -> `endpoint.fast_state.hydrol_previous_step_state.precip2canopy`
- `hydrol_previous_step_state.precip2ground` -> `endpoint.fast_state.hydrol_previous_step_state.precip2ground`
- `hydrol_previous_step_state.runoff` -> `hydrol_previous_step_state.runoff_per_soil`
- `hydrol_previous_step_state.runoff2peat` -> `endpoint.fast_state.hydrol_previous_step_state.runoff2peat`
- `hydrol_previous_step_state.runoff_per_soil` -> `endpoint.fast_state.hydrol_previous_step_state.runoff_per_soil`
- `hydrol_previous_step_state.subsnownobio` -> `endpoint.fast_state.hydrol_previous_step_state.subsnownobio`
- `hydrol_previous_step_state.wat_flux` -> `endpoint.fast_state.hydrol_previous_step_state.wat_flux`
- `hydrol_previous_step_state.water2infilt` -> `endpoint.fast_state.hydrol_previous_step_state.water2infilt`

## Gates Before Training

1. Implement a non-neural materializer and prove an all-Teacher oracle reaches
   160/160 leaves at `atol=1e-8`, `rtol=1e-10`.
2. Prove cold, ordinary later-day, and restart-year exact masks and discrete
   behavior, including the dynamic `rveget` status.
3. After explicit architecture approval, extend both arms identically and run
   local forward/loss/JVP/VJP plus one-day state and water/carbon budget checks.
4. Only then reopen a bounded `gln01` real-v6 smoke and small fit. The 7/30-day,
   365-day, long-held-point, Gate D2, and product gates remain closed.
