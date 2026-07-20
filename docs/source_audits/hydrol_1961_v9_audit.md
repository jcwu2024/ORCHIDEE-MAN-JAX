# HYDROL 1961 v9 Trace Audit

Reference trace:

- `traces/hydrol_1961_v9/jcwu_hydrol_soil_coef_tile4_bottom_v8.csv`
- `traces/hydrol_1961_v9/jcwu_hydrol_tile4_trace.csv`

Fortran source truth:

- `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`

## Audited Line Spans

- `hydrol_soil`: `hydrol.f90` lines 5405-7402
- `hydrol_soil_coef`: `hydrol.f90` lines 8223-8312
- `hydrol_soil_setup`: `hydrol.f90` lines 8480-8568
- `hydrol_soil_tridiag`: `hydrol.f90` lines 8147-8198
- `hydrol_soil_flux`: `hydrol.f90` lines 8053-8126
- `hydrol_diag_soil`: `hydrol.f90` lines 8948-9288
- mineral CWRR table builder in `hydrol_init`: `hydrol.f90` lines 4172-4244

Related constants:

- `imin=1`, `nbint=50`, `imax=51`, `w_time=1`:
  `src_parameters/constantes_soil_var.f90` lines 347-350
- `nslm=11`: `src_parameters/control.f90` lines 528-529

## Trace-Covered Execution Order

The current trace targets soil tile `jst=4`, bottom layer `jsl=11`, mineral CWRR.

1. `ok_freeze_cwrr` prepares `profil_froz_hydro_ns`: lines 5665-5689
2. `hydrol_split_soil`: line 5698
3. soil tile loop starts: lines 5760-5794
4. surface water flux setup: lines 5800-5847
5. first `hydrol_soil_coef`: line 5858
6. `hydrol_soil_infilt`: line 5864
7. second `hydrol_soil_coef`: line 5891
8. `hydrol_soil_setup`: line 5896
9. freeze-aware `mcl` setup and third `hydrol_soil_coef`: lines 5899-5945
10. RHS and matrix assembly: lines 5958-5996
11. `hydrol_soil_tridiag`: line 6004
12. bottom drainage before correction: lines 6007-6016
13. drainage conservation correction: lines 6018-6052
14. frozen/liquid water update: lines 6063-6075
15. over-saturation smoothing: line 6083
16. negative runoff correction into drainage: lines 6102-6114
17. optional `hydrol_soil_flux` water check: line 6338
18. freeze diagnostic `kk/kk_moy`: lines 6746-6750
19. `hydrol_diag_soil`: line 6802

## CSV Semantics

### `jcwu_hydrol_soil_coef_tile4_bottom_v8.csv`

Rows: 1200. Columns: 30.

All rows are:

- `ins=4`
- `jsl=11`
- `branch_peat=F`
- `ok_freeze_cwrr=T`
- `peat_hydro=F`

Important columns:

- `mc`: `mc(ji,jsl,ins)`
- `mcl`: trace context; not used by `hydrol_soil_coef`
- `profil_froz`: `profil_froz_hydro_ns(ji,jsl,ins)`
- `x`: `1 - profil_froz`
- `mc_used`: mineral branch water content used by `hydrol_soil_coef`
- `bin_i`: mineral lookup bin
- `kfact_root`: root scaling factor
- `a_raw`, `b_raw`, `d_raw`: unscaled linear coefficients
- `k_floor_raw`: lower conductivity floor
- `k_eval_raw`: `a_raw * mc_used + b_raw`
- `k_result`: `max(k_floor_raw, k_eval_raw)`
- `a_scaled`, `b_scaled`, `d_scaled`: coefficients written to hydrology state
- `peat_*`: non-executed branch debug values; do not use as parity targets

First-row anchors:

- `k_result = 6.0648789143219517`
- `a_raw = 175.91824059050936`
- `b_raw = -46.710593262830855`
- `d_raw = 27504.887686410824`

### `jcwu_hydrol_tile4_trace.csv`

Rows: 144. Columns: 19.

All rows are:

- `jst=4`
- `dt_days=1/48`
- `free_drain_coef=1`

Important columns:

- `dr_ns_before_corr_at_6012`: drainage before correction at line 6012
- `mc_6012_bottom`: bottom-layer `mc`
- `mcl_6012_bottom`: bottom-layer `mcl`
- `profil_froz_6012_bottom`: bottom-layer frozen profile fraction
- `mc_used_6012_bottom`: mineral `hydrol_soil_coef` input water content
- `k_6012_bottom`: bottom-layer `k`
- `free_drain_coef_6012`: bottom drainage coefficient
- `dt_days`: `dt_sechiba / one_day`
- `check_tr_ns_6047`: conservation check used by correction
- `dr_corrnum_ns_6047`: correction amount
- `dr_after_corr_6047`: corrected drainage

First-row anchors:

- `k_6012_bottom = 6.0648789143219517`
- `dr_ns_before_corr_at_6012 = 0.12635164404837398`
- `dr_after_corr_6047 = 0.12560049299877521`

Notes:

- `bin_6012_bottom` does not match the `hydrol_soil_coef` `bin_i` column and
  should not be used as a line 8275 parity target.
- `k_floor_6012_bottom`, `k_table_bin_6012_bottom`,
  `a_lin_6012_bottom`, and `b_lin_6012_bottom` are sentinel-like values in
  this trace and should not be used as parity targets.

## First Implementation TODO

1. Implement mineral-only CWRR table construction from `hydrol_init`
   lines 4172-4244.
2. Implement mineral `hydrol_soil_coef` for the active trace branch:
   `ok_freeze_cwrr=T`, `peat_hydro=F`, `branch_peat=F`; source lines
   8248-8280.
3. Implement bottom drainage algebra from source lines 6007-6016.
4. Implement drainage correction algebra from source lines 6042-6047.
5. Add lightweight parity tests against both trace CSV files.

Initial test anchors:

- `k_result == 6.0648789143219517`
- `dr_before == 0.12635164404837398`
- `dr_after == dr_before + dr_corrnum`
