# HYDROL Trace Instrumentation Plan

This note is an exact instrumentation plan for the next Fortran trace run. It
does not patch Fortran; it lists where a future, user-approved local/server
Fortran run should emit CSV columns before wiring a full JAX `hydrol_soil`
loop.

Source truth: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`.

## Local Reference/Restart Inventory

Read-only files inspected under `reference/case_001_071/OUT`:

- `sechiba_start.nc` and `sechiba_restart.nc`
- `driver_start.nc` and `driver_restart.nc`
- `stomate_start.nc`, `stomate_restart.nc`, and yearly
  `stomate_history_1961.nc` through `stomate_history_2010.nc`

Restart initial-state and boundary metadata present:

- `njsc(time,y,x)`: local sample value `2`, usable as restart texture metadata.
- `moistc(time,l_c,z_b,y,x)` and `moistcl(time,l_c,z_b,y,x)`: HYDROL total and
  liquid moisture restart state over 6 soil tiles and 11 layers.
- `free_drain_coef(time,z_c,y,x)`: local sample values all `1`.
- `veget`, `veget_max`, `humrel`, `vegstress`.
- `wtp`, `wt_ab`, `wt_ab_tide`, `zwt_force`, `water2infilt`, `ae_ns`,
  `resdist`, `evap_bare_lim_ns`.

History/diagnostic truth present:

- No local `sechiba_history*.nc` file was found in this reference case.
- `stomate_history_*.nc` files are STOMATE yearly diagnostics; they are not
  timestep HYDROL solve truth.
- `driver_start.nc` and `driver_restart.nc` contain driver restart fields such
  as `vevapp`, not HYDROL tridiagonal internals.

Missing for full solve parity:

- `mcr`, `mcs`, `z_m`, `dz_mm`, `profil_froz_hydro_ns`, `kfact_root` at the
  active solve point.
- `a`, `b`, `d`, `k`, `e`, `f`, `g1`, `ep`, `fp`, `gp`.
- `mask_soiltile`, `resolv`, `flux_top`, `rootsink`, RHS, stored equation
  arrays, and tridiagonal forward/backward diagnostics.
- `mcl_after_tridiag`, `tmci`, `tmcf`, `check_tr_ns`, `dr_corrnum_ns`, and
  final post-update `mc/mcl`.

Because the restart files are initial-state/boundary metadata rather than
timestep solve traces, no JAX reference reader was added in Phase 1G. A future
reader should be explicit-variable-only and should not infer missing process
state.

## 1. Coefficients And Setup

Fortran insertion points:

- After source-driven mineral CWRR tables are available in `hydrol_init`,
  lines 4172-4244. Include depth factors from lines 4150-4159 and depth grid
  transfer from lines 4067-4069 if tracing table construction itself.
- After `hydrol_soil_coef` is called for the active soil tile in
  `hydrol_soil`, lines 5931-5944.
- Inside or immediately after `hydrol_soil_coef`, mineral branch lines
  8248-8280.
- Immediately after `hydrol_soil_setup`, subroutine lines 8480-8568.

CSV columns:

- Run selectors: `kjit`, `ji`, `jst`, `jsl`.
- Soil/table inputs: `njsc`, `mcr`, `mcs`, `z_m`, `dz_mm`, `imin`, `imax`.
- State inputs: `mc_before_coef`, `mcl_before_coef`,
  `profil_froz_hydro_ns`, `kfact_root`.
- Selection diagnostics: `mc_used`, `bin_i`.
- Raw selected coefficients: `a_raw`, `b_raw`, `d_raw`, `k_floor_raw`,
  `k_eval_raw`.
- Scaled coefficients: `a`, `b`, `d`, `k`.
- Setup inputs/outputs: `dt_days`, `free_drain_coef`, `e`, `f`, `g1`, `ep`,
  `fp`, `gp`.

Current JAX coverage:

- Implemented: `build_mineral_cwrr_tables`, `select_mineral_cwrr_bin`,
  `hydrol_soil_coef_mineral_from_tables`,
  `hydrol_soil_setup_coefficients`.
- Still blocked for full loop: active `mc/mcl/profil_froz_hydro_ns` state must
  come from trace or audited upstream state wiring.

## 2. Before Main Tridiagonal Solve

Fortran insertion points:

- Liquid-water reconstruction before solve: lines 5899-5944.
- `mclint` and `tmci` construction: lines 5915-5929.
- `resolv = mask_soiltile > 0`: line 5950.
- Main RHS and `tmat` construction: lines 5953-5991.
- Equation store for alternate reuse: lines 5988-5996.

CSV columns:

- Selectors: `kjit`, `ji`, `jst`, `jsl`.
- Branch/mask values: `mask_soiltile`, `resolv`, `ok_freeze_cwrr`,
  `peat_hydro`.
- Pre-solve state: `mc_before_solve`, `mcl_before_solve`, `mclint`,
  `profil_froz_hydro_ns`, `mcr`, `mcs`, `njsc`.
- Flux/source inputs: `flux_top`, `rootsink_by_layer`, `rootsink_sum`, `b`.
- Matrix/RHS values: `e`, `f`, `g1`, `ep`, `fp`, `gp`, `tmat_e`, `tmat_f`,
  `tmat_g1`, `rhs`.
- Stored equations: `srhs`, `stmat_e`, `stmat_f`, `stmat_g1`.
- Conservation input: `tmci`.

Current JAX coverage:

- Implemented: `hydrol_soil_rhs_main`.
- Blocked: generic `mcl` reconstruction and `tmci` integration are not wired
  into a full state step; they require exact upstream state and trace columns
  to avoid silently choosing an inactive branch.

## 3. Main Tridiagonal Solve

Fortran insertion points:

- Main call site: line 6004.
- Solver subroutine `hydrol_soil_tridiag`: lines 8147-8198.

CSV columns:

- Selectors: `kjit`, `ji`, `jst`, `jsl`.
- Inputs: `resolv`, `rhs`, `tmat_e`, `tmat_f`, `tmat_g1`,
  `mcl_before_tridiag`.
- Outputs: `mcl_after_tridiag`.
- Optional solver diagnostics: `bet`, `gam`.

Current JAX coverage:

- Implemented: `hydrol_soil_tridiag_solve`.
- Blocked: no parity claim beyond unit/dense-system tests until trace emits
  `rhs/tmat/mcl_after_tridiag` for the active timestep.

## 4. Drainage And Conservation Correction

Fortran insertion points:

- Bottom drainage before correction: lines 6007-6016.
- `tmcf` after solve: lines 6018-6025.
- `diag_tr` and `check_tr_ns`: lines 6027-6033.
- Drainage correction: lines 6042-6047.
- Optional post-correction recheck: lines 6050-6052.

CSV columns:

- Selectors: `kjit`, `ji`, `jst`.
- Inputs: `k_bottom`, `free_drain_coef`, `dt_days`, `mask_soiltile`,
  `resolv`, `flux_top`, `rootsink_sum`, `tmci`, `tmcf`.
- Before/after correction: `dr_ns_before_corr`, `check_tr_ns`,
  `dr_corrnum_ns`, `dr_ns_after_corr`.
- Optional recheck: `check_tr_ns_after_corr`.

Current JAX coverage:

- Implemented: `bottom_drainage`, `drainage_correction`, optional drainage
  fields in `hydrol_soil_solve_diagnostics`.
- Blocked: `tmci/tmcf/check_tr_ns` should be traced before full water-balance
  wiring.

## 5. Post-Solve Total/Liquid Moisture Update

Fortran insertion points:

- Main post-diffusion update begins at lines 6063-6075.
- Active mineral non-peat formula is lines 6070-6071.

CSV columns:

- Selectors: `kjit`, `ji`, `jst`, `jsl`.
- Inputs: `mc_before_update`, `mcl_after_tridiag`,
  `profil_froz_hydro_ns`, `mcr`, `mcs`, `njsc`, `ok_freeze_cwrr`,
  `peat_hydro`.
- Outputs: `mc_after_update`, `mcl_after_update`.

Current JAX coverage:

- Not implemented. This step crosses from the liquid solve back to total
  water state and must wait for exact trace/state.

## 6. Alternate Residual-Boundary Path

Fortran insertion points:

- Alternate RHS without `rootsink`: lines 6971-6993.
- Alternate equation store: lines 6994-7002.
- First alternate solve call: line 7005.
- Trigger condition: lines 7011-7018.
- Equation reset and residual top boundary: lines 7020-7040.
- Residual-boundary solve call: line 7042.
- Post-alternate `mc` reconstruction: lines 7044-7058.

CSV columns:

- Selectors: `kjit`, `ji`, `jst`, `jsl`.
- Trigger inputs: `mcl_top_after_first_solve`, `mcr`, `mcr_peat`,
  `njsc`, `flux_top`, `min_sechiba`, `peat_hydro`, `resolv_alternate`.
- Alternate RHS inputs/outputs: `mcl_before_alternate`, `b`,
  `free_drain_coef`, `dt_days`, `rhs_alternate`, `tmat_alternate_e`,
  `tmat_alternate_f`, `tmat_alternate_g1`.
- Saved equations: `srhs`, `stmat_e`, `stmat_f`, `stmat_g1`.
- Residual-boundary equations: `rhs_residual`, `tmat_residual_e`,
  `tmat_residual_f`, `tmat_residual_g1`.
- Solve/update outputs: `mcl_after_first_alternate_solve`,
  `mcl_after_residual_solve`, `mc_after_residual_update`.

Current JAX coverage:

- Implemented: `hydrol_soil_residual_boundary_rhs` and
  `hydrol_soil_tridiag_solve`.
- Blocked: trigger and post-update wiring require exact trace confirming
  whether this branch activates for the paper-case active path.

## Minimal Next Trace Patch

For the smallest trace that can unlock full `hydrol_soil` integration, emit one
row per `(kjit, ji, jst, jsl)` for the active paper-case soil tile and all 11
layers, plus one row per `(kjit, ji, jst)` for tile-level drainage fields:

1. After `hydrol_soil_coef` and `hydrol_soil_setup`: `mc_before_solve`,
   `mcl_before_solve`, `profil_froz_hydro_ns`, `a`, `b`, `d`, `k`, `e`, `f`,
   `g1`, `ep`, `fp`, `gp`, `free_drain_coef`.
2. Immediately before line 6004: `mask_soiltile`, `resolv`, `flux_top`,
   `rootsink_by_layer`, `rhs`, `tmat_e`, `tmat_f`, `tmat_g1`, `tmci`.
3. Immediately after line 6004 and before line 6042: `mcl_after_tridiag`,
   `tmcf`, `dr_ns_before_corr`, `check_tr_ns`.
4. Immediately after line 6047: `dr_corrnum_ns`, `dr_ns_after_corr`.
5. Around lines 6971-7044: the alternate trigger values and saved/restored
   equations listed above, even if the branch is inactive, so the inactive
   status is source-observable rather than assumed.
