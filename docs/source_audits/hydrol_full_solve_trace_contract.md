# HYDROL Full-Solve Trace Contract

This contract lists the exact Fortran trace columns needed before wiring a
full `hydrol_soil` path in JAX. It is ordered by the active mineral CWRR solve
segment audited for `traces/hydrol_1961_v9`.

Source truth: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`.

## 1. Coefficients And Setup

Fortran spans:

- `hydrol_init` mineral CWRR table builder: lines 4172-4244
- depth factors in `hydrol_init`: lines 4150-4159
- depth transfer to HYDROL arrays: lines 4067-4069
- `hydrol_soil_coef` mineral branch: lines 8248-8280
- `hydrol_soil_setup`: lines 8480-8568

Required trace columns:

- `kjit`, `jst`, `ji`
- `njsc`, `mcr`, `mcs`
- `z_m`, `dz_mm`
- `mc_before_coef`, `mcl_before_coef`, `profil_froz_hydro_ns`
- `kfact_root`
- `mc_used`, `bin_i`
- `a_raw`, `b_raw`, `d_raw`, `k_floor_raw`, `k_eval_raw`
- `a`, `b`, `d`, `k`
- `free_drain_coef`
- setup coefficients: `e`, `f`, `g1`, `ep`, `fp`, `gp`

Current JAX coverage:

- `build_mineral_cwrr_tables`
- `select_mineral_cwrr_bin`
- `hydrol_soil_coef_mineral_from_tables`
- `hydrol_soil_setup_coefficients`

## 2. Pre-Solve State And RHS

Fortran spans:

- `mcl` reconstruction before solve: lines 5899-5944
- total liquid water before solve `tmci`: lines 5923-5929
- `resolv = mask_soiltile > 0`: line 5950
- main RHS and left matrix construction: lines 5953-5991
- equation store for alternate reuse: lines 5992-5999

Required trace columns:

- `mask_soiltile`, `resolv`
- `mc_before_solve`, `mcl_before_solve`
- `mclint`
- `tmci`
- `flux_top`
- `rootsink_by_layer`, `rootsink_sum`
- `b`
- `e`, `f`, `g1`, `ep`, `fp`, `gp`
- `rhs`
- stored equations: `srhs`, `stmat_e`, `stmat_f`, `stmat_g1`

Current JAX coverage:

- `hydrol_soil_rhs_main`
- no full `mcl` reconstruction kernel yet; needs source-backed input state and
  trace values for `mc`, `profil_froz_hydro_ns`, `mcr`, and active branch.

## 3. Solve

Fortran spans:

- main call: line 6004
- `hydrol_soil_tridiag`: lines 8147-8198

Required trace columns:

- `rhs`
- `tmat_e`, `tmat_f`, `tmat_g1`
- `resolv`
- `mcl_before_tridiag`
- `mcl_after_tridiag`
- optional diagnostics: `bet`, `gam`

Current JAX coverage:

- `hydrol_soil_tridiag_solve`

## 4. Drainage And Conservation Correction

Fortran spans:

- bottom drainage before correction: lines 6007-6016
- total liquid water after solve `tmcf`: lines 6018-6025
- conservation check: lines 6027-6033
- drainage correction: lines 6042-6047
- optional recheck: lines 6050-6052

Required trace columns:

- `k_bottom`
- `free_drain_coef`
- `dt_days`
- `mask_soiltile`, `resolv`
- `dr_ns_before_corr`
- `tmci`, `tmcf`
- `flux_top`
- `rootsink_sum`
- `check_tr_ns`
- `dr_corrnum`
- `dr_after_corr`

Current JAX coverage:

- `bottom_drainage`
- `drainage_correction`
- no generic `tmci/tmcf` integration kernel yet; formulas should be audited
  separately before full water-balance wiring.

## 5. Post-Solve Liquid/Frozen State Update

Fortran spans:

- main post-diffusion `mc` reconstruction: starts after line 6063 and active
  formulas appear at lines 6067-6070
- later alternate-path reconstruction: lines 7046-7058

Required trace columns:

- `mc_before_update`
- `mcl_after_tridiag`
- `profil_froz_hydro_ns`
- `mcr`
- `mc_after_update`

Current JAX coverage:

- not implemented. This must wait for exact trace/state because it crosses
  from liquid solve outputs back to total water state.

## 6. Alternate Residual-Boundary Path

Fortran spans:

- alternate bare-soil RHS without rootsink: lines 6971-6997
- first alternate solve call: line 7005
- trigger condition: lines 7010-7017
- equation reset and top residual boundary: lines 7018-7044
- post-alternate `mc` reconstruction: lines 7046-7058

Required trace columns:

- `mcl_top_after_first_solve`
- `mcr`
- `flux_top`
- `min_sechiba`
- `resolv_alternate`
- saved equations: `srhs`, `stmat_e`, `stmat_f`, `stmat_g1`
- residual boundary equations: `rhs_residual`, `tmat_residual_e`,
  `tmat_residual_f`, `tmat_residual_g1`
- `mcl_after_residual_solve`
- `mc_after_residual_update`

Current JAX coverage:

- `hydrol_soil_residual_boundary_rhs`
- `hydrol_soil_tridiag_solve`
- trigger and post-update are not wired into a full path until trace confirms
  whether and where the branch activates.

## 7. 2026-06-23 List-Directed Full Trace Slice

Trace package:

- `outputs/server_1961_trace_full_20260623/traces/orchjax_hydrol_main_trace.txt`
- `outputs/server_1961_trace_full_20260623/traces/orchjax_hydrol_post_trace.txt`
- `outputs/server_1961_trace_full_20260623/traces/orchjax_hydrol_update_trace.txt`
- `outputs/server_1961_trace_full_20260623/traces/orchjax_hydrol_alt_trace.txt`
- `outputs/server_1961_trace_full_20260623/traces/orchjax_hydrol_alt_residual_trace.txt`

These files are Fortran list-directed records, not CSV. The trace writer is
recorded in `outputs/server_trace_patch/apply_trace_patch.py`, and the current
JAX reader streams first matches instead of loading the 795 MB main file.

Record schemas:

- `pre`: `kjit`, `ji`, `jst`, `jsl`, `njsc`, `resolv`,
  `mask_soiltile`, `mc`, `mcl`, `mclint`, `profil_froz_hydro_ns`,
  `mcr`, `mcs`, `a`, `b`, `d`, `k`, `e`, `f`, `g1`, `ep`,
  `fp`, `gp`, `rhs`, `tmat_e`, `tmat_f`, `tmat_g1`, `rootsink`,
  `tmci`, `flux_top`, `free_drain_coef`, `dt_days`.
  Provenance: emitted after RHS/tmat store around `hydrol_soil` lines
  5899-5996.
- `post_tile`: `kjit`, `ji`, `jst`, `resolv`, `tmci`, `tmcf`,
  `flux_top`, `rootsink_sum`, corrected `dr_ns`, `dr_corrnum_ns`,
  `check_tr_ns`, `k_bottom`, `free_drain_coef`, `dt_days`.
  Provenance: emitted after drainage correction at lines 6007-6052.
- `post_layer`: `kjit`, `ji`, `jst`, `jsl`, `mcl`, `mc`,
  `profil_froz_hydro_ns`, `mcr`, `mcs`. Provenance: same post-correction
  insertion before the post-solve total-moisture update.
- `mc_after_update`: `kjit`, `ji`, `jst`, `jsl`, `mc`, `mcl`,
  `profil_froz_hydro_ns`, `mcr`, `mcs`. Provenance: emitted immediately after
  the main update at lines 6063-6075 and before over-saturation smoothing.
- `alt_first_solve`: `kjit`, `ji`, `jst`, `mcl_top_after_first_solve`,
  `flux_top`, `mcr`, `min_sechiba`. Provenance: after the first alternate
  tridiagonal solve at line 7005.
- `alt_residual`: `kjit`, `ji`, `jst`, `jsl`, `resolv`, `rhs`, `tmat_e`,
  `tmat_f`, `tmat_g1`, `mcl_after_residual_solve`. Provenance: after the
  residual-boundary solve at line 7042.

Closed JAX slice:

- `load_first_hydrol_full_step_slice` streams one requested
  `(kjit, ji, jst)` profile, all 11 layers.
- `discover_hydrol_full_step_sample_keys` streams
  `orchjax_hydrol_main_trace.txt`, inspects only first-layer active tile4
  records, and stops at the requested sample limit. The default sampler
  returns two profiles: `(kjit=1, ji=1, jst=4)` and
  `(kjit=2, ji=1, jst=4)`.
- `load_hydrol_full_step_sample_slices` loads those keys with first-match
  streaming across the main/post/update trace files; it does not materialize
  the 795 MB main trace.
- `hydrol_trace_backed_full_step` composes the existing RHS, tridiagonal,
  drainage/correction, and post-update kernels against this explicit trace.
- Test: `tests/parity/test_hydrol_1961_full_trace_slice.py`.
- Current maximum absolute errors:
  - `kjit=1`: `4.121147867408581e-13`.
  - `kjit=2`: `6.536993168992922e-13`.
- Checked fields: `rhs`, `mcl_after_tridiag`, `mc_after_update`, `tmci`,
  `tmcf`, `check_tr_ns`, `dr_before_corr`, `dr_corrnum`, and
  `dr_after_corr`.

Update/post-solve closure:

- `mc_after_update` records are emitted immediately after source lines
  6063-6075. The adapter validates these records with
  `hydrol_mcl_to_mc_after_solve`; for the first two tile4 samples the maximum
  post-update error is about `1.35e-14`.
- `post_tile` records include `tmci`, `tmcf`, `check_tr_ns`,
  corrected `dr_ns`, and `dr_corrnum_ns`. The adapter validates `tmci/tmcf`
  through the source layer integration formula at lines 5923-5929 and
  6018-6025, and validates `check_tr_ns`/correction through lines 6027-6047.
- `post_layer` records are after drainage correction but before
  `mc_after_update`; they are used as `mcl_after_tridiag` truth for the main
  solve, not as final-smoothed state truth.

Alternate residual closure:

- `load_hydrol_alt_residual_sample_slices` loads a two-sample branch set by
  first-match streaming: first inactive tile4 residual profile and first active
  tile4 residual profile.
- `load_first_alt_residual_slice(..., resolv=False)` finds the first inactive
  residual sample, `(kjit=1, ji=1, jst=4)`.
- `load_first_alt_residual_slice(..., resolv=True)` finds the first active
  residual sample, `(kjit=541, ji=1, jst=4)`.
- `hydrol_trace_backed_alt_residual` validates:
  - trigger condition from source lines 7011-7018,
    `mcl_top_after_first_solve < mcr` and `flux_top > min_sechiba`;
  - residual top boundary reset from lines 7031-7039,
    `rhs(1)=mcr`, `tmat(1,2)=1`, `tmat(1,3)=0`;
  - for inactive samples, top-layer preservation through the tridiagonal
    `resolv` guard at lines 8171-8185 and 8191-8195;
  - for active samples, top-layer result equals `mcr`.
- Both active and inactive residual checks currently close with zero recorded
  error for the fields above.

Remaining diagnostic trace limitations are explicit:

- The list-directed trace does not emit `z_m`, `dz_mm`, `kfact_root`,
  `mc_used`, `bin_i`, raw table coefficients, or `k_eval_raw`, so coefficient
  setup cannot be independently rebuilt from this trace slice.
- `bet` and `gam` are not emitted, so solver parity is based on final
  `mcl_after_tridiag`.
- `dr_ns_before_corr` is not emitted; it is reconstructed from source lines
  6007-6016 and checked against corrected `dr_ns`.
- Post-smoothing and under-residual routing remain blocked without
  `ru_corr_ns`, `dr_corr_ns`, `check_over_ns`, `is_under_mcr`, and
  `check_under_ns`.
- Alternate residual branch wiring remains partial beyond the top-boundary
  check because saved equations, first alternate per-layer solve outputs, and
  `mc_after_residual_update` are not emitted.

These limitations describe what the historical trace slice cannot
independently diagnose inside HYDROL. They are not a claim that the current
production driver has a HYDROL-to-STOMATE process gap: the source-backed
HYDROL module and driver orchestration tests cover the active paper-case
exports, while this trace contract remains a narrow debugging asset for
coefficient/solver internals.
