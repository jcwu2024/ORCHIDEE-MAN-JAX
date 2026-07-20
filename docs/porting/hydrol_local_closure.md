# HYDROL Local Closure Before Full Trace

This document summarizes the HYDROL line at the maximum local closure reached
before a new exact Fortran trace is required. The JAX code intentionally does
not implement a full `hydrol_soil` loop, because several active-path state
values are only available inside the Fortran timestep.

## Implemented Source-Backed Kernels

All functions below live in `jax_orchidee/sechiba/hydrol.py` and include
Fortran provenance in their docstrings.

- Mineral CWRR table and coefficient path:
  `build_mineral_cwrr_tables`, `select_mineral_cwrr_bin`,
  `hydrol_soil_coef_mineral_from_selected_coefficients`, and
  `hydrol_soil_coef_mineral_from_tables`.
  Source spans: `hydrol.f90` lines 4067-4069, 4150-4159, 4172-4244, and
  8248-8280; `vertical_soil.f90` lines 185-263 and 331-363;
  `constantes_soil_var.f90` lines 250-273.
- Tridiagonal setup/RHS/solve:
  `hydrol_soil_setup_coefficients`, `hydrol_soil_rhs_main`,
  `hydrol_soil_residual_boundary_rhs`, and `hydrol_soil_tridiag_solve`.
  Source spans: `hydrol.f90` lines 5953-5991, 6971-7044, 8147-8198, and
  8480-8568.
- Drainage and conservation:
  `bottom_drainage`, `drainage_correction`,
  `hydrol_liquid_redistribution_check`.
  Source spans: `hydrol.f90` lines 6007-6052.
- Post-solve explicit formulas:
  `hydrol_layer_moisture_content`, `hydrol_total_moisture_content`,
  `hydrol_mc_to_mcl`, `hydrol_mcl_to_mc_after_solve`, and
  `hydrol_soil_clip_over_mcs2`.
  Source spans: `hydrol.f90` lines 5899-5908, 6018-6025, 6063-6075,
  6300-6328, 6473-6521, 7044-7108, and 7935-8026.

## Explicit-Input Adapter

`hydrol_soil_explicit_solve_step` composes the audited kernels for one
diagnostic solve step. It requires all upstream state explicitly:

- `mcl_before`, `mc_before`, `profil_froz`, `mcr`, `b`, `dz_mm`,
  `dt_days`, `free_drain_coef`, `flux_top`, `rootsink`, `resolv`,
  `mask_soiltile`, and `k_bottom`.
- Optional `check_tr_ns` can be supplied from trace; otherwise it is computed
  from explicit `tmci/tmcf/flux_top/drainage/rootsink` inputs.
- Optional over-saturation clipping runs only with `clip_over_mcs=True` and an
  explicit `mcs`.

The adapter returns RHS, tridiagonal output, liquid totals, drainage before and
after correction, `mc_after_mcl_update`, and optional over-saturation clipping
diagnostics. It does not compute hidden upstream state such as `flux_top`,
`rootsink`, `resolv`, freeze profile, water-table forcing, vegetation stress,
or alternate-branch triggers.

## Trace Schema And Validator

`jax_orchidee/sechiba/hydrol_trace.py` defines schema groups for the next
Fortran trace:

- `setup`
- `pre_solve`
- `solve`
- `drainage_conservation`
- `post_solve`
- `alternate_path`

`validate_columns` accepts a Mapping, iterable of column names, or explicit
CSV path and returns missing columns grouped by phase. It reads only CSV
headers and does not infer missing values.

## Restart/Reference Anchors

`jax_orchidee/sechiba/hydrol_reference.py` reads explicitly selected local
SECHIBA restart fields and normalizes axes:

- `njsc`
- `moistc`, `moistcl`
- `free_drain_coef`
- `veget`, `veget_max`, `humrel`, `vegstress`
- `wtp`, `wt_ab`, `wt_ab_tide`
- `zwt_force`, `water2infilt`, `ae_ns`, `resdist`

These are restart anchors, not timestep solve traces. They can validate local
input/state loading, but they cannot validate RHS, tridiagonal output,
drainage correction, or post-solve state for a specific timestep.

## Why There Is No Full `hydrol_soil` Loop Yet

The remaining missing values are active-path state produced inside the Fortran
timestep and cannot be reconstructed from restart/history without guessing:

- `mask_soiltile`, `resolv`, `flux_top`, `rootsink_by_layer`, and `b`.
- `rhs`, `tmat_e`, `tmat_f`, `tmat_g1`, and stored equation arrays.
- `mcl_after_tridiag`, `tmci`, `tmcf`, `check_tr_ns`, `dr_corrnum_ns`.
- `mc_after_mcl_update`, over/under saturation post-state, final `mc/mcl`.
- Alternate residual branch trigger and post-alternate state.
- Humrel/vegstress diagnostic inputs: root profiles, vegetation/tile
  aggregation, moisture thresholds, and branch flags.

Adding a full loop before these are traced would require approximating process
state or choosing case-specific branches inside generic kernels, both of which
are explicitly out of scope.

## First Action After New Trace Exists

1. Validate new CSV headers with `hydrol_trace.validate_columns`.
2. For the traced active point/tile/layers, feed `setup`, `pre_solve`, and
   `solve` columns into `hydrol_soil_explicit_solve_step`.
3. Compare JAX outputs against trace columns in this order:
   `rhs`, `mcl_after_tridiag`, `tmci`, `tmcf`, `dr_ns_before_corr`,
   `check_tr_ns`, `dr_corrnum_ns`, `dr_ns_after_corr`,
   `mc_after_mcl_update`.
4. Only after those match, wire over/under saturation and alternate-branch
   state updates behind explicit traced branch flags.
