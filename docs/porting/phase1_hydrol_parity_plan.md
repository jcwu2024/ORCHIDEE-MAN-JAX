# Phase 1 HYDROL Parity Plan

Goal: implement the first source-driven, trace-validated HYDROL slice without
turning the code into a PFT14-only script.

## Scope

Implement only the active trace path first:

- mineral CWRR table construction
- mineral `hydrol_soil_coef`
- bottom drainage algebra
- drainage correction algebra

Reference truth:

- `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`
- `traces/hydrol_1961_v9`

## Non-Scope

- No external JAX implementation.
- No heuristic approximations.
- No calibration tuning.
- No full SECHIBA loop yet.
- No STOMATE implementation yet.
- No server work.

## Implementation Shape

Recommended modules:

```text
jax_orchidee/
  provenance.py
  sechiba/
    hydrol.py
    hydrol_tables.py
tests/
  parity/
    test_hydrol_1961_v9.py
```

Process code should be generic over axes:

- `npts`
- `nstm`
- `nslm`
- later `nvm`

Reference tests may select:

- soil tile 4 as zero-based index 3
- bottom hydrology layer 11 as zero-based index 10
- PFT14 as zero-based index 13, when STOMATE/modelout begins

## Step 1: CWRR Table Builder

Source:

- `hydrol.f90` lines 4172-4244

Outputs:

- `mc_lin`
- `k_lin`
- `a_lin`
- `b_lin`
- `d_lin`

Test anchors should come from the trace-derived coefficients and first-row
`k_result` after `hydrol_soil_coef`.

## Step 2: `hydrol_soil_coef` Mineral Kernel

Source:

- `hydrol.f90` lines 8248-8280

Active branch:

- `ok_freeze_cwrr = True`
- `peat_hydro = False`
- `branch_peat = False`

Inputs:

- `mc`
- `profil_froz`
- `kfact_root`
- `a_lin`
- `b_lin`
- `d_lin`
- `k_lin`

Outputs:

- `a`
- `b`
- `d`
- `k`
- diagnostics: `mc_used`, `bin_i`, `k_eval`, `k_floor`

Trace anchors:

- first row `k_result = 6.0648789143219517`
- all rows satisfy `k_result = max(k_floor_raw, a_raw * mc_used + b_raw)`

## Step 3: Bottom Drainage

Source:

- `hydrol.f90` lines 6007-6016

Formula:

```text
dr_ns = mask_soiltile * k_bottom * free_drain_coef * dt_days
```

Trace anchor:

- first row `dr_ns_before_corr_at_6012 = 0.12635164404837398`

## Step 4: Drainage Correction

Source:

- `hydrol.f90` lines 6042-6047

Required parity:

- `dr_after = dr_before + dr_corrnum`
- correction sign follows `check_tr_ns`
- drainage correction cannot reduce drainage below zero

Trace anchor:

- first row `dr_after_corr_6047 = 0.12560049299877521`

## Completion Criteria

- Tests run with `conda run -n ORCJAX pytest ...`.
- No generated files outside `outputs/`.
- Each implemented function includes Fortran provenance.
- The test uses trace CSVs directly as validation truth.
- No PFT14-specific constants appear inside generic HYDROL process kernels.
