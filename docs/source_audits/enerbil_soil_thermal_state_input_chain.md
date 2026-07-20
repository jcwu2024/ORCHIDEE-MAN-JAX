# ENERBIL Soil Thermal State Input Chain

Scope: exact source chain for `enerbil_main` inputs `soilcap`,
`soilcap_pft`, `soilflx`, and `soilflx_pft`. This audit separates
ENERBIL pre-call state, ENERBIL after-boundary trace, THERMOSOIL after-call
state, and restart start state. It does not use `after_enerbil_main` to infer
pre-call values.

## Run Order

`sechiba_initialize` calls `thermosoil_initialize` before any timestep
`enerbil_main` call. In the CWRR branch, `sechiba.f90` lines 719-730 pass
`soilcap`, `soilcap_pft`, `soilflx`, and `soilflx_pft` to
`thermosoil_initialize`.

During each `sechiba_main` timestep, `diffuco_main` runs first
(`sechiba.f90` lines 997-1005), then `enerbil_main` consumes the current
thermal state (`sechiba.f90` lines 1013-1019). CWRR `hydrol_main`,
`condveg_main`, and `thermosoil_main` run later; `thermosoil_main` receives
and updates the same thermal-state arrays at `sechiba.f90` lines 1109-1118.

Therefore:

- First-step ENERBIL pre-call state is the state left by
  `thermosoil_initialize`.
- Later-step ENERBIL pre-call state is the previous timestep's
  post-THERMOSOIL state, after `sechiba_end`/module state carries it forward.
- `after_enerbil_main` is after the ENERBIL process boundary and is not a
  same-call pre-call source.
- `after_thermosoil` state is valid as next-step ENERBIL state only when the
  target timestep is explicitly the following timestep.

## THERMOSOIL Initialization

`thermosoil.f90::thermosoil_initialize` documents that these coefficients are
needed to avoid a time shift. It reads:

- `soilcap`: lines 668-670.
- `soilcap_pft`: lines 672-674.
- `soilflx`: lines 678-680.
- `soilflx_pft`: lines 682-684.

If any coefficient is absent (`val_exp`), `calculate_coef` becomes true and
`thermosoil_coef` is called at lines 725-735. That path is exact only if all
same-case inputs to `thermosoil_coef` are supplied or ported.

## THERMOSOIL Update And Restart Write

`thermosoil_main` recomputes these coefficients near the end of the timestep
by calling `thermosoil_coef` at `thermosoil.f90` lines 998-1010. The comment
at lines 998-1000 says the coefficients are from this timestep and are used at
the next timestep by ENERBIL or by the thermal numerical scheme.

`thermosoil_finalize` writes the restart state:

- `soilcap` and `soilcap_pft`: lines 1129-1132.
- `soilflx` and `soilflx_pft`: lines 1134-1136.

This makes the local `sechiba_start.nc` variables exact first-step state for
the archived reference run, provided the file is the start restart consumed by
that run.

## Local Reference Availability

The local reference file
`reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/sechiba_start.nc`
contains all four variables:

| Field | Restart dimensions | First point / PFT14 value |
| --- | --- | --- |
| `soilcap` | `(time,y,x)` | `45865.2880547002` |
| `soilcap_pft` | `(time,z_a,y,x)` | `45865.2880547002` |
| `soilflx` | `(time,y,x)` | `-34.047520966054336` |
| `soilflx_pft` | `(time,z_a,y,x)` | `-34.047520966054336` |

`jax_orchidee/sechiba/enerbil.py::read_enerbil_soil_thermal_state_restart`
reads these fields through the existing restart reader and normalizes them to
model-point axes. Missing variables raise instead of being defaulted.

## Coverage Decision

Closed for first-step local reference parity:

- `soilcap`
- `soilcap_pft`
- `soilflx`
- `soilflx_pft`

Source: same-case local `sechiba_start.nc`, with Fortran read provenance
`thermosoil_initialize` lines 668-684 and write provenance
`thermosoil_finalize` lines 1129-1136.

Superseding runtime coverage:

- Non-restart initialization when the four fields are absent is now covered
  through the source-backed `thermosoil_coef` path when its required
  same-case inputs are supplied. Driver orchestration coverage records
  `soilcap_from_thermosoil_coef` and removes
  `thermosoil_coef_non_restart_soilcap_soilflx_waiting_for_refSOC` from the
  missing-component set.
- Same-call inference from `after_enerbil_main`. It remains after-boundary
  only.
- Next-step use of `after_thermosoil` unless the target ENERBIL call is the
  following timestep and run-order continuity is explicitly tested.
