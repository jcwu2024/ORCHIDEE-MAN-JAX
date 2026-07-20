# ENERBIL Surface State Input Chain

Scope: audit the exact source chain for the fields needed by
`enerbil_begin` and `enerbil_surftemp`: `swnet`, `emis`, `soilflx`,
`soilflx_pft`, `soilcap`, `soilcap_pft`, `temp_sol`, and `temp_sol_pft`.

## ENERBIL Boundary

`src_sechiba/enerbil.f90::enerbil_main` declares these as inputs or inout
state at lines 410-438 and 463-466. It passes `temp_sol`, `temp_sol_pft`,
`lwdown`, `swnet`, `pb`, and `emis` into `enerbil_begin` at lines 485-489,
then passes `emis`, `soilflx`, `soilflx_pft`, `soilcap`, `soilcap_pft`,
`lwdown`, and `swnet` into `enerbil_surftemp` at lines 507-510.

The corresponding dummy arguments are `INTENT(in)` in
`enerbil_begin` lines 741-748 and in `enerbil_surftemp` lines 948-968.
`temp_sol` and `temp_sol_pft` are `enerbil_main` `INTENT(inout)` state, but
they are consumed by `enerbil_begin` before the ENERBIL update.

## Field Sources

| Field | Current closure | Exact source chain |
| --- | --- | --- |
| `temp_sol` | covered by pre-ENERBIL trace or initialization/state swap | `enerbil_initialize` reads restart `temp_sol` at `enerbil.f90` line 241 and initializes `temp_sol_new(:)=temp_sol(:)` at lines 259-260. Later steps use `sechiba_end`, `sechiba.f90` lines 3114-3135, to assign `temp_sol(:)=temp_sol_new(:)`. `after_diffuco_main` is after `diffuco_main` and before `enerbil_main` (`sechiba.f90` lines 997-1019), so it is a valid pre-ENERBIL trace source. |
| `temp_sol_pft` | covered by pre-ENERBIL trace or initialization/state swap | `enerbil_initialize` reads restart `temp_sol_pft` at `enerbil.f90` line 253 and initializes missing values with `ENERBIL_TSURF` default 280 K at lines 254-255. It sets `temp_sol_new_pft(:,:)=temp_sol_pft(:,:)` at line 261. Later steps use `sechiba_end` lines 3133-3135. `after_diffuco_main` is a valid pre-ENERBIL trace source. |
| `emis` | source-covered for audited local first-step sources | `condveg_initialize` sets `emis(:)=emis_scal` when `impaze` is true, otherwise sets `emis_scal=un` and `emis(:)=emis_scal`; see `condveg.f90` lines 260-269. `condveg_main` sets `emis(:)=emis_scal` at lines 402-403 and is called after ENERBIL in the main timestep (`sechiba.f90` lines 1084-1090), preparing state for the next step. The current paper setup has `IMPOSE_AZE=FALSE`, so the first-step value is exactly `1.0`; `IMPOSE_AZE=TRUE` scenarios require the audited `CONDVEG_EMIS` value or a compatible pre-call trace. |
| `soilcap` | source-covered by restart or THERMOSOIL coefficient path | `thermosoil_initialize` reads restart `soilcap` at `thermosoil.f90` lines 668-670 and calls `thermosoil_coef` when restart coefficients are missing at lines 725-735. `thermosoil_coef` declares `soilcap` output at line 1423, initializes it at line 1491, and assigns snow/no-snow effective values at lines 1715-1725. Runtime `thermosoil_main` is called after ENERBIL (`sechiba.f90` lines 1109-1118), producing next-step state. |
| `soilcap_pft` | source-covered by restart or THERMOSOIL coefficient path | `thermosoil_initialize` reads restart `soilcap_pft` at lines 672-674. `thermosoil_coef` declares it at line 1424, initializes it at line 1493, and assigns PFT values at lines 1568-1570 and 1577-1579. |
| `soilflx` | source-covered by restart or THERMOSOIL coefficient path | `thermosoil_initialize` reads restart `soilflx` at lines 678-680. `thermosoil_coef` declares it at line 1426, initializes it at line 1492, and assigns snow/no-snow effective values at lines 1718-1725. |
| `soilflx_pft` | source-covered by restart or THERMOSOIL coefficient path | `thermosoil_initialize` reads restart `soilflx_pft` at lines 682-684. `thermosoil_coef` declares it at line 1427, initializes it at line 1494, and assigns PFT values at lines 1567 and 1571-1581. |
| `swnet` | source-covered only with same-step two-band albedo | `sechiba_main` receives `swnet` and `swdown` as separate `INTENT(in)` fields at `sechiba.f90` lines 889-891 and passes `swnet` to DIFFUCO and ENERBIL at lines 997-1018. In the offline driver, `orchideedriver.f90` computes `swnet(:) = (1.-(albedo(:,1)+albedo(:,2))/2.)*swdown(:)` at lines 598 and 659. Therefore `swdown` closes `swnet` only with same-step two-band `albedo`; `swdown` alone is not a valid substitute. |

## Trace Boundary Rule

`after_enerbil_main` is after `CALL enerbil_main` in `sechiba_main` lines
1013-1019. It can verify ENERBIL outputs and diagnostics, but it cannot be
used to infer same-call pre-ENERBIL inputs unless a field is independently
proved unchanged and has an `INTENT(in)` pre-call source. This audit does not
use `after_enerbil_main` to close `soilcap`, `soilcap_pft`, `soilflx`,
`soilflx_pft`, `temp_sol`, or `temp_sol_pft`.

`after_diffuco_main` is different: `diffuco_main` is called at
`sechiba.f90` lines 997-1005 and `enerbil_main` at lines 1013-1019. Fields
traced at `after_diffuco_main` are therefore valid pre-ENERBIL state when the
field is present in that trace.

## JAX Helpers

`jax_orchidee/sechiba/enerbil.py` now includes:

- `driver_swnet_from_swdown_albedo`, implementing the exact offline-driver
  `swnet` formula from `orchideedriver.f90` lines 598 and 659.
- `enerbil_surface_state_input_coverage`, which classifies the eight audited
  fields as trace-covered, source-kernel-covered, or missing. It explicitly
  records `after_enerbil_main` values as after-boundary only and never uses
  them to close pre-call coverage.

## Current Status

Closed by current local coverage:

- `temp_sol`
- `temp_sol_pft`
- `swnet`, when `swdown` is paired with same-step two-band driver albedo.
- `emis`, for the audited `IMPOSE_AZE=FALSE` initialization branch.
- `soilcap`, `soilcap_pft`, `soilflx`, `soilflx_pft`, from same-case restart
  fields or the source-backed `thermosoil_coef` initialization/update path.

`temp_sol` and `temp_sol_pft` are closed when supplied by
`after_diffuco_main`, a pre-ENERBIL trace.

Still explicit, not guessed:

- `swdown` alone is not accepted as `swnet`.
- `after_enerbil_main` is never used as a pre-call source for the same call.
- `IMPOSE_AZE=TRUE` needs the corresponding `CONDVEG_EMIS` source value.
- Any THERMOSOIL path must be coherent with the same runtime truth: restart
  fields for a restart case, or `thermosoil_coef` inputs for non-restart
  initialization/update.
