# ENERBIL Soil Thermal State Input Chain

Scope: exact source chain for `enerbil_main` inputs `soilcap`,
`soilcap_pft`, `soilflx`, and `soilflx_pft`. This audit separates
ENERBIL pre-call state, ENERBIL after-boundary trace, THERMOSOIL after-call
state, and restart start state. It does not use `after_enerbil_main` to infer
pre-call values.

## Current component evidence (2026-11-17; accepted in the declared domain)

The [thermal/snow/coupling batch](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/README.md)
adds eight original-source full-component references on main
`82d622722feea3cfe5034d8b15f98cda78e04872`. Actual production owners select
the complete explicit or no-explicit eager/JIT delegate. A second bare-ground
operand omission was found at both final coefficient callers: source
`thermosoil_main:1002-1009` passes `veget_max_bg`. Only those two operands were
repaired; accepted C50 energy calls remain unchanged. Reusing the original
references after repair passes 552 fields / 112332 scalars; the 52 original
failed coefficient fields remain saved. This is accepted correction C51,
distinct from C50's energy consumer. Code/test commit:
`9a856c6cf859437d2436f4f3748d4c9e770838a1`; see the
[controller acceptance](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-acceptance.json).

The same commit accepts C52 at the separate driver boundary: the actual
ENERBIL payload carries its executed timestep, and the shared driver writer
converts evaporation amount to rate once, as in `intersurf:743-749`.
ENERBIL/SECHIBA retain per-step amounts; the driver codec retains the rate.
The external PBL coefficient producer remains unidentified.

The root's 15 bounded joint checks exercise the actual shared writer,
current getdiff `pcapa_en`, energy `temp_sol_beg`, profile/diagnostics and
end coefficients. Sparse explicit and bucket-thin cases each run a second
thermal component and consume the new capacity/begin memory. The cold case
verifies first-call carry and observes the actual next delegate's arguments;
its second synthetic input hits the existing cross-PFT freeze guard and is
not a successful cold trajectory. Actual snow output also reaches the thermal
boundary, without claiming an independent coupled numerical oracle.

No-explicit initialization, compiled illegal-freeze abort semantics, arbitrary
flags/peat/masks, actual restart reader and historical producer remain outside
this evidence. Whole-model numerical acceptance remains false. The accepted
initialization evidence below retains its original identity and scope.

## Accepted initialization and first-consumer evidence (2026-11-16)

The [local source-extracted oracle](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/README.md)
uses main `551cc8cdbc4f16fa37ed2c89f260c36b57c596a8` and supplied-array
restart seams. It compiles original THERMOSOIL procedures and initialization
statements, plus HYDROL moisture/temperature statements; it does not compile
the IOIPSL/NetCDF reader or replay the historical reference producer.

The nine-member coefficient group is `soilcap`, `soilcap_pft`, `soilflx`,
`soilflx_pft`, `cgrnd`, `dgrnd`, `cgrnd_snow`, `dgrnd_snow`, and `lambda_snow`.
Source lines 667-714 set `calculate_coef` when any member is wholly `val_exp`.
The unconditional final `var_init` at 719-723 reconstructs `pcapa`,
`pcapa_en`, `pkappa`, `stempdiag`, and the humidity/profile diagnostics.
For a complete restored group its thin-snow overlay remains. For a missing
group the following `coef` calls `getdiff` again, overwriting the transient
thin-snow material overlay; its thin-snow call at 1499-1504 is commented out.
The [phase contract](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/phase-contract.md)
records the separate cold, restored, missing-group, and continued phases.

`pcapa`, `pcapa_en`, and `pkappa` are material fields used inside THERMOSOIL;
they are not the four direct ENERBIL coefficient inputs listed above.
Initial `temp_sol_beg` comes from the pre-ENERBIL surface temperature
(570-573). The first THERMOSOIL energy consumer combines that begin memory
with post-ENERBIL temperature, initial `soilcap`, and initial `pcapa_en`.
It then updates begin memory to the current temperature/profile (2458-2462).
The actual carry writer subsequently stores current `getdiff.pcapa_en`,
profile `stempdiag`, and end-of-component coefficients for the next call.

Seventeen standard same-input cases passed 871 field / 159450 scalar
comparisons. An additional sparse-vegetation original Fortran case exposed
two JAX energy callers passing raw `veget_max`; source `thermosoil_main:913`
passes `veget_max_bg`. Accepted correction C50 changes those two operands only,
in code commit `6a5d4a7f2092f32ef9de064e61ee3224c5b48b01`.
The [verification](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/verification-results.json)
passes 288 energy/begin-memory field comparisons over 18 saved cases,
both actual caller expressions, eager and narrow energy JIT. One real
isolated explicit thermal owner and carry writer also passed profile,
diagnostic and energy checks. Fresh end-of-component coefficients have
writeback evidence, not a new independent numerical oracle claim.

This evidence is limited to the declared default USDA, explicit-snow,
source-enabled refSOC/freezing, nonpeat domain, whole-array missing markers,
and the tested dimensions. It does not certify full no-explicit-snow parity,
the complete compiled thermal path, arbitrary partial markers, a whole model
timestep, or historical producer identity. The legacy initializer result
does not expose all three material arrays; production cold closure and C34
restart helper are verified separately. Formal correction/production-commit
counts are 50/29; 93/126/181 remain unchanged and `numerical_acceptance=false`.
The [controller acceptance](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-acceptance.json)
reuses the owner's numerical checks and records a separate CLI-only guard
against overwriting existing evidence. Neither the reader nor the complete
compiled thermal component acquired new numerical certification.

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
