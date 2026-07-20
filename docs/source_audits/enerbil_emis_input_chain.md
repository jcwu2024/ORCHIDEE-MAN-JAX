# ENERBIL `emis` Input Chain

Scope: exact source chain for `enerbil_main` input `emis`.

## Call Order

`sechiba.f90::sechiba_initialize` calls `condveg_initialize` at lines
690-698 and passes the module state `emis`. It then exposes
`emis_out(:)=emis(:)` at line 782.

`sechiba.f90::sechiba_main` calls `enerbil_main` at lines 1013-1019 and passes
the saved module state `emis` as an ENERBIL input. The same timestep
`condveg_main` call is later, at lines 1084-1090. Therefore same-step
`condveg_main` output cannot be used as the current ENERBIL input; it prepares
state for output and subsequent timestep use.

`enerbil.f90::enerbil_main` declares `emis` as `INTENT(in)` at line 432, passes
it to `enerbil_begin` at lines 485-489, to `enerbil_surftemp` at lines
507-510, to `enerbil_pottemp` at lines 523-526, and to `enerbil_flux` at lines
534-537.

## Initialization Source

`condveg.f90::condveg_initialize` lines 260-269 are the exact source:

- if `impaze` is true, `emis(:)=emis_scal`;
- otherwise, `emis_scal=un` and `emis(:)=emis_scal`.

`src_parameters/constantes_var.f90` lines 670-707 initialize
`impaze=.FALSE.` and `emis_scal=1.0`. `src_parameters/constantes.f90` lines
622-678 reads `IMPOSE_AZE`, and reads `CONDVEG_EMIS` only when `IMPOSE_AZE`
is true.

The local expanded run configuration records `IMPOSE_AZE = FALSE` in
`outputs/server_1961_trace_full_20260623/run/used_run.def` line 2448. No local
paper/reference run.def override for `CONDVEG_EMIS` was found. For the current
local first-step case, `condveg_initialize` therefore supplies exact
`emis=1.0`.

## JAX Coverage

`jax_orchidee/sechiba/enerbil.py::condveg_initialized_emis` implements only
the `condveg_initialize` emissivity branch. It requires an explicit
`emis_scal` when `impaze=True`, so `CONDVEG_EMIS` is not silently defaulted.

Coverage helpers may mark `emis` source-covered only when
`condveg_initialize_executed=True` and the `IMPOSE_AZE` branch is explicit.
For `IMPOSE_AZE=False`, `emis=1.0` is exact from the Fortran assignment to
`un`; for `IMPOSE_AZE=True`, same-case `CONDVEG_EMIS` or a compatible
pre-ENERBIL trace is still required.
