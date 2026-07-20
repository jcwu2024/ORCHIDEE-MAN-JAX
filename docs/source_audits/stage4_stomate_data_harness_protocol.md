# Stage4 stomate_data Local Harness Protocol Audit

Scope: `scripts/dev/oracle_stage4_stomate_data.py` and its generated
`outputs/reference_mode/micro_oracles/stage4_stomate_data` package. This is a
local transfer harness, not a server execution or a closure claim.

`fortran_run_scripts/paper_250919/Job0_bio` defines `remplace` at lines 31-37:
it deletes every `run.def` line containing the key and appends `key=value`.
The active paper protocol first copies `run.def.vn`, then applies the active
tide/peat branch. Its literal PFT14-relevant writes include `NVM 14`,
`SECHIBA_VEGMAX__00014 1.0`, `PREF_SOIL_VEG__00014 4`,
`PFT_TO_MTC__00014 2`, `IS_PEAT__00014 y`, `NSTM 6`, and `tides y` (lines
62-77, 159-163, and 182). `run.def.vn` supplies the concrete `__00014`
parameter fields, including `SLA__00014=0.0153` and its other listed PFT14
values.

The generator reproduces that literal source-ordered transform in isolation.
It does not resolve path/year shell variables or invent their values. The
Fortran template checks the configured `nvm` is exactly 14 and writes only
PFT14 records. It emits schema 3 after `getin_name -> Init_orchidee_para ->
control_initialize -> pft_parameters_main` and before `stomate_data::data`.
The capture boundary is the complete PFT14 call signature of
`jax_orchidee.stomate.source_helpers.stomate_data_owner`: its four mutable
destination values (including all 12 carbon pools), all PFT14 source vectors,
the two three-coefficient PFT14 matrices, `nleafages`, `ok_dgvm`,
`use_age_class`, and every `StomateDataConstants` member. It also records the
Fortran `pi` used by the tree formula, which the comparator checks against the
JAX owner's `np.pi`.

`compare_capture.py <capture.log>` parses the supplied capture itself, calls
the owner with captured values, and compares its PFT14 outputs only against
the log's `STAGE4_DATA` records. It has no numeric expected-value fixture and
fails closed for duplicate, absent, pre-schema-3, or non-PFT14 records. The
stored successful server PFT14 log is retained as real-run evidence, but is
deliberately rejected for strict comparison because it predates schema 3 and
does not contain the complete owner input boundary. This does not claim
closure.

The server-side profile uses the original linked `stomate_data.f90` object.
`fortran_server_original_branch_coverage.txt` is generated with Intel
`codecov -spi ../../build/pgopti.spi -dpi stomate_data_coverage.dpi`, explicitly
selecting the source object's static profile rather than the driver-local
`pgopti.spi`. It shows the original `stomate_data_mp_data_` function. This is
execution provenance for the paper PFT14 case only; it is not branch-complete
owner evidence and therefore does not close the owner.

Fortran process provenance: `fortran_source/ORCHIDEE/src_stomate/stomate_data.f90`,
subroutine `data`, lines 261-588 for the materialized PFT fields and lines
129-725 for the procedure boundary. Constant provenance is
`fortran_source/ORCHIDEE/src_parameters/constantes_var.f90`, lines 1087-1180,
loaded by `fortran_source/ORCHIDEE/src_parameters/constantes.f90` through the
`getin_p` calls at lines 1246-1566. PFT-vector provenance is
`fortran_source/ORCHIDEE/src_parameters/pft_parameters*.f90`, materialized by
`pft_parameters_main` before this capture.
