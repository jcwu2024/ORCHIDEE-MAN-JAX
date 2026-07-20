# ENERBIL to HYDROL Boundary Contract Audit

Scope: minimal name-only contract for the fields `hydrol_main` consumes after
`enerbil_main` in the paper-case SECHIBA order. This is a contract step before
building a real `hydrol_main` adapter. It does not compute, tune, default, or
synthesize missing process values.

## Boundary Fields

The required ENERBIL-to-HYDROL payload is:

- `temp_sol_new`
- `transpir`
- `transpot`
- `vevapwet`
- `vevapnu`
- `vevapnu_pft`
- `vevapsno`
- `vevapflo`
- `evapot`
- `evapot_corr`
- `pgflux`
- `temp_sol_add`

The current `after_enerbil_main` bridge trace covers the first ten fields. It
does not contain `pgflux` or `temp_sol_add`. Both remain required because the
paper-case HYDROL path uses explicit snow, and HYDROL passes them into the
explicit-snow update.

## Fortran Provenance

- `src_sechiba/enerbil.f90::enerbil_main` declares the HYDROL-facing flux,
  evaporation, temperature, and snow-energy fields at lines 449-472.
- `src_sechiba/enerbil.f90::enerbil_flux` updates `pgflux` and
  `temp_sol_add` in the explicit-snow energy path at lines 1541-1588.
- `src_sechiba/enerbil.f90::enerbil_evapveg` computes `vevapsno`,
  `vevapnu`, `vevapnu_pft`, `vevapflo`, `vevapwet`, `transpir`, and
  `transpot` at lines 1756-1833.
- `src_sechiba/sechiba.f90::sechiba_main` calls `hydrol_main` after
  `enerbil_main` and passes these fields at lines 1049-1064.
- `src_sechiba/hydrol.f90::hydrol_main` declares ENERBIL-derived water and
  evaporation inputs at lines 984-1002.
- `src_sechiba/hydrol.f90::hydrol_main` declares `pgflux` at line 1010.
- `src_sechiba/hydrol.f90::hydrol_main` declares `evapot`, `evapot_penm`, and
  `vevapflo` at lines 1064-1067.
- `src_sechiba/hydrol.f90::hydrol_main` declares `temp_sol_add` at line 1090.
- `src_sechiba/hydrol.f90::hydrol_main` passes `temp_sol_new`, `pgflux`,
  `vevapsno`, and `temp_sol_add` to explicit-snow handling at lines
  1182-1196.
- `src_sechiba/hydrol.f90::hydrol_main` sends `vevapwet` and `vevapflo` into
  canopy and flood updates at lines 1232-1238.
- `src_sechiba/hydrol.f90::hydrol_main` sends `transpir`, `vevapnu`,
  `vevapnu_pft`, `evapot`, and `evapot_penm` into `hydrol_soil` at lines
  1278-1284.

## Trace Coverage

Audited trace:

- `outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_enerbil_trace.txt`
- tag: `after_enerbil_main`
- schema/audit anchor: `docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_enerbil_main`

The trace includes `transpir`, `transpot`, `vevapwet`, `vevapnu`,
`vevapnu_pft`, `vevapsno`, `vevapflo`, `evapot`, `evapot_corr`, and
`temp_sol_new`. It also includes related ENERBIL diagnostics such as `qsurf`,
`t2mdiag`, and PFT surface-temperature fields, but those are outside this
minimal HYDROL boundary contract.

The trace does not include `pgflux` or `temp_sol_add`; the validator therefore
reports them as missing for current traced payloads. A future real
`hydrol_main` adapter must receive audited values for both fields instead of
using placeholders.

## Implemented Contract

`jax_orchidee/sechiba/enerbil_bridge.py` now exposes:

- `EnerbilToHydrolBoundaryField`
- `EnerbilToHydrolBoundaryValidation`
- `enerbil_to_hydrol_boundary_field_names`
- `enerbil_to_hydrol_after_main_trace_fields`
- `enerbil_to_hydrol_missing_after_main_trace_fields`
- `validate_enerbil_to_hydrol_boundary_payload`

Validation is intentionally name-only. It reports covered and missing fields
from a payload, with `pgflux` and `temp_sol_add` called out as required
explicit-snow fields. It never derives values from neighboring fields and never
fills absent HYDROL inputs.
