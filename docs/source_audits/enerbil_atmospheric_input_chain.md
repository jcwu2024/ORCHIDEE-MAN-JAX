# ENERBIL Atmospheric Input Chain Audit

Scope: source audit for the ENERBIL pre-call inputs `rau`, `epot_air`,
`petAcoef`, `petBcoef`, `peqAcoef`, `peqBcoef`, and `zlev` along
`sechiba_main -> enerbil_main -> enerbil_surftemp`. This audit does not use
approximations or default values to hide missing Fortran processes.

## ENERBIL Boundary

- `src_sechiba/sechiba.f90::sechiba_main` calls `sechiba_var_init` at line 982
  before DIFFUCO/ENERBIL, then calls `diffuco_main` at lines 997-1005 and
  `enerbil_main` at lines 1013-1019.
- `src_sechiba/enerbil.f90::enerbil_main` declares `zlev`, `epot_air`,
  `petAcoef`, `petBcoef`, `peqAcoef`, `peqBcoef`, and `rau` as inputs at
  lines 410-423.
- `src_sechiba/enerbil.f90::enerbil_main` passes the same values to
  `enerbil_surftemp` at lines 507-510.
- `src_sechiba/enerbil.f90::enerbil_surftemp` declares the fields at
  lines 927-950. In the audited source body, `rau`, `petAcoef`, `petBcoef`,
  `peqAcoef`, and `peqBcoef` are consumed in lines 999-1239. `zlev` is an
  interface input but is not consumed by the local `enerbil_surftemp` solve.

## Closed From Source

- `rau`: closed by `sechiba_var_init`.
  `src_sechiba/sechiba.f90::sechiba_var_init` lines 3069-3092 compute
  `rau = pa_par_hpa * pb / (cte_molr * temp_air)`. Constants are in
  `src_parameters/constantes_var.f90` lines 379 and 394. The JAX helper is
  `sechiba_air_density_from_pb_temp_air`.
- Non-WATCHOUT, non-relaxation driver coupling bundle: path-specific closure
  only after this driver branch is audited. `src_driver/dim2_driver.f90`
  lines 914-920 compute `eair_obs = cp_air*tair_obs + cte_grav*zlev_vec`;
  lines 995-1003 set `petAcoef=0`, `peqAcoef=0`, `petBcoef=eair_obs`, and
  `peqBcoef=qair_obs`. The JAX helper is
  `dim2_driver_non_watchout_energy_coupling_inputs`.

## Read From Forcing Or Trace

- `zlev`: driver forcing traces expose `height_lev1`, which maps to `zlev`.
  `src_driver/readdim2.f90` lines 1782-1805 and 1833-1839 show first-level
  height construction/reading. The current `intersurf_main` trace does not
  include `zlev`; the `driver_forcing:first_forcing` trace does.
- WATCHOUT `epot_air`/PET/PEQ: `src_driver/readdim2.f90` lines 1833-1857 read
  `levels`, `SWnet`, `Eair`, `petAcoef`, `peqAcoef`, `petBcoef`, and
  `peqBcoef` from forcing. These are exact only when the forcing row or an
  audited trace actually carries those columns.

## Still Missing By Default

For the current local `intersurf_main` trace alone, `epot_air`, `petAcoef`,
`petBcoef`, `peqAcoef`, and `peqBcoef` remain missing. They are not derived
from `temp_air`, `qair`, or `pb` in the ENERBIL helper unless the audited
non-WATCHOUT, non-relaxation driver branch is explicitly enabled.

`zlev` remains a pre-call interface field, but this source version of
`enerbil_surftemp` does not use it in the local solve. Coverage therefore
records it as an interface input and documents that the local solve does not
consume it.
