# ENERBIL Albedo -> Swnet Input Chain

Scope: first-step source audit for the two-band `albedo(:,1:2)` used by the
offline driver to compute ENERBIL pre-call `swnet`. This audit covers local
source/init/trace/reference only; no server inspection is used.

## Fortran Call Order

`sechiba_main` receives `swnet` and `swdown` as separate `INTENT(in)` fields
in `src_sechiba/sechiba.f90` lines 889-891. It passes `swnet` to
`diffuco_main` and then to `enerbil_main` at lines 997-1019. The same-step
`condveg_main` call that updates `albedo` is later, at lines 1084-1090.
Therefore ENERBIL consumes the driver/intersurf pre-call `swnet`; it does not
consume albedo freshly produced by same-step `condveg_main`.

For initialization, `sechiba_initialize` calls `condveg_initialize` at
`sechiba.f90` lines 690-697. `condveg_initialize` calls `condveg_albedo` at
`condveg.f90` lines 302-305. `condveg_albedo` itself is the final two-band
surface albedo calculation at lines 612-839.

## Driver Sources

Two local offline driver paths matter:

- `src_driver/orchideedriver.f90` sets a temporary `albedo(:,:)=0.13` at line
  512, computes provisional `swnet` at line 598, calls `sechiba_initialize`
  at lines 644-654, then recomputes `swnet` from the returned two-band
  `albedo` at line 659 before the first `sechiba_main` call at lines 685-697.
- `src_driver/dim2_driver.f90` also initializes `albedo(:,:,:)=0.13` at line
  794 and computes a provisional non-WATCHOUT `for_swnet` at line 919. For the
  restart path used by the local reference case, it calls
  `intersurf_initialize_2d` at lines 1111-1125, reads driver restart
  `albedo_vis` and `albedo_nir` at lines 1139-1152, and recomputes
  `for_swnet(i,j) = (1.-(albedo(i,j,1)+albedo(i,j,2))/2.)*swdown(i,j)` at
  line 1163 before `intersurf_main_2d` at lines 1293-1307. It writes the two
  albedo bands back to driver restart at lines 1421-1422.

`src_driver/readdim2.f90` supplies/interpolates `swdown` from forcing, for
example lines 1347-1363 and 1602-1616. It does not supply final two-band
surface `albedo`.

## Local Reference And Trace Status

The current local trace packages expose `swdown` in `driver_forcing` and
`intersurf_main`; they do not expose `swnet`, `albedo_vis`, or `albedo_nir`.
Thus trace `swdown` alone is not enough.

The local reference run uses driver restart inputs:

- `reference/.../run.def` lines 266-268 set `RESTART_FILEIN=driver_start.nc`
  and `SECHIBA_restart_in=sechiba_start.nc`.
- `driver_start.nc` contains `albedo_vis=0.04526853859061869` and
  `albedo_nir=0.22753724443710746` for the single local point.
- `sechiba_start.nc` contains `soilalbedo_bg` but not final
  `albedo_vis/albedo_nir`; `soilalbedo_bg` is a background soil component, not
  the driver two-band albedo consumed by the `swnet` formula.

Therefore the current local first-step `swnet` is exactly source-closed only
when using `swdown` from the compatible driver/intersurf trace plus two-band
`albedo` from `driver_start.nc`. It is not closed by `swdown` alone, by a
constant albedo, by same-step post-ENERBIL `condveg_main`, or by history output
field definitions.

## JAX Coverage

Added local helpers in `jax_orchidee/sechiba/enerbil.py`:

- `read_driver_albedo_restart(path)`: reads exact `albedo_vis/albedo_nir` from
  driver restart and exposes an `(npts,2)` albedo payload.
- `driver_swnet_from_swdown_albedo(swdown, albedo)`: existing exact formula
  helper; still requires supplied two-band albedo.
- `enerbil_swnet_input_chain_coverage(...)`: audit-only coverage detail that
  marks `swnet` covered by source kernel only when exact `swdown` and exact
  two-band driver albedo are both available.
- `enerbil_first_step_input_coverage(...)` and
  `enerbil_surface_state_input_coverage(...)` now accept
  `driver_albedo_payload` and close `swnet` only under the same audited
  preconditions.

Tests in `tests/unit/test_enerbil.py` prove both sides: `swdown` alone leaves
`swnet` missing, while the current local `driver_start.nc` albedo plus
driver/intersurf `swdown` closes the formula chain.
