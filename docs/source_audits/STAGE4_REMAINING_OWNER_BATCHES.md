# Stage 4 Remaining Owner Batches

Initial checkpoint: 2026-07-15, `200/260` passed. Batch A and B completed at
`218/260`; 42 owners remain. This plan is an execution order, not a relaxation
of the evidence policy.

## Shared Rules

- Every transition owner uses extracted original Fortran procedure bytes,
  branch-complete deterministic micro-cases, float64 `rtol <= 1e-12`, and
  exact discrete-state comparison.
- Every state IO owner verifies cold-start, restart, shape, missing/default,
  and SAVE/write-back behavior.  It is not certified by a scientific kernel
  comparison alone.
- No owner may be marked passed before its comparison asset, branch coverage,
  and owner-region evidence all pass the aggregate audit.
- A family may share a dependency module and harness, but each owner retains
  its own arm coverage.  Do not create a second runner or a special tolerance.
- Run the family Oracle and its focused model tests after each batch.  Run the
  aggregate owner audit after each completed batch, not after each tiny edit.

## Batch A: Parameter And Static Initialization (7 owners)

Status: complete. All seven owner records pass. The batch also corrected two
source-reachability errors in the static ledger: the hard-disabled
`vertical_soil_init` refinement block and the impossible
`control_initialize` `ok_sechiba=.FALSE.` edge.

Owners: `activate_sub_models`, `config_sechiba_parameters`,
`config_stomate_parameters`, `veget_config`, `config_soil_parameters`,
`control_initialize`, `vertical_soil_init`.

Purpose: establish the immutable configuration, PFT table, and vertical-soil
state consumed by all later families.

Attention:
- Compare all parameter/SAVE writes and selected logical switches, not only
  returned scalar values.
- Test PFT14 plus a contrasting non-PFT14 column where the source indexes a
  PFT table; retain PFT14 as the acceptance target.
- Exercise default, overridden, and invalid/edge parameter paths that remain
  reachable under permitted PFT14 configuration changes.
- This batch supplies real parameter modules to later harnesses; do not encode
  their scientific constants in individual downstream stubs.

## Batch B: Driver, Forcing Geometry, And Interpolation (11 owners)

Status: complete. The batch closed four `dim2_driver` regions, `forcing_info`,
four aggregation owners, and both rank-2 interpolation owners. It also
reclassified three structurally impossible interpolation arms: the subsumed
`mini < 0.01` ELSE IF, the paired-index diagnostic, and two dateline ELSE IF
false arms.

Owners: four `dim2_driver:driver` regions, `readdim2:forcing_info`,
`aggregate_2d`, `aggregate_2d_p`, `aggregate_vec`, `aggregate_vec_p`,
`interpweight_2d`, `interpweight_2dcont`.

Purpose: certify the forcing-to-landpoint transformation for changed forcing
resolution, masks, coordinate order, and interpolation geometry.

Attention:
- Keep singleton, multi-cell, periodic/dateline, reversed-latitude, masked,
  missing/zero-weight, and boundary-cell cases distinct.
- Compare weights, indexes, masks, dimensions, and normalization state as
  first-class outputs.  A matching aggregate alone is insufficient.
- Extract real interpolation and aggregation callees.  Stubs may provide only
  grid metadata and IO boundaries.
- `dim2_driver` is orchestration-heavy: isolate the minimal real call segment
  required for each owner rather than simulating its science in Python.

## Batch C: SECHIBA Surface And Lifecycle (20 owners)

Owners: `condveg_background_soilalb`, `condveg_initialize`,
`condveg_soilalb`; `diffuco_initialize`, both `diffuco_main` regions,
`diffuco_trans`, `diffuco_finalize`; `enerbil_fusion`, `enerbil_initialize`;
`hydro_subgrid_main`; `intersurf_initialize_2d`, `intersurf_main_2d`,
`intsurf_time`; `sechiba_init` (two regions), `sechiba_initialize`,
`sechiba_main` (two regions), `sechiba_finalize`.

Purpose: close SECHIBA construction, half-hour transition routing, surface
state hand-off, and finalize lifecycle.

Attention:
- Split into two families if necessary: surface kernels and lifecycle/call
  routing.  They can share only genuine Fortran dependencies.
- Test cold start versus initialized state, first versus later call,
  active/inactive land masks, and PFT14/bare-soil routing.
- Compare every `INTENT(inout)` water/energy/vegetation state, not just fluxes.
- `sechiba_main`/`intersurf_main_2d` require a narrow real call graph and
  explicit state packet; do not turn the harness into a new model driver.

## Batch D: Slow Process And Thermal Initialization (6 owners)

Owners: `slowproc_init`, `slowproc_soilt`, `read_refsocfile`,
`thermosoil_initialize`, `thermosoil_var_init`, `stics_init`.

Purpose: verify daily/slow boundary initialization and thermal/soil profile
state before it enters the coupled SECHIBA-STOMATE path.

Attention:
- Separate cold-start, restart-like already-initialized, and daily-boundary
  cases; SAVE state and allocation/default paths are part of the contract.
- Include snow/no-snow, shallow/deep active-layer, and masked multi-point
  profiles where source arms permit them for PFT14.
- File readers are tested with minimal deterministic real-format inputs; do
  not replace source parsing/default behavior with hand-built Python arrays.

## Batch E: STOMATE Daily Carbon And Active-Layer Processes (14 owners)

Owners: three `stomate_main` regions, `stomate_data:data`, `readstart`,
`writerestart`, `control_moist_func_moyano`, `control_temp_func`, `stomatelpj`,
`microactem`, `snowlevels`, `prescribe`, `altcalc_doc`, `soilcarbon`.

Purpose: close daily carbon routing, active-layer, litter/soil carbon, and the
STOMATE restart boundary.

Attention:
- Keep daily accumulator, carbon pool transition, and restart serialization
  as separately evidenced contracts, even if one harness shares setup.
- Test PFT14 and bare soil; zero/nonzero biomass, carbon pools, root profile,
  moisture/temperature threshold sides, first/later call, and active masks.
- `readstart` has 362 arms and is the largest remaining state-IO owner.
  Partition cases by field group but require one final evidence record with
  all source arms and all state fields covered.
- Require exact logical/integer/mask/restart layout equality before float
  comparisons.  No long time-series run substitutes for this local contract.

## Batch F: Output Acceptance And Final State/IO Contracts (2 owners)

Owners: `ioipslctrl_history`, `xios_orchidee_init`.

Purpose: certify production output setup and accepted modelout contract after
the scientific and lifecycle owners are closed.

Attention:
- Compare variable identity, dimensions, frequency, enabled/disabled routing,
  missing-value behavior, and stable output ordering; no scientific formula is
  being tested here.
- XIOS/IO stubs must only replace external transport.  The real Fortran owner
  must decide names, shape, and control flow.
- This is the final Stage 4 batch because it validates the interface produced
  by all prior state and science owners.

## Batch Gates

1. A through E may proceed only after their prerequisite batch has a clean
   aggregate audit.  Independent families inside a batch can be parallelized
   after their shared harness contract is written once.
2. At every batch boundary run its Oracle(s), focused JAX tests, evidence tests,
   Ruff, `py_compile`, and `audit_pft14_owner_region_evidence.py`.
3. Stage 4 exits only at `260/260`, with no invalid or duplicate evidence.
   Then Stage 5 is production wiring/integration and Stage 6 is representative
   multi-landpoint acceptance; neither is a substitute for these owner gates.
