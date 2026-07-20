# STOMATE Phase 1C State Contract

Scope: paper-case PFT14 mangrove active branch only. This contract documents
local restart/history validation boundaries for wiring the existing Phase 1B
carbon kernels. It does not define a full STOMATE loop and does not permit using
history output to fit parameters.

## Local Reference Inventory

Reference run directory:
`reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658`

Files found:

- `stomate_start.nc`
- `stomate_restart.nc`
- `stomate_history_1961.nc` through `stomate_history_2010.nc` (`50` yearly files)

Restart state truth:

- `biomass(time,m_a,l_a,z_a,y,x) = (1,1,12,14,1,1)`
- `maint_resp(time,l_a,z_a,y,x) = (1,12,14,1,1)`
- `resp_maint(time,z_a,y,x) = (1,14,1,1)`
- `resp_growth(time,z_a,y,x) = (1,14,1,1)`
- `gpp_daily(time,z_a,y,x) = (1,14,1,1)`
- `npp_daily(time,z_a,y,x) = (1,14,1,1)`
- `PFTpresent(time,z_a,y,x) = (1,14,1,1)`
- `sla_calc(time,z_a,y,x) = (1,14,1,1)`
- `t2m_daily(time,y,x) = (1,1,1)`
- `tsoil_daily(time,z_c,y,x) = (1,11,1,1)`
- `t2m_longterm(time,y,x) = (1,1,1)`
- `moiavail_week(time,z_a,y,x) = (1,14,1,1)`
- `tsoil_month(time,z_c,y,x) = (1,11,1,1)`
- `soilhum_month(time,z_c,y,x) = (1,11,1,1)`
- `leaf_age(time,l_b,z_a,y,x) = (1,4,14,1,1)`
- `leaf_frac(time,l_b,z_a,y,x) = (1,4,14,1,1)`
- `age(time,z_a,y,x) = (1,14,1,1)`

Restart gaps for direct Phase 1B integration:

- `f_alloc` is not present.
- `bm_alloc` is not present.
- `lai` is not present.
- No before/after snapshots around `maint_respiration`, `alloc`, or `npp_calc`
  are present in restart.

History/modelout validation truth:

- Modelout fields are present in yearly history:
  `LEAF_M`, `SAP_M_AB`, `SAP_M_BE`, `HEART_M_AB`, `HEART_M_BE`, `ROOT_M`,
  `FRUIT_M`, `RESERVE_M`, `AGR_SAP_ST_M`, `AGR_HRT_ST_M`, `AGR_SAP_PN_M`,
  `AGR_HRT_PN_M`, `GPP`, `NPP`.
- Diagnostic fields are present:
  `LAI`, `AGE`, `VEGET_MAX`, `MAINT_RESP`, `GROWTH_RESP`,
  `MAINT_RESP_AGRSAPST`, `MAINT_RESP_AGRSAPPN`, `MAINT_RESP_AGRHRTST`,
  `MAINT_RESP_AGRHRTPN`, `BM_ALLOC_LEAF`, `BM_ALLOC_SAP_AB`,
  `BM_ALLOC_SAP_BE`, `BM_ALLOC_ROOT`, `BM_ALLOC_FRUIT`, `BM_ALLOC_RES`,
  `NPP_ABOVE`, `NPP_BELOW`.
- History lacks full restart/process state variables:
  `biomass`, `PFTpresent`, `f_alloc`, and full-pool `bm_alloc`.

Important boundary: yearly history pool outputs and final restart biomass are
not equal pointwise. History can validate output formulas and exposed
diagnostics, but it is not a process before/after trace.

## Source Execution Spans

Daily accumulation and maintenance respiration in `stomate.f90`:

- `stomate_main` daily forcing accumulation: lines 3200-3208.
- `stomate_main -> maint_respiration`: lines 3244-3252.
- `resp_maint_part` accumulation: lines 3254-3267.

Daily STOMATE active process call:

- `stomate_main -> StomateLpj`: lines 4297-4335.
- `StomateLpj -> phenology`: `stomate_lpj.f90` lines 1070-1082.
- `StomateLpj -> alloc`: `stomate_lpj.f90` lines 1093-1102.
- `StomateLpj -> setlai`: `stomate_lpj.f90` lines 1111-1113.
- `StomateLpj -> npp_calc`: `stomate_lpj.f90` lines 1115-1130.

Implemented Phase 1B kernel source spans:

- `maint_respiration`: `stomate_resp.f90` lines 122-170, 203-232,
  234-303, 319-376.
- Closed AGR split inside `alloc`: `stomate_alloc.f90` lines 146-152,
  187-201, 760-834.
- Closed `npp_calc` algebra: `stomate_npp.f90` lines 116-180, 230-247,
  280-295, 299-386, 447-531.

History/modelout writes:

- `stomate_lpj.f90` XIOS sends for `LAI`, `GPP`, biomass pools,
  respiration, and AGR maintenance diagnostics: lines 1675-1707.
- `stomate_npp.f90` writes `BM_ALLOC_*`, `SLA_CALC`, `NPP_ABOVE`,
  `NPP_BELOW`: lines 681-720.
- Paper modelout formula remains in
  `fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py`
  lines 50, 60-76, 115-118.

## Kernel Integration Contract

### `maintenance_respiration`

JAX status: implemented in `jax_orchidee/stomate/carbon_kernels.py`.

Required exact inputs:

- `biomass_before[npts,nvm,nparts,nelements]`: restart has final state only.
- `t2m`, `t2m_longterm`, `stempdiag/tsoil`, `rprof`: restart has some daily
  meteorology but lacks a clear process-time before/after anchor for the kernel.
- `sla_calc[npts,nvm]`: restart has it.
- `coeff_maint_zero`, `maint_resp_slope`, `ext_coeff`, `is_tree`: parameter
  sources must be loaded from parameter files, not history.

Validation available now:

- Restart `maint_resp` can be axis-normalized and used as final stored state
  inventory.
- History `MAINT_RESP*` can validate exposed diagnostics only.

Still required before process parity:

- Trace at `stomate.f90` immediately before and after lines 3248-3267:
  `biomass`, `lai`, `t2m`, `t2m_longterm`, `stempdiag`, `rprof`,
  `sla_calc`, `resp_maint_part_radia`, and accumulated `resp_maint_part`.

### `agr_allocation_split`

JAX status: closed AGR split helper implemented; full `alloc` is blocked.

Required exact inputs:

- `biomass_before_alloc` and `reserve_biomass`.
- `senescence`.
- `l_to_lsr`, `s_to_lsr`, `r_to_lsr`, `alloc_sap_above`.
- `sla_calc`, `lai_max`, `alloc_agr_st`, `alloc_agr_pn`, `ecureuil`.

Validation available now:

- History exposes `BM_ALLOC_LEAF`, `BM_ALLOC_SAP_AB`, `BM_ALLOC_SAP_BE`,
  `BM_ALLOC_ROOT`, `BM_ALLOC_FRUIT`, `BM_ALLOC_RES`, but not `f_alloc` and not
  AGR-specific `BM_ALLOC_AGR*` fields.

Still required before integration:

- Full `alloc` upstream state or trace for `l_to_lsr`, `s_to_lsr`, `r_to_lsr`,
  `alloc_sap_above`, `senescence`, `f_alloc`, and AGR pool allocation fractions.
- Source spans currently blocking full allocation are `stomate_alloc.f90`
  lines 335-759, which depend on season/stress state such as
  `moiavail_week`, `tsoil_month`, `soilhum_month`, `lai_around`,
  `when_growthinit`, `leaf_age`, and `leaf_frac`.

### `npp_closed_update`

JAX status: closed algebra implemented, stopping before leaf-age/SLA updates.

Required exact inputs:

- `biomass_before_npp[npts,nvm,nparts,nelements]`.
- `gpp_daily[npts,nvm]`.
- `f_alloc[npts,nvm,nparts]`.
- `resp_maint_part[npts,nvm,nparts]`.
- `PFTpresent[npts,nvm]`.
- `frac_growthresp[nvm]`.

Validation available now:

- Restart contains final `gpp_daily`, `npp_daily`, `resp_maint`,
  `resp_growth`, `maint_resp`, `PFTpresent`, and `biomass`.
- History contains annual `GPP`, `NPP`, `MAINT_RESP`, `GROWTH_RESP`, and
  partial `BM_ALLOC_*` diagnostics.

Still required before process parity:

- Trace immediately before and after `stomate_lpj.f90` lines 1118-1130:
  `biomass`, `gpp_daily`, `f_alloc`, `bm_alloc`, `resp_maint_part`,
  `resp_maint`, `resp_growth`, `npp_daily`, `leaf_age`, `leaf_frac`, `age`.
- Full `f_alloc` cannot be reconstructed from current restart/history without
  running or tracing `alloc`.

## Next Safe Boundary

The next safe task is not full STOMATE. It is a trace/restart bridge:

1. Add Fortran-side trace points or locate existing trace columns for PFT14 at
   the three process cuts: after `maint_respiration`, after `alloc`, after
   `npp_calc`.
2. Verify daily process-time inputs/outputs with exact arrays:
   `biomass_before`, `resp_maint_part`, `f_alloc`, `bm_alloc`,
   `biomass_after`, `gpp_daily`, `npp_daily`, `resp_maint`, `resp_growth`,
   and AGR pools before/after.
3. Only then wire the JAX kernels into a small adapter sequence for one day and
   one PFT14 point.

