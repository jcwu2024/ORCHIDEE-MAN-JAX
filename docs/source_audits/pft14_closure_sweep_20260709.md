# PFT14 Closure Sweep 2026-07-09

Purpose: classify remaining "not closed", `requires_trace`, and unsupported
references after the current source-backed PFT14 branch sweep. This document
separates model semantic gaps from stale bridge-trace wording, intentionally
deferred non-PFT14 branches, and production asset/performance gates.

## Classification

| Area | Old gap phrase or source | Current classification | Evidence | Remaining action |
| --- | --- | --- | --- | --- |
| STOMATE entry payload | Bridge trace missed static, hydrol, thermosoil, erosion, restart, and routing fields | Stale bridge-trace gap, not a production process gap | `StomateMainPayloadAssembly.blocking_missing_inputs()` ignores IO handles and `tests/unit/test_stomate_entry.py::test_fully_sourced_entry_payload_has_no_blocking_process_gaps` leaves no ecological process gaps | Keep IO handles outside ecological state; do not reopen old bridge trace gaps as model gaps |
| ENERBIL `swnet` | `swdown` alone did not close `swnet` | Source-covered only with same-step two-band albedo | `driver_swnet_from_swdown_albedo`; `test_enerbil_first_step_coverage_closes_swnet_only_with_driver_albedo_chain`; `orchideedriver.f90` lines 598 and 659 | Continue rejecting `swdown`-only substitution |
| ENERBIL `emis` | No pre-call `emis` trace | Source-covered for audited `IMPOSE_AZE=FALSE`; conditional for `IMPOSE_AZE=TRUE` | `condveg.f90::condveg_initialize` lines 260-269; `test_enerbil_surface_state_coverage_closes_emis_from_audited_condveg_initialize` | Require `CONDVEG_EMIS` or trace when `IMPOSE_AZE=TRUE` |
| ENERBIL `soilcap*`/`soilflx*` | Early audit closed restart only and left non-restart open | Source-covered by coherent restart fields or `thermosoil_coef` path | `thermosoil_initialize` lines 668-735; `thermosoil_coef` lines 1423-1725; driver orchestration coverage contains `soilcap_from_thermosoil_coef` and removes the old missing component | Keep restart/cold-start truths separate; never use `after_enerbil_main` as same-call pre-state |
| ENERBIL module parity | Early cold-start bridge lacked all pre-call fields | Current active module boundary is closed; internal work-array traces remain optional diagnostics | `test_active_enerbil_server_bridge_assembly_matches_fortran_active_trace_outputs` covers active pre/after trace outputs including surface temperature, evaporation, transpiration, `pgflux`, `temp_sol_add`, and `t2mdiag` | Add kernel-level trace only if a future mismatch needs localization |
| DIFFUCO module parity | Early audit still listed ENERBIL/HYDROL boundaries as DIFFUCO-open work | PFT14 active DIFFUCO beta/resistance/GPP boundary is closed; downstream boundaries live in their own ledgers | `validate_diffuco_module_closure`; `test_server_1961_diffuco_pft14_active_main_beta_closure_matches_after_main_trace` | Keep non-PFT14 multi-PFT wiring out of current PFT14 closure claim |
| HYDROL full-solve trace contract | Full trace slice lacks coefficient/solver internals | Diagnostic trace limitation, not current production HYDROL export gap | `hydrol_full_solve_trace_contract.md`; driver orchestration tests remove HYDROL same-step and STOMATE export gaps | Keep trace contract as debugger; use source micro-cases for untriggered branch coverage |
| Routing | `routing_flow` and `routing_initialize` still need complete production maps | Asset/production-wiring gate for `nbp_glo>1`, not 669 independent one-cell paper blocker | `routing_pft14_branch_ledger.md`; local `routing.nc` lacks `gravel` and `drainage_area`; no-routing branch applies when `nbp_glo=1` | Do not spend near-term effort here unless target workflow enables multi-landpoint routing |
| Driver static thermal/albedo/SOC inputs | `READ_REFTEMP`, `use_refSOC`, `ALB_BG_MODIS`, and `use_refSOC_hydrol` could be toggled by run configuration | Source-covered PFT14-conditional static input branches | `read_paper_thermosoil_reftemp_static_field`, `read_paper_thermosoil_refsoc_static_field`, `read_paper_condveg_background_soilalbedo`, `read_paper_hydrol_refsoc_1d_static_field`, and focused static-helper tests cover explicit geometry overlaps and source fallbacks | Keep arbitrary file/regridding ownership at the explicit driver/static boundary; do not fabricate geometry |
| HYDROL `use_refSOC_hydrol` thresholds | HYDROL may alter hydraulic thresholds from 1D SOC when the switch is enabled | Source-covered PFT14-conditional branch | `hydrol_refsoc_1d_thresholds` follows `hydrol_var_init` lines 4071-4088 for `mcs/mcw/mcf`; cold-start and later-day HYDROL templates carry `refSOC_1d` only when enabled | Current paper `used_run.def` has `use_refSOC_hydrol=FALSE`; this branch remains local micro-case coverage until a target run enables it |
| OK_PC/deep carbon full driver | Daily full-driver OK_PC trace unreachable | Source-local PFT14-conditional coverage; full-driver path is blocked by Fortran guards | `control.f90` lines 271-278 force `OK_PC=.FALSE.` with `OK_LEAK=y`; `stomate.f90` lines 3093-3127 abort with `OK_LEAK=n`; JAX switch guard rejects that unsupported combination | Keep source-local micro-cases unless Fortran truth changes |
| Crop/STICS/grassland grazing/fire | `NotImplementedError` guards remain | Non-PFT14 natural triggers or out-of-target management branches | `stomate_restart_state_ledger.md` and branch matrices classify them inactive for paper PFT14 | Keep guarded; do not let them displace PFT14 peat/DOC/water/restart work |
| Paper mosaic assets | Local workspace sees only `case_001_071` | Asset scope fact, not model semantic failure | `pft14_completion_execution_plan.md` closure checkpoint; paper product is 669 independent one-cell cases | Validate additional cases when assets are present, after source branch closure and performance gates |

## Sweep Result

No new model semantic blocker was found in this pass. The actionable cleanup
was documentation/state classification: old bridge-trace gaps should stay
available as diagnostic history, but the current production closure ledger
must not count them as missing ecological processes.

The remaining real work is unchanged:

- continue source-driven PFT14 branch closure for branches that can activate
  under another PFT14 land point, forcing product, hydrological/tide/salinity
  state, or continuous parameter set;
- keep non-PFT14 natural-trigger branches explicitly guarded;
- treat `nbp_glo>1` routing and missing routing-map fields as production
  asset work, not a blocker for the 669 independent single-landpoint paper
  workflow;
- optimize performance before routine long annual or 669-case validation.
