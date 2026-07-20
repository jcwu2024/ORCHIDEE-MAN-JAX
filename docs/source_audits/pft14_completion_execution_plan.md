# PFT14 Completion Execution Plan

Scope: ORCHIDEE-MAN JAX rebuild for the PFT14 mangrove target, with source-driven
coverage beyond the active paper-case path only where the branch is PFT14
relevant.

## Non-Negotiables

- Fortran source is the process truth.
- Reference outputs and traces validate source-backed implementations; they do
  not replace source provenance.
- Do not bridge gaps with approximations, tuning, or "make it run first" logic.
- Every supported process or branch must have a Fortran file, subroutine, and
  line-span reference or an audited trace contract.
- Single-point paper-case closure proves only that active path. It is not
  coverage of PFT14-relevant conditional branches.
- The paper-scale spatial product is a mosaic of independent single-landpoint
  cases, not one `nbp_glo>1` multi-landpoint routing run. Current paper
  reproduction should therefore prioritize one-cell driver parity, restart
  handoff, and 669-case orchestration before true multi-landpoint routing.
- Passing all 669 paper mosaic cases is still only the paper-workflow target.
  It must not be used to defer source-driven audit and equivalence of PFT14
  branches that could activate after changing forcing data, landpoint state,
  hydrological/salinity/tide conditions, or continuous model parameters.
- The one-week execution focus is PFT14 semantic closure, not expanding the
  paper mosaic execution layer. The 669 cases validate and assemble the paper
  product after the source-backed PFT14 branch ledger is closed; they do not
  multiply the amount of model process logic to port.
- Do not spend near-term implementation effort on branches that require
  non-mangrove PFT roles or unrelated management switches and cannot activate
  merely by changing PFT14 forcing, grid, or continuous parameters.

## Execution Order

1. Close one active paper-case PFT14 single-landpoint path from forcing through
   annual modelout and restart/year handoff, using short, checkpointed
   validation gates and the existing 1961-2010 reference/modelout targets.
2. Generalize that closed single-landpoint path into the paper mosaic runner:
   enumerate the 669 landpoint case definitions, keep each case as `nbp_glo=1`,
   preserve the Fortran run.def/forcing/static selection for each case, and
   aggregate outputs only after each one-cell run has passed its local gates.
3. Run a small representative landpoint subset before the full 669-case
   mosaic. Use failures to return to source-backed process/branch ledgers, not
   to invent landpoint-specific fixes.
4. Complete source-backed SECHIBA/STOMATE structural branches that can affect
   PFT14 under changed land point, forcing data source, hydrological/salinity
   regime, or continuous parameters:
   HYDROL runoff/tide/peat routing, DIFFUCO salinity/tide controls, ENERBIL
   active snow/energy boundaries, THERMOSOIL recurrence state, OK_LEAK/TF-DOC,
   OK_PC/dynamic peat, LD_DOC, DGVM/PFT in-out, and PFT14 LCC/agripeat.
5. Complete source-backed STOMATE structural branches that can affect PFT14:
   LCC, harvest, age-class movement, product pools, DGVM/PFT in-out, and
   annual-memory state.
6. Audit restart/SAVE/annual-memory fields as a ledger before chasing later
   years or new grid/forcing cases.
7. Build source-driven branch coverage from Fortran `IF`, `WHERE`,
   configuration switches, PFT masks, time branches, and state masks, then
   classify each branch as `PFT14 active`, `PFT14 conditional`, or
   `Non-PFT14 natural trigger`.
8. Design minimal micro-cases for uncovered PFT14-relevant branches; do not rely
   on long single-point runs to discover them.
9. Optimize performance before scaling from the representative subset to all
   669 cases if runtime would otherwise make validation impractical. Keep every
   optimization behind profile evidence, A/B timing, and semantic checks.
10. Use long annual or multiyear runs, including the 2010 paper CSV target, as
   validation gates for the active paper path and final 669-case mosaic after
   the relevant source-backed branch work and performance work are complete.
11. After the 669-case paper mosaic closes, continue the PFT14-relevant
    branch ledger before treating the model as generally equivalent for new
    forcing products or retuned parameters. Long runs may confirm the ledger,
    but must not replace it.

## Current Near-Term Track

1. LCC local dispatcher and state writeback are source-backed through
   `gross_glcc_firstday_fh` allocation, receiver application, product-pool
   aging, and restart/season state update.
2. Paper mosaic orchestration is now represented explicitly:
   `configs/orchidee_man_250919.yaml` records the 669 independent
   single-landpoint execution unit, `jax_orchidee.driver.paper_mosaic`
   discovers `arg2_*/*/I*/S*/run.def` case directories, builds manifest rows
   with run-definition zoom metadata and reference-history availability, and
   emits deterministic case-year tasks for validation runners. The
   `scripts/dev/inspect_paper_mosaic_cases.py` CLI audits case counts, missing
   history years, one-cell history dimensions, and optional case-year task
   readiness. The manifest summary explicitly keeps changed forcing,
   landpoint-state, and continuous-parameter PFT14 branch audit outside the
   paper mosaic closure claim.
   Driver bundle preparation now threads the selected case `run.def` through
   both domain zoom metadata and run-scalar/PFT parameter reads, so 669-case
   runs and future parameter experiments do not silently mix case-specific
   limits with default run-scalar values.
   Archived case `run.def` files are not assumed to contain materialized
   Fortran defaults: `jax_orchidee.driver.run_def_materialization` merges the
   audited base `used_run.def` defaults with static case overrides while
   excluding year-dynamic forcing, CO2, and restart keys. The
   `scripts/dev/materialize_paper_mosaic_run_defs.py` CLI now writes
   per-case complete `used_run.def` files under `outputs/`, and the local
   reference case prepares a driver context from that generated file.
   Runtime task readiness now also requires case-local `driver_start.nc`,
   `sechiba_start.nc`, and `stomate_start.nc`. The driver restart aggregator,
   first-step STOMATE boundary, and prepared driver context accept an explicit
   case run directory, so mosaic execution can use the selected landpoint's
   restart files rather than silently falling back to the default local
   reference case.
3. Minimal daily `do_now_stomate_lcchange` scheduling for the supported
   multi-age gross-LCC branch is covered by a micro-case: inactive scheduling
   returns state unchanged; active scheduling requires explicit GLCC matrices,
   mapping arrays, process outputs, clears the switch, sets the done flag, and
   normalizes post-LCC `veget_max`.
4. Remaining LCC branch work is source-driven: ordinary non-age-class
   `lcchange_main` is covered for the MICT leak-pool path with explicit daily
   state writeback; non-age-class `lcchange_deffire` is covered for the
   `OK_PC=.FALSE.` legacy litter/carbon path with explicit fire-proxy inputs.
   Non-degenerate `SingleAgeClass` gross-LCC first-day allocation and daily
   dispatch are now covered against `stomate_glcchange_SinAgeC_fh.f90` for the
   simplified harvest aggregation, `glcc_compensation_full`, and downstream
   receiver/product-pool boundary. Peat dynamic-cover/agripeat daily entry
   variants are now covered for their source hard-coded paths, including
   `ok_pc=True` deepC redistribution and restart-state writeback through the
   daily dispatcher. Age-class gross LCC ignores `dyn_peat`/`agri_peat` after
   the separate peat-cover boundary, matching the source section order. A
   multi-age gross-LCC multi-transition stress case is covered against explicit
   allocation-plus-apply composition. The multi-age gross-LCC deforestation-fire
   proxy boundary is covered for tree-to-non-tree transitions from
   `glcc_pftmtc`. OK_PC/deep_carbcycle now has local restart-gas firstcall,
   source-order explicit deep-carbon kernels, and two-day sidecar handoff
   micro-cases. A 2026-07-09 narrow server run confirmed that the unmodified
   full Fortran driver aborts at `stomate.f90::stomate_main` lines 3093-3127
   when `OK_LEAK=n`; combined with `control.f90` lines 271-278 forcing
   `OK_PC=.FALSE.` whenever `OK_LEAK=y`, this makes daily OK_PC/deep_carbcycle
   full-driver tracing unreachable without changing Fortran truth. Keep OK_PC
   as source-local branch coverage unless the project explicitly changes that
   Fortran guard.
5. Restart/SAVE annual-memory audit now has three source-ledger assets:
   `stomate_io` restart-field scan, `src_stomate` SAVE-state scan, and a
   focused 56-field active annual-memory audit. The broad SAVE coverage audit
   now classifies all 1094 SAVE records with 0 unclassified records.
6. The broad PFT14-relevant SAVE backlog is cleared in the audit. Former
   leak/permafrost/DGVM-light records were either moved to source-covered local
   process state (`rootlev`, `z_root`, cryoturbation coefficients, `Cmax`,
   Moyano moisture scratch, and `fpc_max`), static configuration
   (`use_new_cryoturbation`, cryoturbation method/depth/diffusion constants),
   paper-inactive OK_PC/permafrost-Cforcing daily buffers, or source-local
   snow-geometry caches with no downstream active OK_LEAK state effect.
   `stomate_permafrost_soilcarbon.f90` SAVE records remain routed to the
   inactive OK_PC/deep-cycling branch ledger because the paper run has
   `OK_PC = n` and `STOMATE_CFORCING_PF_NM = NONE`.
   Annual/multiyear runs remain validation gates, not branch discovery.
7. The SECHIBA branch ledger is tracked in
   `docs/source_audits/sechiba_pft14_branch_microcase_matrix.md`. Near-term
   PFT14-relevant SECHIBA branch work should come from that matrix, especially
   explicit-snow nonzero cases if any of the 669 landpoint cases can activate
   snow. The switch-driven bucket-snow path now has HYDROL, CONDVEG, ENERBIL,
   and THERMOSOIL no-explicit-snow local handoff coverage. `READ_LAI=y` is
   covered for explicit already-interpolated `laimap` payloads through
   `slowproc_lai` and cold-start `slowproc_veget`; LAI-map file interpolation
   remains a separate source boundary. True
   multi-landpoint routing is not a paper blocker because the paper product is
   assembled from independent `nbp_glo=1` cases; routing/LD_DOC should only
   re-enter the near-term track if a target workflow enables `nbp_glo>1`
   routing or long-distance DOC.
8. DGVM local kernels include pftinout/light/establish/cover/kill and now the
   `stomate_season` annual `ok_dgvm=true` maxfpc/leaf-mass relaxation branch.
   Remaining DGVM work is run-level validation/configuration if that switch is
   promoted to a target workflow, not a known unimplemented local annual-memory
   formula.
9. Continue branch micro-case matrix updates after each supported branch.
10. Explicitly defer grassland/crop/STICS/fire and other non-PFT14 natural
   triggers unless the project scope changes. These branches may remain guarded
   or locally tested, but they should not displace peat, DOC, hydrology,
   restart, or OK_LEAK/OK_PC work.
11. Near-term work must return to source-driven PFT14 branch closure whenever
    paper mosaic orchestration is sufficient for validation. Adding more
    669-case plumbing is lower priority than clearing PFT14-relevant source
    guards and micro-cases.

## 2026-07-09 Closure Checkpoint

Current executable evidence supports the intended closure target without
expanding the claim beyond available assets:

- Local paper mosaic discovery currently sees only the checked-in
  `case_001_071` asset. That case is runtime-ready with its local
  `driver_start.nc`, `sechiba_start.nc`, `stomate_start.nc`, reference
  `stomate_history_1961.nc`, and materialized `used_run.def`. This confirms
  the independent single-landpoint execution protocol locally; it is not a
  claim that all 669 paper cases are present in the local workspace.
- Focused modelout, paper-mosaic, branch-ledger, and switch-guard tests pass:
  `tests/unit/test_stomate_modelout.py`,
  `tests/unit/test_paper_mosaic.py`,
  `tests/unit/test_pft14_branch_ledgers.py`,
  `tests/unit/test_driver_orchestration_switches.py`, and
  `tests/parity/test_stomate_pft14_parameters_modelout.py`.
- Driver-level three-day gates pass for scaffold, lite modelout, compact
  later-day modelout, and year-handoff contract tests in
  `tests/parity/test_driver_1961_orchestration.py`. These gates confirm the
  local chain can carry forcing-driven state through day-end modelout and a
  rebased year-handoff packet without missing dynamic components.
- STOMATE entry-boundary tests now distinguish legacy bridge-trace gaps from
  the current source-backed production assembly. The bridge trace alone still
  reports missing thermosoil/static/restart/hydrol/erosion fields, but
  `tests/unit/test_stomate_entry.py::test_fully_sourced_entry_payload_has_no_blocking_process_gaps`
  assembles the corresponding driver, slowproc, restart, thermosoil, hydrol,
  erosion, and no-routing sources and leaves no blocking process input gaps;
  only IO handle arguments remain outside the ecological process state.
- Focused PFT14-conditional STOMATE branch regressions pass for LCC/agripeat,
  DGVM/PFT in-out, OK_PC/deep-carbon source-local kernels, dynamic peat,
  fixed cryoturbation, LD_DOC, Cforcing, season, and restart classifications.
- Focused PFT14-conditional SECHIBA/soil/energy regressions pass for bucket
  snow, peat and water-table hydrology, irrigation/flood boundaries,
  explicit-snow switch handling, LD_DOC/OK_LEAK coupling, and no-routing
  single-landpoint behavior.
- The retained five-year annual validation
  `outputs/reference_mode/multiyear_1961_1965_current_paper_csv_gate.json`
  completes 1961-1965 with zero year-handoff gaps and annual modelout
  absolute errors below `1e-6` scale for the recorded AGB/BGB/GPP/NPP fields.
- The follow-up closure sweep is recorded in
  `docs/source_audits/pft14_closure_sweep_20260709.md`. It classifies stale
  bridge-trace wording separately from current source-backed production
  closure, keeps routing as a production asset gate for `nbp_glo>1`, and
  leaves non-PFT14 natural-trigger branches guarded rather than counted as
  near-term PFT14 semantic work.

Remaining work must continue to respect the same boundary: source-driven
PFT14 branch closure first, performance before routine long mosaics, and full
669-case paper validation only when those case assets are available.

## Current PFT14-Relevant Priority

1. `PFT14 active`: current paper-case forcing-to-modelout, restart/year
   handoff, 669 independent single-landpoint case orchestration,
   HYDROL/DIFFUCO/ENERBIL/THERMOSOIL, OK_LEAK, TF-DOC, peat/perma-peat active
   path, and modelout diagnostics.
2. `PFT14 conditional`: branches that still concern mangrove peat/DOC/water or
   redox/soil state but require switches not active in the paper run, including
   OK_PC/deep_carbcycle, dynamic peat, LD_DOC, fixed cryoturbation, DGVM, and
   PFT14 LCC/agripeat scenarios.
3. `Non-PFT14 natural trigger`: grassland grazing/cutting, crop/STICS, ordinary
   grassland management, and fire. These do not naturally activate by changing
   PFT14 forcing/grid/continuous parameters and are not current closure targets.

## Stop And Report Conditions

- A new server trace or Fortran run is required.
- A source ambiguity affects the implementation direction.
- A validation result reveals a semantic mismatch in an already claimed path.
- A real stage closes, such as LCC local dispatcher closure or restart ledger
  completion.

## Performance Rule

Optimize only with profile evidence, A/B timing, and semantic verification.
Do not keep a faster path if it changes numerical semantics without an explicit
tolerance decision in the numeric ledger.
