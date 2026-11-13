# PFT14 Completion Execution Plan

Scope: ORCHIDEE-MAN JAX rebuild for the PFT14 mangrove target, with source-driven
coverage beyond the active paper-case path only where the branch is PFT14
relevant.

## Current Controller Decision (2026-11-13)

This section and the execution order below govern current main hardening.
The July implementation notes retained below are historical evidence, not a
current-source acceptance claim or an instruction to restart completed work.
The release authority remains the
[PFT14 equivalence contract](pft14_equivalence_delivery_contract.md).

The user requests source-semantic equivalence and maintainable repairs. Keep
the existing process/driver/runner architecture and the explicit JAX state
design. Functional state, pytrees, JIT and other necessary JAX restructuring
are acceptable when they preserve the source contract. No general rewrite of
JAX or cleanup of original Fortran is planned.

The main engineering risk to reduce is drift between parameter/state owners
at cold start, first day, ordinary steps, compiled steps, day/year boundaries
and restart I/O. A repair should close the relevant real paths, not only a
helper. Consolidate genuinely shared field/default/adapter rules locally;
keep intentional phase-specific behavior explicit. Do not introduce one
unconditional state copier across different Fortran phases.

For each accepted repair retain source necessity, a reproducible old failure
and new pass, caller/writeback coverage, exact commit, and evidence limits in
the existing correction records. Keep structural cleanup separable from
semantic changes. Avoid point-specific exceptions, silent fallback defaults,
parameter tuning and tolerance changes that conceal lost or incorrect state.
Maintain relevant provenance and update stale nearby documentation.

Natural floating-point sensitivity is a separate finding from a translation
defect. Once the native observations, source conditions and bounded same-input
checks support that conclusion, record its scope and stop reopening the same
causal question without new contradictory evidence. Do not require bit identity
or a unique machine-instruction explanation. A small unexplained difference is
not automatically roundoff; a bounded numeric attribution does not make a
failed strict-state or scientific gate pass. Keep these acceptance results
separate and retain the original thresholds.

### Starting Point And Immediate Work

- Published baseline for this decision: main
  `73c942b9f086e6ac8255c5fc03ac21155e896376`. C1-C40 and their existing
  evidence are reused, not re-audited from scratch. Their scopes and historical
  producer identities remain in [the current review](main_physics_review.md),
  [the correction index](shared_core_branch_review.md) and
  [the origins audit](correction_origins.md).
- The existing Sol owner continues the assigned offline DOC/carbon accounting
  of saved 319/1962 memory/file-restart outputs. That assignment has zero new
  model integrations or remote runs. Determine physical units, geometry,
  weighting, stock differences and missing flux observations; do not turn a
  raw coordinate sum into physical mass or infer unrecorded fluxes.
- Retain the open DOC/full-restart-state limits, independent thermal
  initialization/first-consumer numerical truth, explicit-snow maxmass domain,
  and historical 001 input/IOIPSL identity limits. These are not all confirmed
  bugs. Source/schema gaps retained in the current review must also be included
  in the remaining-work reconciliation; this short list is not exhaustive.
- The next execution package reconciles the existing source/state ledgers with
  accepted evidence into one finite remaining-work table. Reuse accepted gas
  lifecycle and original-bug provenance work. Independent source checks need
  not wait for another DOC trajectory or for unavailable historical identity.
- Continue manual coordination through the existing Sol owner. The controller
  owns scope, scientific judgments and acceptance. Finish coherent packages
  rather than stopping after each small finding. Daily/global synchronization
  remains paused until main reaches the agreed handoff checkpoint.

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
- The current execution focus is PFT14 semantic closure. The 669 cases validate
  and assemble the paper product after source and state closure; they do not
  multiply the amount of model process logic to port.
- Do not spend near-term implementation effort on branches that require
  non-mangrove PFT roles or unrelated management switches and cannot activate
  merely by changing PFT14 forcing, grid, or continuous parameters.

## Execution Order

1. **Reconcile the remaining coverage.** Reuse `pft14_reachable_ledger.yaml`,
   source-owner/arm-disposition ledgers, branch micro-case matrices,
   `restart_state_lifecycle.yaml`, compiled-state records and accepted reviews.
   For each relevant source branch/state edge record its existing evidence
   level, actual production owner/caller, producer identity, supported domain,
   missing evidence and next smallest action. Distinguish confirmed defects,
   untested paths, numeric sensitivities and unavailable reference identity.
   Do not inflate coverage from a test count or an old generic `verified` label.
2. **Close state and parameter seams first.** Check effective configuration,
   source defaults, units, shapes/axes, update phase, masks, integer/index rules,
   and writeback from cold/restart initialization through the real first-day,
   ordinary, compiled, day-end/year-end and file writer/reader paths. Use
   nondefault distinguishable inputs so defaults cannot hide omissions. Make
   shared contracts executable where feasible, retaining phase-specific rules.
   This complements, rather than repeats, the accepted gas lifecycle cases.
3. **Complete module semantic coverage.** Work through driver/forcing/parameters,
   HYDROL/DIFFUCO/ENERBIL/THERMOSOIL, STOMATE including carbon/DOC, and
   restart/modelout. Select only unresolved entries from step 1. Pair source
   conditions, sequential mutations, callers and returned state with original
   Fortran micro-drivers or audited immutable boundaries. Use legal cases on
   both sides of thresholds, zero/nonzero and mask/configuration combinations;
   do not wait for long runs to happen to trigger them. Validate complete
   same-input tuples before attributing cross-trajectory differences to code.
4. **Protect each closure automatically.** As steps 2-3 land, add lightweight
   core regressions and ledger/ownership checks to CI, alongside the existing
   CLI/configuration smoke checks. Exercise the production compiled route at
   the appropriate bounded gate; Python helper or JAX-vs-JAX passes alone do
   not supply Fortran truth. Keep external-data/Fortran/long-running gates
   separate from fast CI. Every repair carries its targeted regression and
   source rationale; do not defer all protection until the final sweep.
5. **Validate integration on a declared release candidate.** Choose a small
   representative set from the branch/state coverage map, including cold start,
   nondefault file restart, day/year handoff and relevant hydrology/snow/carbon
   regimes. Compare declared outputs, carried state and unit-consistent balance
   checks at matching boundaries. Reuse unchanged evidence with an explicit
   source/input bridge; historical runs keep their original producer. Only
   after source/state closure and representative gates, run the contract's
   complete 669-case independent-single-landpoint acceptance when assets and
   execution resources are available. This plan does not launch that workload.

### Progress And Completion

The next coverage reconciliation must publish counts against a stated source
and state denominator: classified entries, each evidence level, missing
production edges, confirmed unfixed defects, and numerical/reference limits.
Report both the counts closed and the remaining work after each coherent
package; do not estimate a completion percentage from the number of bugs fixed.

The stopping condition is the release gates in the equivalence contract:
reachable branches classified and production-verified, no missing/ambiguous
required state edges, and the declared integration/scientific checks passed.
Accepted floating-point effects retain their bounded status; unresolved or
failed gates remain visible. Completion is a claim about the declared PFT14
domain, not a guarantee covering every original ORCHIDEE configuration or all
possible real-valued inputs. A missing historical producer may bound a
historical comparison without blocking unrelated source work; a new identified
Fortran reference cannot retroactively supply that historical identity.

## Historical Implementation Track (Through 2026-07-09)

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

## Execution Boundaries And Escalation

- A roadmap entry is not a new model-run or resource grant. Apply the current
  package's existing authorization; prepare any genuinely out-of-scope capture
  or compute request concretely before asking the user to decide.
- Routine in-scope fixes, validation and related documentation should finish
  within their assigned coherent package. Source ambiguity that changes the
  intended model semantics goes to the controller before choosing a behavior.
- Report a confirmed defect in an accepted path with its affected evidence and
  a minimal regression; do not discard unrelated completed validation.
- Delivery includes actual artifacts, checks performed, remaining uncertainty
  and one prioritized next package. Follow-up is manual, with no scheduled
  messages or unsolicited reports sent back to the controller chat.

## Performance Rule

Optimize only with profile evidence, A/B timing, and semantic verification.
Do not keep a faster path if it changes numerical semantics without an explicit
tolerance decision in the numeric ledger.
