# Compiled Transition State Ledger

Scope: active paper-case PFT14 driver runtime after semantic closure through
the current one-year validation windows. This ledger is an implementation plan
for performance work only; it does not change Fortran process semantics.

## Current Bottleneck

The warmed 30-day lite/compact profile is dominated by Python half-hour
orchestration, repeated JAX array conversion/indexing, and daily STOMATE
carbon:

- `paper_1961_driver_later_day_runtime_result`
- `_paper_1961_next_step_runtime_result_compact`
- JAX `asarray`, indexing, and scatter helpers
- `_paper_day_stomate_daily_carbon_from_bundles`
- `stomate_daily_process_fold_single_pass_from_entries`
- `run_hydrol_first_step_module_from_precall`

Small wrapper and payload experiments were reverted in
`docs/source_audits/current_performance_notes.md`; the next meaningful
boundary is a coarser compiled transition with static configuration separated
from dynamic arrays.

## Candidate 1: Half-Hour SECHIBA Runtime Step

Fortran order:

- `src_driver/dim2_driver.f90` lines 839-908 advances forcing steps.
- `src_sechiba/sechiba.f90::sechiba_main` lines 997-1224 calls DIFFUCO,
  ENERBIL, HYDROL, CONDVEG, and THERMOSOIL.
- `src_stomate/stomate.f90` lines 3187-3267 consumes same-step SECHIBA
  outputs for daily accumulators.

Dynamic state packet:

- `driver_previous_step_state`: `albedo`.
- `diffuco_previous_step_state`: `temp_sol`, `soilalbedo_bg`, `height`,
  `control_salinity`, `control_inudate`.
- `enerbil_previous_step_state`: thermal/albedo state used by
  `assemble_enerbil_first_step_precall_payload`.
- `hydrol_previous_step_state`: soil water, snow, `free_drain_coef`,
  `zwt_force`, `njsc`, `fwet_new`, water-table and peat-routing state.
- `thermosoil_previous_step_state`: `ptn`, `cgrnd`, `dgrnd`, `cgrnd_snow`,
  `dgrnd_snow`, `lambda_snow`, `gtemp`, `refSOC`.
- `slowproc_stomate_previous_step_state`: vegetation and daily state carried
  unchanged within a STOMATE day, including `veget`, `veget_max`, `lai`,
  `frac_nobio`, `height`, `biomass`, `leaf_frac`, `leaf_age`,
  `when_growthinit`, `daily_accumulators`, and carbon pools.

Per-step forcing inputs:

- `zlev`, `zlevuv`, `u`, `v`, `qair`, `temp_air`, `pb`, `precip_rain`,
  `precip_snow`, `lwdown`, `swdown`, `ccanopy`, `salinity`, `tide_height`.
- These must remain source-backed through the active `readdim2.f90`
  interpolation/spreading path.

Static config candidates:

- `Paper1961PreparedDriverContext` arrays and scalars:
  `dt_sechiba`, `dt_sechiba_days`, `min_wind`, `ok_explicitsnow`,
  `ok_freeze_cwrr`, `ok_thermodynamical_freezing`, HYDROL vertical grids and
  constants, DIFFUCO PFT14 static kwargs, THERMOSOIL grids, `ok_laidev`,
  `is_tree`, `is_peat`, and run-definition booleans.
- `DriverDiffucoDayStaticCache` for DIFFUCO SAVE controls and slowproc-derived
  static fields within a STOMATE day.
- HYDROL static pre-call template built from the day's initial slowproc state.

Objects to keep outside a first compiled prototype:

- `DriverPreviousStepStatePacket` dictionaries.
- `SimpleNamespace` wrappers.
- `StomateRestartInputBundles` dictionaries.
- string-valued configuration such as `pheno_model`.
- provenance tuples and diagnostic scaffold objects.

Minimum safe prototype:

1. Add a typed numeric state container for one later-day half-hour step.
2. Write adapters from the existing packet/dicts to that container and back.
3. Compile only the numeric component transition for one step, preserving the
   current Python wrapper for provenance, missing-field checks, and fallback.
4. A/B against the existing compact runtime with:
   - 3-day lite/scaffold parity at bit-exact tolerance.
   - day60 or day365 GPP/carbon trace comparison.
   - 30-day warmed timing profile.

Current local runtime boundary:

- `jax_orchidee.driver.orchestration::_paper_1961_later_day_half_hour_transition`
  now owns the 48 half-hour SECHIBA/STOMATE-entry transition for a later
  STOMATE day. `paper_1961_driver_later_day_runtime_result` binds the per-day
  static caches and calls this boundary instead of carrying the step loop
  inline.
- The function still calls the existing source-backed compact step and is not
  itself a compiled `lax.scan`. Its purpose is concrete: there is now one
  replacement point for typed state/forcing work. A 7-day cProfile sample after
  the refactor reported this boundary directly (`~6.06 s` cumulative under a
  noisy profiled run), with `_paper_1961_next_step_runtime_result_compact`
  remaining the inner hotspot (`~5.60 s`).
- Required guard before replacing it with a scan: preserve
  `tests/parity/test_driver_1961_orchestration.py::test_multiday_modelout_lite_run_matches_scaffold_outputs_through_third_day`
  and
  `test_multiday_lite_prebuilt_day_payloads_match_step_local_payloads_through_second_day`.

## Candidate 2: Daily STOMATE Carbon Core

Fortran order:

- `src_stomate/stomate.f90` lines 4225-4364 enters daily STOMATE processes.
- `src_stomate/stomate_lpj.f90` lines 944-960 calls prescribe and
  constraints.
- `src_stomate/stomate_phenology.f90` lines 319-563 runs the active
  `pheno_model='none'` scheduling path and the shared leaf-onset growth/reset
  block. `pheno_hum` lines 650-789, `pheno_moi` lines 838-958,
  `pheno_humgdd` lines 1029-1197, `pheno_moigdd` lines 1261-1456,
  `pheno_ncdgdd` lines 1520-1615, and `pheno_ngd` lines 1668-1750 are
  covered for explicit source-backed onset triggers with non-`ok_LAIdev`
  leaf/root reserve allocation. `pheno_moi_C4` lines 1757-1902 and
  `pheno_siggdd` lines 1910-2061 are also covered. The `pheno_ncdgdd` path
  returns the Fortran `gdd_midwinter=undef` writeback when onset begins. The
  `pheno_moigdd` natural-PFT path includes `pheno_moigdd_t_crit`; its
  `ok_LAIdev` crop-LAI branch is covered for the `slai/pdlai` onset condition
  and `deltai/ssla` biomass forcing in lines 1381-1387 and 532-548.
- `src_stomate/stomate_lpj.f90` lines 1358-1380 now has a local dispatch
  helper for the `update_peatfrac` branch, and the post-NPP daily cover wiring
  now dispatches `wire_cover=True, update_peatfrac=True` through the same
  source-backed boundary. The downstream
  `lpj_cover_peat` single-peat-PFT path is covered for
  target fraction application, initial peat establishment, non-peat fraction
  adjustment, biomass/flux dilution, and `carbon_save`/`delta_fsave`
  bookkeeping from lines 2583-3337, including the `OK_PC=.TRUE.` deepC
  save/scale clauses in lines 3093-3097, 3113-3117, 3136-3147, 3160-3164,
  3183-3204, and 3220-3304. The high-level post-NPP test
  `test_stomate_daily_carbon_kill_gap_turnover_explicit_wires_peat_cover_dispatch_when_requested`
  verifies the daily cover entry against direct `lpj_cover_peat_step`.
  `stomate_lcchange::agripeat_adjust_fractions`
  lines 1640-1863 is covered for the hard-coded PFT12-16 fraction adjustment;
  `lcchange_main_agripeat` lines 1342-1558 is covered locally for agripeat
  litter, biomass-to-litter, legacy/deep-carbon, and product-entry
  redistribution after adjusted fractions; lines 1561-1613 are covered for
  product-pool aging, history-flux scaling, and aboveground-litter fuel
  rebalancing. The full hard-coded `lcchange_main_agripeat` wrapper lines
  1253-1613 is covered locally by `lcchange_main_agripeat_step`, including
  expansion initialization and exhausted-PFT clearing. The non-age-class daily
  LCC entry dispatch in `stomate_lpj.f90` lines 1494-1505 is now covered by
  the restart-state adapter `stomate_lcchange_main_agripeat_from_restart_state`
  and the daily scheduling test for the hard-coded PFT12-16 agripeat path.
- `src_stomate/grassland_management.f90` lines 1321-1353 and 970-1037 are
  covered for the grazing gate and C3/C4 role selection. The pre-animal
  development/regrowth state in `Main_appl_pre_animal`, `cal_devstage`, and
  `cal_tgrowth` lines 2553-2722 is covered for explicit temperature-memory,
  `devstage`, `tgrowth`, and `tcut0` writeback. The user-defined cutting
  scheduler in lines 1872-1897 is covered for due-date flagging, `tcut_verif`,
  `compt_cut`, `when_growthinit_cut`, and `lm_before` writeback; the triggered
  execution boundary in lines 1898-1954 and `grassland_cutting.f90::cutting_spa`
  lines 48-304 is covered for cut biomass, yield/loss, `regcount`, and
  `wshtotsumprev` writes. The auto-fauche scheduling tests in lines 1975-1993,
  2075-2089, 2177-2211, and 2296-2314 are covered locally for their flag,
  `countschedule`, `compt_cut`, and `when_growthinit_cut` updates. Full animal
  modules, fertilization follow-up, and complete autogestion/postauto optimizer
  sequences remain outside this candidate.
- `src_stomate/stomate_alloc.f90` and downstream NPP/turnover/gap processes
  are called before modelout fields.

Dynamic daily inputs:

- `prescribe_inputs`: biomass, PFT presence, individual density, leaf age and
  fraction, `when_growthinit`, `veget_max`, and related mutable STOMATE state.
- `constraints_inputs`: `adapted`, `regenerate`, temperature state, and PFT
  masks.
- `alloc_inputs`: LAI, age, soil/temperature/humidity daily fields, SLA, and
  allocation state.
- `post_npp_inputs`: daily GPP, maintenance respiration, turnover/litter/carbon
  pools, phenology state, and OK_LEAK downstream state.

Static config candidates:

- Numeric parameter arrays from `stomate.parameters`.
- Allocation kwargs currently parsed from `run.def`: `F_FRUIT`, `ECUREUIL`,
  `ALLOC_SAP_ABOVE_GRASS`, `MIN_LTOLSR`, `MAX_LTOLSR`, `Z_NITROGEN`.
- Boolean masks such as `pheno_is_none`, `ok_laidev`, `is_tree`, `natural`,
  `pasture`, and `pft_to_mtc`.

Objects to keep outside a first compiled prototype:

- `pheno_model` string tuples. The active PFT14 path must be converted to a
  source-backed numeric/static mask (`pheno_is_none`) before compiling.
- Python dictionaries for `prescribe_inputs`, `constraints_inputs`,
  `phenology_inputs`, `alloc_inputs`, and `post_npp_inputs`.
- Named diagnostic result graphs when not needed for runtime modelout.

Minimum safe prototype:

1. Keep phenology source-driven as onset models are added. `phenology_none_step`
   preserves active `pheno_model='none'` scheduling, and public
   `phenology_step` now also covers active `pheno_model='hum'`, `moi`,
   `humgdd`, `moigdd`, `ncdgdd`, `ngd`, `moi_C4`, and `siggdd` with explicit
   inputs and non-`ok_LAIdev` biomass/leaf-age reset.
2. Keep the existing `phenology_step` public function and unsupported-branch
   checks for source-driven branch coverage.
3. Compile only the daily numeric core after inputs are packed into a typed
   container.
4. Require bit-exact parity with the current explicit chain before enabling it
   by default. Previous `jax.lax.scan` daily folding changed results at
   `~1e-16`; this path must explicitly decide whether bit-exactness is
   required or whether a documented tolerance applies.

## Validation Gate For Any Retained Performance Change

Every retained performance change must provide:

- profile evidence identifying the target boundary;
- A/B warmed timing using the same command shape;
- short semantic parity (`tests/parity/test_driver_1961_orchestration.py -k
  "later_day or day2 or runtime"`);
- one process-trace check at day60 or later;
- an update to `docs/source_audits/current_performance_notes.md`;
- no silent fallback for unsupported Fortran branches.
