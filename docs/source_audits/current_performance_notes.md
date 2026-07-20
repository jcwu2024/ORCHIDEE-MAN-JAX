# Current Performance Notes

Scope: local paper-case PFT14 multiday modelout runner on the Windows ORCJAX
environment. These numbers are engineering diagnostics, not scientific truth.

## Baseline

Command shape:

```powershell
C:\Users\admin\.conda\envs\ORCJAX\python.exe scripts\dev\run_multiday_modelout_timing.py --mode lite --single-pass-daily-fold on --compact-later-days on --show-days 0
```

Observed on 2026-07-03:

- 3 days, cold process/no warmup: `99.04 s`.
- 3 days, after one in-process warmup: `5.73 s`, `1.91 s/day`.
- 7 days, cold process/no warmup: `105.90 s`.
- 7 days, after one in-process warmup: `10.27 s`, `1.47 s/day`.
- 14 days, after one in-process warmup: `17.19 s`, `1.23 s/day`.

Interpretation:

- The dominant cost for short validation scripts is JAX/XLA compilation and
  cache setup, not steady-state model arithmetic.
- Running separate short scripts repeatedly is therefore inefficient. Longer
  validation should reuse a single Python process and an already-warmed runner.

## Warmed Profile

Profile artifact:

- `outputs/dev_multiday_7day_warm_current.prof`
- `outputs/dev_multiday_7day_warm_after_vmax_vectorized.prof`

Top cumulative regions from the warmed 7-day run:

- `paper_1961_driver_later_day_runtime_result`: `12.51 s`.
- `paper_1961_next_step_hydrol_precall_scaffold`: `4.54 s`.
- `stomate_daily_process_fold_single_pass_from_entries`: `4.16 s`.
- `stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit`:
  `4.27 s`.
- JAX scatter/indexing calls account for several seconds of the remaining
  warmed runtime.

After vectorizing `stomate_vmax::vmax` leaf-age/Vcmax algebra and reusing the
prepared STOMATE static/boundary inputs, the warmed 7-day profile shifted to:

- `paper_1961_driver_later_day_runtime_result`: `9.40 s`.
- `paper_1961_next_step_hydrol_precall_scaffold`: `5.22 s`.
- `paper_1961_driver_day_scaffold`: `3.56 s`.
- `_paper_day_stomate_daily_carbon_from_bundles`: `2.10 s`.
- `stomate_daily_process_fold_single_pass_from_entries`: `1.71 s`.

## Optimization Direction

Immediate validation workflow:

- Prefer one warmed process for 14-day and longer checks.
- Avoid repeatedly launching cold scripts for small windows unless testing
  import/compile behavior itself.
- Keep short first-divergence tests for semantic debugging; run long windows
  only after the short window is clean.

Model runner optimization:

- The next useful speedup is not inside one physical formula. It is reducing
  Python-level day/timestep orchestration and repeated JAX scatter/indexing.
- Candidate work:
  - carry less diagnostic payload in later days when `compact_later_days=True`;
  - move repeated half-hour stepping into a coarser JAX loop once the state
    packet is stable enough;
  - keep compiled module kernels keyed only by shape/static config, not by
    changing day/tstep values;
  - add A/B timings for every performance change and revert changes that do
    not improve warmed runtime.

Year-scale implication:

- A naive cold one-year validation may pay about `~100 s` compile/setup plus
  warmed day-loop cost. At the current warmed `~1.2-1.5 s/day`, this is still
  too slow for frequent development loops.
- Before routine one-year validation, either run it in a warmed process or
  first improve the day-loop/runtime runner. Do not use repeated cold yearly
  scripts as the default development workflow.

## 2026-07-04 Optimization Pass

Retained changes:

- `stomate_daily_process_fold_single_pass_from_entries` now uses the existing
  source-backed batched maintenance path when runtime diagnostics are not
  retained. This preserves the Fortran entry order while avoiding per-entry
  maintenance result objects in the lite runner.
- `vmax_step` now evaluates the `stomate_vmax::vmax` leaf-age, leaf-fraction,
  and Vcmax algebra across all PFTs with array operations instead of Python
  PFT/leaf-class scatter loops. The Fortran branch for DGVM evergreen PFTs is
  still explicit.
- `Paper1961PreparedDriverContext` now carries the paper-case STOMATE static
  and vertical boundary kwargs for reuse in later-day folds. If the point count
  does not match, the runner falls back to the original source-backed builder.
- Later-day lite runtime now advances half-hour steps through a compact
  component path instead of constructing and then discarding the full
  DIFFUCO/ENERBIL/HYDROL audit scaffold objects. The process order and
  source-backed kernels are unchanged; the debug scaffold path remains
  available for parity checks and trace diagnosis.
- Development scripts now enable JAX persistent compilation cache under
  `outputs/jax_compilation_cache`. This does not change model semantics, but it
  reduces repeated cold-process validation time after the cache has been
  populated.
- `scripts/dev/run_multiday_modelout_timing.py` accepts `--warmup-days` so long
  timing runs can compile the current kernel set with a short warmup window
  before measuring a longer sequence. This improves development-loop cost; it
  does not change steady-state model arithmetic.
- `scripts/dev/compare_multiday_modelout_to_reference.py` now defaults to the
  lite/compact modelout runner and reads the sparse STOMATE history file once
  per invocation. Use `--mode scaffold` when the full audit scaffold is needed.

Discarded experiment:

- Replacing maintenance temperature/leaf scatter writes with broad `jnp.where`
  masks did not improve warmed runtime and triggered XLA algebraic-simplifier
  warnings, so that edit was reverted.
- Carrying the 1961 point forcing cache directly on
  `Paper1961PreparedDriverContext` avoided a small wrapper lookup, but 14-day
  timing regressed and 30-day timing improved only by noise-scale amounts, so
  it was reverted.
- JIT-wrapping the full daily STOMATE carbon adapter failed because the active
  phenology input still carries string-valued `pheno_model` configuration. A
  future compiled daily-carbon path should first split static configuration
  from array state instead of hiding that issue in the driver.
- Online daily-accumulator folding inside the half-hour loop preserved the
  current 3-day lite/scaffold parity but made 14-day warmed timing worse
  (`~15.7 s` under load), so it was reverted. Batching the daily fold at the
  day boundary remains better for current small-array dispatch behavior.
- Rewriting the HYDROL explicit-snow zero-state active-field check to merge
  device booleans before host synchronization did not improve warmed timing and
  was reverted.
- Adding a fast path around HYDROL `_as_1d/_as_2d/_as_3d/_as_4d` for existing
  JAX arrays did not show stable warmed benefit and was reverted.
- Passing `firstcall=False` into later-day `prescribe_step` preserved local
  tests but made 14-day warmed timing worse, so it was reverted pending a
  separate source-semantics audit of STOMATE SAVE `firstcall` behavior.
- Vectorizing the `prescribe_step` cold-start PFT initialization loop preserved
  unit/parity tests but made 14-day warmed timing substantially worse; the
  original PFT loop/scatter form is faster for the current small paper-case
  arrays.
- Skipping repeated HYDROL explicit-snow zero-state active-field checks via a
  paper-case zero-snow assumption preserved parity but did not improve warmed
  timing, so the strict checked adapter remains in use.
- Replacing the runtime STOMATE daily accumulator loop with a stacked
  `jax.lax.scan` changed bit-exact daily/modelout parity at `~1e-16` and
  triggered an XLA algebraic-simplifier warning, so it was reverted. Daily
  accumulator compilation needs a dedicated bit-exact strategy before it can be
  retained.
- Caching the later-day restart season template in
  `Paper1961PreparedDriverContext` preserved parity but did not show a stable
  warmed timing benefit in the 7-30 day checks, so it was reverted.
- Reusing prepared-context HYDROL static vectors inside the later-day runtime
  static template preserved short parity but did not improve the 30-day warmed
  profile (`44.43 s` versus the current `44.36 s` reference), so it was
  reverted.
- Keeping only daily-fold fields for the 47 intermediate later-day runtime
  entry payloads preserved short parity but made the 30-day warmed run slower
  (`47.63 s`), so it was reverted. The extra Python dict filtering is not the
  right way to attack the payload boundary.
- Bypassing the public `paper_1961_next_step_runtime_result` wrapper and
  calling the compact step helper directly from the later-day runtime preserved
  short parity but made the 30-day warmed profile slower (`47.02 s`), so it
  was reverted. The wrapper is not a meaningful performance boundary.
- Turning off `module_jit` did not complete a 14-day warmed check within the
  3-minute diagnostic timeout, and turning off only `diffuco_local_jit` made a
  7-day warmed run slower (`15.80 s`). The current module JIT settings remain
  the right default for the active single-point paper case.

A/B timings, same command shape as above:

- 7 days warmed: baseline `10.27 s`; after retained changes `7.56 s`.
- 14 days warmed: baseline `17.19 s`; after `vmax_step` vectorization
  `13.29 s`; after STOMATE context cache `12.73 s`; after compact runtime
  `12.15 s`.
- 30 days warmed after STOMATE context cache: `26.03 s`, `0.868 s/day`; after
  compact runtime: `25.00 s`, `0.833 s/day`.
- Persistent cache cold-process check after enabling the cache in dev scripts:
  3-day cold timing went from `96.68 s` on the cache-fill run to `65.96 s` and
  `70.65 s` on subsequent processes in the same environment.
- Short warmup support check:
  `--days 7 --warmup 1 --warmup-days 3 --mode lite --compact-later-days on`
  completed with a `3`-day warmup and a `7`-day measured run; the measured run
  was `8.09 s` in that process. The benefit is avoiding a full-length duplicate
  warmup before long runs, not reducing the measured model slope.
- Modelout-reference script smoke checks after switching the default to
  lite/compact:
  `compare_multiday_modelout_to_reference.py --days 3` completed with
  `closed_modelout_days=3`, and
  `compare_multiday_modelout_to_reference.py --days 1 --mode scaffold`
  completed with `closed_modelout_days=1`.

Semantic checks:

- `tests/unit/test_stomate_carbon_kernels.py`,
  `tests/unit/test_stomate_integration.py`, and
  `tests/unit/test_stomate_daily.py`: `119 passed`.
- `tests/parity/test_driver_1961_domain.py`: `6 passed`.
- Multiday lite/scaffold parity through day 3: `2 passed`.
- 60-day cold-start GPP trace comparison:
  `outputs/day60_gpp_compare_after_context_cache.json`, `ok=true`,
  max abs error `6.817717803642154e-6` under the audited `1e-5`
  whole-driver SWdown tolerance.
- After compact runtime:
  `outputs/day60_gpp_compare_after_compact_runtime.json`, `ok=true`,
  max abs error unchanged at `6.817717803642154e-6`.

Current year-scale estimate:

- A warmed one-year paper-case PFT14 run is roughly `5.1 min` by the compact
  runtime 30-day slope, plus one cold setup/compile cost of about `~100-125 s`
  in this Windows environment.
- 2026-07-04 measured in-memory full-year handoff smoke:
  `outputs/restart_year_1961day365_to_1962day3_smoke.json`

## 2026-07-09 Structural Runtime Work

Retained staging change:

- Added `jax_orchidee.driver.fast_state.DriverFastStateBundle`, a
  pytree-registered fixed-layout representation of
  `DriverPreviousStepStatePacket`. The bundle keeps component names, field
  names, and provenance in a static spec while carrying timestep and dynamic
  field values as pytree leaves. This is a no-runtime-effect staging layer for
  future half-hour/day `lax.scan` work; the current model runner is unchanged.
- Added an opt-in `use_fast_state_loop` path for the compact later-day runtime.
  The fast bundle exposes the same audited read-only packet interface used by
  existing source-backed process functions, so the compact step can read the
  fast state directly. The loop still rebuilds a fast bundle from each produced
  packet and is not expected to speed up the model yet; its purpose is to
  validate the state boundary before replacing Python step orchestration.

Validation:

- Real 3-day lite compact day-end state round-trips through
  `fast_state_from_previous_packet` and `previous_packet_from_fast_state` with
  exact component order, field order, provenance, and numeric values
  (`rtol=0`, `atol=0`).
- A 2-day A/B test compares the existing compact later-day loop with the
  opt-in fast-state loop and matches daily modelout fields plus the final
  previous-step state packet exactly (`rtol=0`, `atol=0`).
- A separate pytree smoke check confirms that the static spec survives
  flatten/unflatten without becoming dynamic leaves.

A/B timing:

- 7-day warmed lite/compact, one measured repeat:
  old compact loop `7.90 s`; opt-in fast-state loop `7.18 s`.
- 30-day warmed lite/compact, two measured repeats:
  old compact loop `27.44 s`, `27.61 s`; opt-in fast-state loop `27.68 s`,
  `28.25 s`.
- Interpretation: the fast-state boundary is valid as a structural staging
  layer, but the current opt-in path does not provide stable runtime benefit.
  Keep it default-off until the compact step reads component arrays without
  rebuilding small dictionaries.

Next structural step:

- Replace repeated full `fields_by_component` materialization with
  component-level fast-state accessors in the compact step. Only after exact
  parity with the existing compact runner should the 48 half-hour steps be
  wrapped in `lax.scan`.

## 2026-07-09 Performance Ceiling Assessment

Current evidence:

- A warmed 7-day cProfile run after retained daily-accumulator optimizations
  spent `8.55 s` cumulative in later-day runtime for 6 later days, or about
  `1.43 s/later-day` under profiling.
- The 288 later-day half-hour runtime steps in that 7-day profile spent
  `4.58 s` cumulative in `paper_1961_next_step_runtime_result`, or about
  `0.76 s/day` before daily STOMATE carbon/fold work.
- The same profile shows large small-operation pressure:
  `72,400` JAX primitive applications, `52,536` `asarray` calls,
  `16,980` indexing rewrites, and `3,583` scatter updates in only 7 model
  days.
- Daily STOMATE carbon is still about `0.29 s/day` under the same profile;
  daily accumulator/fold work is now about `0.12 s/day` after the retained
  scalar-branch and guarded `veget_cov_max` reuse.

Interpretation:

- The remaining bottleneck is structural: repeated Python half-hour
  orchestration, small-array JAX dispatch, and state payload assembly. The
  current single-point paper-case arrays are too small for many separate JAX
  kernels to amortize dispatch overhead.
- More micro-optimizations inside individual formulas are unlikely to move the
  30-day slope by more than a few percent unless they remove a repeated
  scatter/indexing pattern across many calls.
- The realistic next speed step is to compile coarser transitions:
  first the 48 half-hour SECHIBA step sequence for one day, then daily STOMATE
  carbon/fold, then month/year blocks only after day-level exact parity.

Expected upper bounds on this Windows single-point runner:

- If only Python dictionaries/state packet churn are reduced, expect modest
  gains: roughly `5-15%`.
- If 48 half-hour steps are wrapped into a bit-checked day-level compiled
  transition while keeping process formulas equivalent, a plausible target is
  `0.25-0.45 s/day` warmed steady-state for the current single point.
- If daily STOMATE carbon/fold is also compiled without changing reduction
  order beyond an audited tolerance policy, a plausible target is
  `0.15-0.30 s/day`.
- A year-level or month-level scan can reduce Python overhead further, but it
  should not be attempted before day-level parity; otherwise first-divergence
  debugging becomes too expensive.

Validation rule for structural optimization:

- Every retained structural performance change must have exact packet/modelout
  parity against the current compact runner for short windows first. If a
  compiled scan changes floating-point order, it must be treated as a policy
  decision with explicit tolerance and diagnostics, not silently accepted as a
  performance tweak.

Discarded/blocked structural experiments:

- A daily-carbon JIT wrapper that moved `phenology_inputs["pheno_model"]` to a
  static argument still failed because `prescribe_step` uses Python boolean
  branches on `ok_dgvm`.
- Moving the obvious daily-carbon boolean switches (`ok_dgvm`,
  `lpj_gap_const_mort`, `wire_vmax`, `ok_nlim_vmax`) out of input dictionaries
  still failed: `prescribe_step` also branches with `bool(cold_static[pft])`,
  where `cold_static` depends on PFT arrays such as `natural` and `pasture`.
- Interpretation: compiling the whole daily-carbon adapter requires a real
  static/dynamic split of PFT-type masks inside the STOMATE kernels, not a
  shallow JIT wrapper around the current dictionary adapter.

## 2026-07-09 Static Daily-Carbon JIT

Retained changes:

- `prescribe_step`, `constraints_step`, and `phenology_step` now accept
  static host tuples for PFT-type/configuration masks while preserving the
  previous dynamic-array path. This makes their Python branch scheduling
  traceable for the paper-case PFT14 active path without changing ordinary
  explicit execution.
- Added
  `stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit`,
  which splits static PFT dispatch/configuration from dynamic arrays, compiles
  the daily carbon chain, and restores the Python-side
  `DailyCarbonBoundary.requires_trace` diagnostics after the jitted numeric
  result returns.
- The lite multiday driver and timing script now expose
  `use_static_jit_daily_carbon` / `--use-static-jit-daily-carbon`. The default
  remains off until longer-window validation is run.
- `read_annual_co2` now uses a small resolved-path/year cache. This removes
  repeated same-year text reads in half-hour step bundle construction and does
  not change driver values.

Validation:

- STOMATE carbon/integration tests: `207 passed`.
- Paper-case daily-carbon static-JIT result matches the explicit bundle path
  exactly for key daily state/model variables (`rtol=0`, `atol=0`) and
  preserves boundary `requires_trace`.
- 3-day driver lite/compact/static-JIT modelout parity matches the scaffold
  output exactly for GPP, LEAF_M, NPP_model, and AGB_model.
- Driver domain/forcing tests after CO2 cache: `11 passed`.

A/B timing:

- 30-day warmed lite/compact default path after the CO2 cache:
  measured repeats `29.78 s`, `26.74 s`.
- 30-day warmed lite/compact with static daily-carbon JIT:
  measured repeats `21.38 s`, `21.38 s`, or about `0.713 s/day`.

Interpretation:

- This is the first retained structural optimization with a stable material
  slope improvement, roughly `20%` versus the nearby default-path 30-day
  measurement.
- The optimization still leaves the half-hour SECHIBA loop in Python; the next
  large target remains day-level compilation of the 48 half-hour transitions.
  - 1961 365-day source run: `361.57108110000263 s`
  - 1962 3-day restart run in the same process: `6.22090290000051 s`
  - This confirms the current practical year-scale cost is still about
    `~1 s/day` on this Windows ORCJAX setup after the initial compiled kernels
    are available in-process.
- The next large performance boundary is the Python half-hour
  SECHIBA/HYDROL/ENERBIL state transition and the daily STOMATE carbon chain.
  Meaningful improvement beyond the compact wrapper likely requires splitting
  static configuration from dynamic arrays and compiling a coarser state
  transition, not more scalar formula edits.

## 2026-07-08 Mosaic Replan Check

The paper-scale workflow is now treated as 669 independent single-landpoint
cases, not one multi-landpoint routing run. This changes the performance
priority: cold-process setup/compile cost must be amortized across many cases
or reduced before routine 669-case validation is practical.

Latest smoke checks:

- `scripts/dev/run_restart_year_handoff_smoke.py --source-days 3 --restart-days 2`
  completed with `source_ready=true`, `handoff_gap_count=0`, and
  `restart_ready=true`. The 3-day source leg took `87.57 s`; the 2-day restart
  leg in the same process took `5.03 s`.
- `scripts/dev/run_multiday_modelout_timing.py --days 1 --mode lite
  --compact-later-days on --show-days 1` completed with
  `ready_for_requested_days=true`, no missing components, and one closed
  modelout day. Cold elapsed time was `86.78 s`.

Interpretation:

- Semantically, the one-landpoint day/year handoff path is currently runnable
  at short windows with no missing components.
- Engineering-wise, launching one cold Python/JAX process per landpoint would
  be far too slow for the 669-case mosaic. The next runner should keep one
  warmed process alive across a representative landpoint subset and eventually
  across the full mosaic, or compile a coarser static-shape transition shared
  by cases.

## 2026-07-09 Performance Pass

Retained changes:

- Driver forcing solar-angle redistribution now caches the forcing-interval
  mean solar angle separately from each half-hour step's instantaneous solar
  angle. This preserves the `readdim2.f90::forcing_read_interpol` calculation
  order within one interval, but avoids recomputing the same interval mean for
  all 12 split SECHIBA steps.
- Paper-case STOMATE bundle builders accept already parsed `run_def_values`.
  The driver runtime passes the prepared-context dictionary through the
  first-day, later-day, and cold-start bundle paths instead of reopening and
  reparsing the same `used_run.def` during daily STOMATE input assembly.
- DIFFUCO PFT14 local pre-call assembly now treats prepared static kwargs
  lazily. Previously `dict.get(key, expensive_parse(...))` evaluated the
  parse fallback even when the prepared context supplied the value. The active
  fast path no longer reparses PFT14 CO2/LAI/run.def defaults or copies the
  whole run-def dictionary unless the static kwargs are absent.

A/B diagnostics:

- Profiled warmed 7-day lite/compact run, single-pass daily fold on:
  baseline `15.84 s`; after solar interval cache `14.96 s`.
  `_fortran_time_zone` calls dropped from `2496` to `384`, and
  `_paper_1961_step_bundle_from_context` cumulative time dropped from
  `1.045 s` to `0.720 s`.
- Warmed 7-day lite/compact run, single-pass daily fold off:
  after solar interval cache `14.23 s`; after run-def value propagation
  `12.95 s`; after DIFFUCO lazy static parsing/copying `12.71 s` under
  cProfile.
- Unprofiled 30-day lite/compact run, single-pass daily fold off, one short
  3-day warmup and two measured repeats: before the final DIFFUCO lazy-copy
  edit the second repeat was `37.31 s`; after it the second repeat was
  `35.71 s`, or `1.19 s/day`.

Validation:

- `tests/unit/test_driver_domain_multigrid.py` and
  `tests/parity/test_driver_1961_domain.py`: `11 passed`.
- Coupled STOMATE input-builder targeted tests: `5 passed`.
- DIFFUCO unit/bridge tests: `94 passed`.
- Driver lite/compact/multiday parity subset: `6 passed`.
- `compare_multiday_modelout_to_reference.py --days 3 --mode lite
  --compact-later-days on` completed with `closed_modelout_days=3` and no
  missing components; its history comparison remains a sparse/yearly
  diagnostic rather than daily truth.

Current interpretation:

- The retained edits remove redundant Python/static-input work without changing
  process formulas or branch decisions.
- The remaining large boundary is still the Python half-hour state transition
  plus small-array JAX dispatch inside HYDROL, STOMATE daily carbon, and daily
  accumulators. Further large speedups likely require a coarser compiled
  transition or a carefully bit-checked daily-carbon compilation strategy, not
  more run-def parsing cleanup.

## 2026-07-09 Daily Accumulator Pass

Retained changes:

- `stomate_accumulate_daily` now uses a Python boolean branch when `do_slow`
  is a scalar Python/NumPy bool, matching the Fortran scalar `ldmean` branch
  and avoiding a `jnp.where` primitive for every daily-accumulator update.
  Non-scalar/traced `do_slow` values still fall back to the previous JAX
  `where` expression.
- The runtime daily accumulator fold precomputes `veget_cov_max` once per day
  only when all entry payloads have byte/equality-identical `veget_max` and
  `totfrac_nobio`. If any entry differs, it falls back to the per-entry source
  formula. This preserves the `stomate_main` GPP daily increment algebra while
  avoiding repeated normalization on the active fixed-vegetation daily path.
- Later-day/cold-start daily fold setup now avoids eager construction of the
  `t2m_longterm` fallback when the state already carries `t2m_longterm`.

Discarded experiment:

- Replacing the GPP daily bare-soil PFT zeroing scatter with a PFT mask passed
  local tests and improved a 7-day timing, but the 30-day repeat was slightly
  slower (`24.36 s` versus `23.93-24.16 s` nearby baselines), so it was
  reverted.
- Replacing HYDROL explicit-snow active-field checks with NumPy host checks
  preserved local snow/driver tests, but slowed both 7-day and 30-day timings
  (`9.61 s` for 7 days and `28.99 s` for 30 days in the measured repeats), so
  the strict JAX reduction checks remain in place.
- Caching daily STOMATE allocation run.def parameters preserved allocation and
  driver tests, but did not show stable runtime benefit (`29.59 s` for a
  30-day repeat), so it was reverted.
- Replacing daily-input scalar `one_day / dt_sechiba` JAX expressions with a
  Python-float fast path passed daily/integration tests, but did not improve
  30-day timing (`28.05 s`), so it was reverted.

A/B diagnostics:

- Before this pass, the retained 2026-07-09 static-input cleanup measured
  `35.71 s` for a 30-day unprofiled lite/compact run under one 3-day warmup
  and two measured repeats.
- After the scalar `do_slow` branch, 30 days measured `23.93 s`
  (`0.798 s/day`) in the second repeat.
- After the guarded daily `veget_cov_max` reuse, 30 days measured `23.63 s`
  (`0.788 s/day`) in the second repeat.
- A profiled 7-day run after these retained changes shows
  `_stomate_daily_accumulation_fold_runtime_from_entries` at about `0.84 s`
  cumulative, down from about `1.15 s` after the static-input cleanup and
  about `1.59 s` in the earlier single-pass profile.

Validation:

- `tests/unit/test_stomate_daily.py`: `29 passed`.
- `tests/unit/test_stomate_daily.py tests/unit/test_stomate_integration.py`:
  `72 passed`.
- Driver lite/compact/multiday parity subset: `6 passed`.
- `compare_multiday_modelout_to_reference.py --days 3 --mode lite
  --compact-later-days on` completed with `closed_modelout_days=3`, no missing
  components, and the same diagnostic values as the previous performance-pass
  comparison. The history comparison remains sparse/yearly, not daily truth.
- A later final-state smoke timing after reverting the discarded experiments
  completed 30 days with `closed_modelout_days=30`, no missing components, and
  a second repeat of `27.38 s`. This run was slower than the best retained
  `23.63 s` measurement, consistent with the observed Windows/JAX timing
  variability; no additional code was retained from the slower experiments.

## 2026-07-09 Static Daily Carbon / HYDROL Snow Dispatch Pass

Retained changes:

- The performance baseline for later work is the opt-in static-JIT daily
  carbon path (`--use-static-jit-daily-carbon on`). It separates static PFT
  configuration from dynamic arrays and removes non-JAX string diagnostics from
  the compiled result, then restores `DailyCarbonBoundary.requires_trace` after
  the compiled call.
- The HYDROL explicit-snow no-snow adapter now has an opt-in JIT array helper
  used when `module_jit=True`. The helper performs the active-field check and
  constructs the zero snow state arrays in one compiled call. If any snow or
  nobio-snow field is active, it falls back to the original detailed strict
  path so the full explicit-snow branch remains required rather than silently
  approximated.
- The OK_LEAK litter path now uses `littercalc_leak_core_with_controls_jit`
  for the non-cryoturbation path. The soilcarbon core was already compiled;
  this removes the remaining daily littercalc small-kernel chain from the
  profile without changing the source-backed litter/soilcarbon formula order.
- DIFFUCO PFT14 ENERBIL pre-call assembly now inserts the five locally
  computed PFT columns (`gpp`, `gsmean`, `rveget`, `rstruct`, `cimean`) through
  one compiled helper instead of five independent mask/where dispatches.
  Shape checks are still performed before the compiled call.
- HYDROL JIT mode now builds module outputs and THERMOSOIL moisture inputs in
  one compiled post-processing helper. The public non-JIT helpers remain
  unchanged; the compiled helper only combines the same downstream array
  mapping after module diagnostics have already been computed.
- Later-day runtime HYDROL static-template assembly now reuses the prepared
  driver context arrays for `EXT_COEFF_VEGETFRAC`,
  `PERCENT_THROUGHFALL_PFT`, `HYDROL_HUMCSTE`, and `STOMATE_OK_DGVM` instead
  of reparsing those `used_run.def` entries at every day boundary.
- The experimental fast-state wrapper now caches component/field name indexes
  instead of doing repeated linear `tuple.index()` lookups. This is retained as
  a small staging cleanup for future compiled state bundles, but the
  fast-state loop remains default-off because it is still slower than the
  current dictionary-packet runtime path.

Discarded experiment:

- Fusing `hydrol_module_explicit_step` and `hydrol_module_diagnostics` into one
  JIT call passed HYDROL unit tests and the 3-day driver parity subset, but did
  not improve the 30-day static-daily-carbon timing (`21.71 s` second repeat
  versus the nearby `21.38 s` baseline). The fusion was removed because it
  added complexity without stable benefit.
- A stacked `lax.scan` implementation of the runtime daily accumulator reduced
  Python-loop structure, but failed the existing bit-exact daily accumulator
  regression by about `5.5e-17` in `soilhum_daily`. This is scientifically
  negligible, but it would add diagnostic noise to the current strict parity
  workflow, so the experiment was removed and the existing bit-exact runtime
  fold remains active.
- Re-testing `--use-fast-state-loop on` after cached component/field lookup
  still measured slower than the default path (`19.88 s` second repeat for 30
  days versus `18.12 s` nearby default-off timing), so it remains disabled for
  performance runs.

A/B diagnostics:

- Static-JIT daily carbon baseline, 30-day lite/compact run with one 3-day
  warmup and two measured repeats: about `21.38 s` best nearby repeat
  (`0.713 s/day`).
- After the HYDROL no-snow checked-array JIT helper, the same 30-day command
  measured `20.44 s` and `20.18 s` (`0.673 s/day`) across the two repeats.
- After adding the non-cryoturbation littercalc JIT helper, the same command
  measured `19.17 s` and `20.03 s` (`0.668 s/day` second repeat). The 7-day
  profiled run dropped from `8.81 s` to `8.33 s`, and the OK_LEAK/littercalc
  hotspot disappeared from the top profile.
- After fusing the DIFFUCO PFT-column insertions, the same 30-day command
  measured `18.02 s` and `18.94 s` (`0.631 s/day`). The profiled 7-day run
  measured `7.81 s`; `run_pft14_local_enerbil_precall_from_first_step_precall`
  dropped to about `0.512 s` cumulative.
- After fusing HYDROL outputs plus THERMOSOIL moisture post-processing, the
  same 30-day command measured `17.97 s` and `18.72 s` (`0.624 s/day`). In the
  7-day profile, `run_hydrol_first_step_module_from_precall` dropped from
  about `1.34 s` to about `1.17 s` cumulative, though overall 7-day timing is
  still noisy on Windows.
- After replacing later-day run.def reparsing with prepared-context arrays,
  the same 30-day command measured `19.08 s` and `18.12 s` (`0.604 s/day`).
  The profiled 7-day run remained noisy (`8.44 s`), so this change is retained
  as a low-risk static-input cleanup rather than treated as a large structural
  speedup.
- A profiled 7-day run after reverting the discarded HYDROL fusion but before
  retaining the no-snow helper had `run_hydrol_first_step_module_from_precall`
  at `2.153 s` cumulative and `hydrol_explicit_snow_zero_state` at `0.845 s`.
- The profiled 7-day run after the retained helper had
  `run_hydrol_first_step_module_from_precall` at `1.322 s`; the separate
  `hydrol_explicit_snow_zero_state` hotspot disappeared from the top profile.

Validation:

- `tests/unit/test_hydrol_cold_start.py`: `35 passed`.
- `tests/unit/test_hydrol_module_step.py`: `23 passed` before and after the
  discarded HYDROL fusion cleanup.
- OK_LEAK targeted STOMATE integration tests: `3 passed`.
- `tests/unit/test_diffuco.py`: `79 passed`.
- HYDROL module/cold-start tests after post-processing fusion: `58 passed`.
- `tests/unit/test_stomate_daily.py`: `29 passed` after removing the
  non-bit-exact stacked scan experiment.
- `tests/parity/test_driver_1961_orchestration.py::test_multiday_modelout_lite_run_matches_scaffold_outputs_through_third_day`:
  `1 passed`.

Current interpretation:

- The retained HYDROL change is still a local dispatch reduction, not the
  day-level compiled transition itself.
- The next structural target remains a compiled 48-half-hour SECHIBA day
  transition, but this requires moving dynamic state and per-step forcing into
  a stable array/pytree contract before replacing the Python loop with
  `lax.scan`.

## 2026-07-10 Validation Default Fast Path

Retained changes:

- `scripts/dev/run_multiyear_modelout_lite.py` and
  `scripts/dev/compare_multiday_modelout_to_reference.py` now default
  `--single-pass-daily-fold off` and `--use-static-jit-daily-carbon on`.
  The non-single-pass runtime path is faster because it uses the existing
  runtime-only daily accumulator plus batched maintenance respiration; the
  single-pass path still creates per-entry accumulator result objects.
- The same scripts expose `--prebuild-day-payloads`, defaulting on, so A/B
  timing can disable the later-day prebuilt step/payload path without changing
  other validation settings.
- `paper_1961_driver_restart_year_multiday_modelout_lite_run` and
  `paper_1961_driver_cold_start_multiday_modelout_lite_run` accept
  `use_static_jit_daily_carbon`, so annual/restart-handoff validation can use
  the same retained daily-carbon fast path as same-year lite runs.
- Later-day compact runtime now prebuilds one day's 48 source-backed
  `DriverStepBundle` objects and base `IntersurfFirstStepPayload` objects when
  no trace override is active. Each half-hour step still recomputes wind from
  the current dynamic driver `z0`; only state-independent bundle/payload
  packaging moved to the day boundary.
- The fast-state loop remains default-off. Retesting showed it was still slower
  than the dictionary packet path for the current Python loop.

A/B diagnostics from this pass:

- Baseline 7-day lite/compact run, `single_pass_daily_fold on`,
  `use_static_jit_daily_carbon off`, one 1-day warmup and two measured repeats:
  `12.10 s`, `10.90 s` (`1.56 s/day` second repeat).
- Static daily-carbon fast path, same command shape with
  `use_static_jit_daily_carbon on`: `5.45 s`, `5.38 s` (`0.77 s/day` second
  repeat).
- 14-day static daily-carbon fast path, one 1-day warmup and one measured
  repeat: `10.01 s` (`0.71 s/day`).
- Prebuilt day payload A/B with fast static daily-carbon and compact later days:
  7-day off/on second repeats were `5.14 s` / `5.06 s`; 14-day off/on second
  repeats were `9.80 s` / `9.14 s`.
- Single-pass daily fold A/B after prebuilt payloads showed the runtime
  accumulator path is faster at year-scale-relevant windows: 30-day
  `single_pass_daily_fold off` measured `17.15 s` (`0.572 s/day`), while
  `single_pass_daily_fold on` measured `19.94 s` (`0.665 s/day`).
- A follow-up experiment that shared the unchanged slowproc component mapping
  between half-hour state packets preserved short parity but regressed the
  30-day timing sample to `18.29 s` and introduced unnecessary aliasing risk,
  so it was reverted.
- Retained two low-complexity compact-step cleanups after parity checks:
  removed one redundant `dict(...)` copy of the per-day DIFFUCO slowproc
  derivvar payload, and changed DIFFUCO restart-payload assembly to check the
  27 required restart fields directly instead of scanning all fields from
  DIFFUCO/HYDROL/SLOWPROC state components. Local timing samples were noisy, so
  these are recorded as Python orchestration cleanup rather than a claimed
  major speedup.
- DIFFUCO and ENERBIL pre-call assemblers now avoid copying input mappings
  unless legacy alias fields actually need to be materialized. Unit coverage
  (`tests/unit/test_diffuco.py`, `tests/unit/test_enerbil.py`) and 3-day driver
  parity passed. A 30-day timing sample remained in the noisy `~20 s` range,
  so this is retained as an inner-loop structure cleanup, not counted as a
  demonstrated breakthrough.
- First warmup/compile cost remains large (`56-61 s` in these samples); this
  pass improves warmed validation throughput, not initial compilation latency.

Validation:

- `tests/parity/test_driver_1961_orchestration.py::test_static_jit_daily_carbon_matches_explicit_paper_case_bundles`:
  `1 passed`.
- `tests/parity/test_driver_1961_orchestration.py::test_multiday_lite_fast_state_loop_matches_compact_outputs_through_second_day`:
  `1 passed`.
- `tests/parity/test_driver_1961_orchestration.py::test_multiday_lite_prebuilt_day_payloads_match_step_local_payloads_through_second_day`:
  `1 passed`.
- `tests/parity/test_driver_1961_orchestration.py::test_multiday_modelout_lite_run_matches_scaffold_outputs_through_third_day`:
  `1 passed` after enabling the prebuilt payload default.
- Earlier non-legacy semantic smoke before the single-pass default flip:
  `scripts/dev/compare_multiday_modelout_to_reference.py --landpoint-id 003.0-077.0 --year 1961 --days 1 --mode lite --compact-later-days on`
  completed with `ready_for_requested_days=true`, `closed_modelout_days=1`,
  `missing_components=[]`, `single_pass_daily_fold=on`, and
  `use_static_jit_daily_carbon=on`.

Current interpretation:

- This is a practical default-path speedup for local semantic validation.
- It does not remove the main remaining structural bottleneck: Python still
  drives 48 half-hour SECHIBA steps per day. The next large speedup requires a
  day-level compiled transition over stable state/forcing pytrees.

## 2026-07-11 Compiled SECHIBA Day Transition

Retained opt-in implementation:

- Added a real coarse-grained later-day transition. The first half-hour uses
  the strict compact runner to normalize the incoming restart/day-end packet;
  the remaining 47 steps use one compiled `lax.scan` over dynamic forcing and
  a stable fast-state pytree.
- The scan preserves source order across DIFFUCO, ENERBIL, HYDROL, CONDVEG,
  THERMOSOIL, and STOMATE-entry export. HYDROL uses the full explicit-snow
  process under tracing, so snowfall remains reachable instead of being
  replaced by a paper-point no-snow assumption.
- CWRR coefficient tables are now explicitly bound at the day-static boundary
  rather than relying on an inner-loop cache side effect.
- Dynamic forcing is stacked once. Compiled STOMATE-entry arrays are copied to
  host once per field and then split with NumPy views; this removed tens of
  thousands of tiny JAX slice dispatches per 30-day run.
- The path is exposed by `--compiled-sechiba-day on` in timing, reference
  comparison, and multiyear runners. It remains default-off until the next
  multi-point/multi-year acceptance stage.

Performance evidence:

- Previous strict baseline, same 30-day lite/compact/static-carbon command:
  `15.66-16.92 s`, representative `16.47 s` (`0.549 s/day`).
- Initial compiled scan with JAX slicing of 47x44 entry fields:
  `10.20-10.26 s` (`~0.340 s/day`).
- Final compiled scan with one host transfer per stacked entry field:
  `7.83-8.11 s` (`~0.261 s/day`), approximately `2.1x` the strict throughput.
- The isolated full five-module half-hour transition executes in about
  `1.2-1.4 ms` warm; the isolated 48-step scan executes in about `25.5 ms`
  warm. End-to-end day cost is now dominated by daily STOMATE and payload
  preparation rather than half-hour process dispatch.

Numerical and restart evidence:

- `outputs/performance/compiled_sechiba_day_3day_parity_20260711.json` compares
  every daily modelout field and every final-state numeric leaf. It passes
  `rtol=1e-10, atol=1e-8`; modelout maximum absolute drift is about `2e-15`.
- The largest absolute state drift is `3.1665e-7` in `pcapa_en`, with relative
  drift `2.1405e-13`. This is XLA fusion/reassociation roundoff, not a missing
  process or changed branch.
- `outputs/performance/compiled_sechiba_day_restart_handoff_smoke_20260711.json`
  closes a `1961:3 day -> 1962:2 day` handoff with zero missing state fields.
- A dedicated parity test checks compiled versus strict modelout, final state,
  and the year-handoff contract.

Discarded experiments in this pass:

- A source-proven day-level zero-snow shortcut showed only about `0.8%`
  timing difference (`16.34 s` versus `16.47 s` for 30 days), within Windows
  run variability, and was removed.
- Re-enabling the NumPy daily accumulator after compiled entry unpacking was
  slower (`~0.308 s/day` versus `~0.261 s/day`) and remains off.

Next gate:

- Do not add another structural optimization before acceptance. Run the
  representative multi-point/multi-year matrix in strict and compiled modes,
  compare annual modelout and restart state under the documented tolerance,
  then decide whether compiled mode becomes the production default.

Discarded structural experiment:

- Reintroduced a scoped stacked daily-accumulator fast path for later-day
  runtime only, including active PFT14 peat daily fields. The first attempt
  exposed a real vectorization bug: the non-stacked `gpp_daily` helper zeros
  PFT1 with `at[:, 0]`, which is not valid after adding a leading step axis.
  After fixing the stacked form to zero `[..., 0]`, the fast path matched
  3-day modelout to roundoff (`~4.4e-16`) and final state to about `5.7e-14`.
  It was still substantially slower: 7-day timing measured about `8.77 s`
  (`1.25 s/day`) and 14-day timing about `15.19 s` (`1.08 s/day`) versus the
  retained default fast path near `0.7-0.8 s/day`. The code was reverted.
  This confirms the next useful structural target is not the already-small
  daily accumulator fold, but the full 48-step SECHIBA transition and its
  repeated small JAX kernel dispatch.

## 2026-07-10 Day-Transition Boundary Follow-Up

Retained changes:

- The later-day OK_LEAK pre-step boundary reuses the already prebuilt
  day-start `IntersurfFirstStepPayload` when available instead of rebuilding
  the same state-independent payload at the day boundary.
- `scripts/dev/run_multiday_modelout_timing.py` exposes opt-in
  `--use-compiled-accumulator` and `--use-numpy-accumulator` flags for
  controlled experiments. Both remain default-off; strict validation defaults
  are unchanged.

Discarded/default-off experiments:

- A cached `DriverRuntimeStepStateView` staging layer for the compact
  half-hour step preserved parity, but did not show a useful timing benefit
  and added inner-loop object construction. It was removed from the default
  runtime path; typed state work should be introduced as a real array/pytree
  boundary, not as another mapping wrapper.
- Preconverting prebuilt intersurf payloads to JAX arrays made the 7-day
  warmed/profiled run slower (`9.98 s` versus a nearby strict default sample
  around `8-9 s`) and increased `asarray`/primitive pressure, so it was
  reverted.
- Caching derived `totfrac_nobio`/`tot_bare_soil` on the per-step state view
  preserved short parity but slowed the 7-day sample (`9.24 s`), so it was
  reverted.
- A compiled static daily accumulator matched only to roundoff
  (`~4.4e-16` in GPP), triggered an XLA algebraic-simplifier warning in the
  parity run, and was slower in the A/B timing (`9.01 s` for 7 days), so it
  remains opt-in and default-off.
- A pure NumPy runtime accumulator improved one 7-day sample (`7.61 s`) but
  changed GPP by `~4.4e-16`; a hybrid that kept GPP in the original JAX order
  still changed NPP at `~4.4e-16` and was slower. The pure NumPy path remains
  opt-in only for possible tolerance-based long-run screening, not strict
  semantic validation.

Validation:

- Default strict path:
  `tests/parity/test_driver_1961_orchestration.py::test_multiday_modelout_lite_run_matches_scaffold_outputs_through_third_day`
  and
  `test_multiday_lite_prebuilt_day_payloads_match_step_local_payloads_through_second_day`:
  `2 passed`.

Current interpretation:

- The daily accumulator is not the next productive strict target. The
  remaining high-value work is still the 48 half-hour SECHIBA transition:
  replacing Python orchestration and repeated small JAX kernels with a typed
  dynamic state/forcing boundary that can be compiled without changing
  floating-point order silently.

## 2026-07-10 Strict Runtime Retained Pass

Retained changes:

- HYDROL now uses a fused strict JIT boundary for
  `hydrol_module_explicit_step`, `hydrol_module_diagnostics`, module outputs,
  and HYDROL-to-THERMOSOIL moisture arrays. The JIT returns only JAX array
  leaves; Python restores the provenance-bearing `NamedTuple` wrappers outside
  the compiled function. This keeps the same source-order HYDROL formulas and
  avoids returning strings through JAX.
- The runtime-only STOMATE daily accumulator fold now applies the same
  `field_out = field_in + increment * dt_sechiba` and final
  `field_out / dt_stomate` formula inline after input validation, instead of
  calling `stomate_accumulate_daily` thousands of times. The public/debug
  helper remains unchanged.
- DIFFUCO local ENERBIL payloads now pass through the already source-backed
  `rau` field, and ENERBIL pre-call assembly reuses upstream `rau`/`swnet`
  when present before falling back to the exact local source formulas. This
  removes duplicate same-step derivations without changing the source formula.

Discarded experiment:

- Fusing the no-snow checked-array helper into the HYDROL fused boundary
  preserved tests but did not improve 30-day timing (`~14.53 s`, essentially
  unchanged from the immediately preceding retained HYDROL fusion sample), so
  it was reverted.

Validation:

- `tests/unit/test_hydrol_module_step.py`: `23 passed`.
- `tests/unit/test_stomate_daily.py`: `29 passed`.
- `tests/unit/test_diffuco.py tests/unit/test_enerbil.py`: `121 passed`.
- `tests/parity/test_driver_1961_orchestration.py::test_multiday_modelout_lite_run_matches_scaffold_outputs_through_third_day`
  and
  `test_multiday_lite_prebuilt_day_payloads_match_step_local_payloads_through_second_day`:
  `2 passed`.

Timing, unprofiled 30-day lite compact run with one 3-day warmup,
`--use-static-jit-daily-carbon on`, strict accumulator defaults:

- Before this retained pass: `17.14 s`, `17.06 s` (`0.569 s/day`) in the
  first same-session baseline.
- After HYDROL fused closure: best measured repeat `15.02 s`
  (`0.501 s/day`) after reverting the no-benefit snow fusion.
- After runtime daily accumulator inline formula: best measured repeat
  `13.94 s` (`0.465 s/day`).
- After DIFFUCO-to-ENERBIL `rau`/`swnet` reuse: best measured repeat
  `13.67 s` (`0.456 s/day`).

Current interpretation:

- This pass keeps strict bit-exact validation defaults and improves warmed
  local throughput by about 20% versus the same-session baseline.
- The next likely structural target is still the half-hour SECHIBA transition
  boundary; remaining profile cost is dominated by Python orchestration,
  HYDROL wrapper/snow checks, ENERBIL/DIFFUCO assembly, and many small JAX
  dispatches.

## 2026-07-11 Post-Performance Five-Year Gate

Validation:

- Current code was rerun for 1961-1965 with the retained cold-start paper
  reference-mode gate:
  `scripts/dev/run_multiyear_modelout_lite.py --start-year 1961 --years 5
  --days-per-year 365 --initial-state cold-start --run-def
  outputs/reference_mode/used_run.def --single-pass-daily-fold off
  --use-static-jit-daily-carbon on --prebuild-day-payloads on --module-jit on
  --diffuco-local-jit on`.
- Output:
  `outputs/reference_mode/multiyear_1961_1965_after_perf_coldstart_20260711.json`.
- All five years completed 365 modelout days with zero year-handoff gaps after
  1961. Maximum annual reference absolute errors by year were:
  1961 `1.75e-7`, 1962 `6.81e-7`, 1963 `5.77e-7`, 1964 `2.29e-7`, and
  1965 `5.36e-7` across AGB/BGB/GPP/NPP.

Interpretation:

- The current retained performance changes did not break the established
  1961-1965 cold-start reference-mode gate.
- A same-day exploratory run using `initial_state=restart-backed` and the
  materialized selected-landpoint `used_run.def` produced large annual deltas;
  that result is a validation configuration mismatch relative to the retained
  paper reference gate, not evidence of a source-semantic regression.
- `scripts/dev/run_multiyear_modelout_lite.py` now records the first-year
  initial-state mode in its output description and documents that
  `outputs/reference_mode/used_run.def` is the retained cold-start
  reference-mode run definition.

## 2026-07-11 Cross-Landpoint Compiled SECHIBA Reuse

The opt-in compiled later-day SECHIBA transition now receives landpoint
payload arrays, all non-structural HYDROL template/table arrays, and PFT14
DIFFUCO continuous parameters dynamically. The cache key no longer contains
the prepared-context object identity. It retains shapes, process dimensions,
true discrete process switches, PFT masks, and mineral-table index bounds.

Verification command:

`scripts/dev/verify_compiled_sechiba_landpoint_reuse.py --days 3`

Verification artifact:

`outputs/performance/compiled_sechiba_landpoint_reuse.json`

Results for `001.0-071.0` followed by `003.0-077.0` in one Python process:

- Both landpoints passed compiled-versus-strict comparison for every modelout
  field and final-state leaf. Maximum relative errors were `9.64e-13` and
  `9.81e-13`; the largest absolute differences were THERMOSOIL `pcapa_en`
  values of `2.28e-7` and `1.43e-7`.
- The first point created two executable variants for the Day-2 versus later
  day state signatures. The second point added zero executable variants and
  used the same jitted scan callable. Repeating either point also added zero.
- First-point compiled startup was `99.39 s`; second-point first use was
  `10.84 s`; warmed three-day repeats were `1.75 s` and `1.67 s`. The second
  point therefore avoided the expensive SECHIBA scan compilation, while its
  initial context/restart/forcing preparation remained point-specific.
- A subsequent 30-day timing retained the previous throughput class:
  `8.61 / 7.88 / 8.38 s`, with best throughput `0.263 s/day`.

Scope: this proves executable reuse for the two paper landpoints, including
their differing `ARJV` value and point-specific forcing/restart/soil arrays.
It does not claim that every continuous configuration parameter outside this
compiled boundary is already suitable for no-recompile parameter sweeps.

## 2026-07-13 Final-Semantics Performance Gate

All measurements in this section use the Stage 3-5 closed semantic version,
the compact production runner, one `001.0-071.0` landpoint, module/local JIT,
static daily carbon, and prebuilt day payloads. A three-day pass warms the
process before timed repetitions.

Baseline timing:

- Strict 30-day repeats: `18.28`, `18.42`, and `17.94 s`; median
  `0.609 s/day`.
- Compiled SECHIBA 30-day repeats: `7.90`, `7.20`, and `7.17 s`; median
  `0.240 s/day`, or `2.54x` strict throughput.
- Strict 365-day run: `246.24 s` (`0.675 s/day`).
- Compiled SECHIBA 365-day run: `111.52 s` (`0.306 s/day`) when measured in a
  separate process after a short warmup.
- The compiled scan warmup cost was `120-155 s`, versus about `66 s` for the
  strict small-kernel path. This is a one-time executable cost, not per-year
  runtime.

Profile evidence:

- `outputs/performance/final_semantic_compiled_30day_20260713.prof` records a
  cProfile-instrumented 30-day compiled run. cProfile overhead makes its wall
  time unsuitable as a throughput benchmark, but its cumulative hotspots are
  useful.
- The daily STOMATE fold accounts for `5.62 s`, including `4.13 s` in the
  48-entry accumulator. The compiled half-hour transition accounts for
  `4.00 s`, day input/payload preparation for `2.02 s`, and cached forcing
  extraction for `1.38 s`. These cumulative times overlap and must not be
  summed as independent shares.
- No repeated NetCDF or annual output IO appears among the runtime hotspots;
  annual JSON is written after model execution. The main remaining cost is
  Python daily orchestration plus small JAX/NumPy conversions around the
  compiled SECHIBA scan and STOMATE day boundary.

Retained fast configuration:

- The multiyear production runner defaults to compiled SECHIBA. It retains an
  explicit `off` switch for strict A/B and diagnosis; low-level orchestration
  APIs keep strict defaults.
- The source-order NumPy daily accumulator remains available only as a
  default-off experiment. The three-point annual matrix rejected it because
  `319.0-057.0` reached a DOC state absolute difference of `1.345e-8`, above
  the fixed `1e-8` fast-path gate. The tolerance was not enlarged.
- The accepted three-point annual/restart gate is
  `outputs/performance/compiled_fast_path_acceptance/final_3point_annual_compiled.json`.
  Compiled annual runtimes were `79.33`, `94.16`, and `91.37 s`, versus
  strict `237.84`, `249.40`, and `268.63 s` (`2.65-3.00x`).
- All 12 annual history fields, all four derived modelout values, complete
  year-end state, direct 1962 Day1-Day2, and independent strict/compiled
  three-file split restarts passed. Each mode's NetCDF roundtrip was exact.

Cross-landpoint executable stabilization:

- JAX cache-miss diagnostics showed that variable-length
  `soilclass_sub_index`/`soilclass_sub_area` Driver metadata changed the scan
  input from `[1,0]` to `[1,4]` and forced recompilation. Neither field has a
  scientific consumer inside the half-hour transition; the fixed-shape
  `soilclass` field carries the process input.
- Those metadata fields remain in the full Driver payload and strict path but
  are excluded from the compiled scan dynamic pytree. The before/after assets
  are `final_semantic_landpoint_reuse_diagnostic_20260713.json` and
  `final_semantic_landpoint_reuse_after_metadata_fix_20260713.json`.
- After the change, `001.0-071.0`, `069.0-119.0`, and `319.0-057.0` use one
  scan callable. The first point establishes the fixed lifecycle executable
  variants; later points add zero variants, repeats add zero, and all three
  points retain strict numerical parity. Evidence is
  `outputs/performance/compiled_fast_path_acceptance/reuse_3points_current.json`.

Remaining performance work:

- The next structural target is a compiled multi-day/month block that includes
  the STOMATE day boundary and stable day forcing/state bundles. It should not
  be attempted as a monolithic annual graph before profiling compile size and
  restart checkpoints.
- A new restart directory still incurs substantial non-scan JIT/context setup
  on first use. Continuous in-memory year handoff is fast; filesystem split
  restart startup should be optimized separately from scientific runtime.

## 2026-07-13 Lane P1 Daily-Fold Experiments

The production-safe configuration remains unchanged: compiled SECHIBA day is
on and the NumPy daily accumulator is off. All candidates below preserved the
48-entry source order and used the fixed `rtol=1e-10`, `atol=1e-8`, exact
discrete/schema policy.

Performance candidates, 30-day compiled-SECHIBA runs after a three-day warmup:

- Strict accumulator baseline: `7.10`, `6.63`, `6.52 s`.
- Existing all-JAX compiled daily accumulator: `6.84`, `6.72`, `6.44 s`.
  The tail improvement was about 1.3% and not stable across repeats, so it was
  not retained or promoted.
- Host-stacking all daily fields before that kernel regressed to `8.72`,
  `7.75`, `7.84 s`; host copies cost more than the removed small dispatches.
- Host accumulation while keeping all 48 GPP calculations and additions on
  the strict JAX path regressed to `7.98`, `7.37`, `7.80 s`; the retained GPP
  dispatch chain consumed the host-accumulator benefit.
- Computing the 48 independent `gpp_d` values as one JAX bundle before the
  source-order NumPy recurrence measured `6.01`, `5.68`, `5.69 s`, versus
  `5.91`, `5.61`, `5.55 s` for the existing all-NumPy experiment. It retained
  most of the NumPy speed but did not improve numerical behavior.

Difference diagnosis and annual gate:

- The bundled-GPP 30-day full-state/modelout report was exactly identical to
  the prior all-NumPy report, including `DOC` absolute drift
  `3.8025138593411612e-15`. This rules out the independent
  `gpp / veget_cov_max` calculation as the source of the annual rejection;
  the remaining difference comes from the daily source-order recurrence or
  another host-accumulated temperature/moisture driver chain.
- The required `319.0-057.0` annual gate failed with the same year-end `DOC`
  absolute difference as the prior NumPy experiment:
  `1.3449738934859524e-8`. Annual outputs passed, but full year-end state and
  both direct and split-restart cross-year comparisons failed the strict gate.
- Evidence is
  `outputs/performance/lane_p1_batched_gpp_30day_001.json` and
  `outputs/performance/compiled_fast_path_acceptance/lane_p1_batched_gpp_319_annual.json`.
  The experimental code was withdrawn after the failed gate.

## 2026-07-13 Lane P2 Multi-Day Block Feasibility

Profile and baseline:

- The accepted configuration remains compiled SECHIBA day on, static daily
  carbon on, prebuilt day payloads on, and NumPy accumulator off.
- Re-reading `final_semantic_compiled_30day_20260713.prof` confirms that 29
  later-day calls account for `16.19 s` cumulative under cProfile. The daily
  STOMATE fold is `5.47 s`, day transition-input preparation is `2.02 s`, the
  OK_LEAK boundary is about `1.15 s`, and the compiled half-hour transition is
  `4.00 s`. These cumulative regions overlap.
- A fresh unprofiled 30-day run after a 3-day warmup measured `10.316`,
  `9.772`, and `9.546 s`. The warmup itself took `114.89 s`, so this sample is
  slower than the accepted `7.17-7.90 s` gate and reinforces the requirement
  for repeated A/B measurements in one process.

Minimum viable block design:

- Keep Day 1 and new-executable/restart firstcall setup outside the block.
- Carry a fixed-layout `DriverFastStateBundle` and stack forcing as
  `[block_days, 48, ...]` leaves. A pure compiled day transition must derive
  the state-dependent HYDROL day inputs, run the existing 48-step SECHIBA
  scan, fold the 48 STOMATE entries, run the existing static daily-carbon and
  OK_LEAK/season kernels, and write the complete day-end state.
- Only then wrap that pure transition in an outer `lax.scan`, initially with a
  small fixed block such as 8 days. Month/year drivers should checkpoint and
  materialize state at block boundaries; a monolithic annual graph is not an
  acceptable first implementation.

Feasibility probe and decision:

- A two-day JIT-lowering probe used the real Day 1 state and attempted to
  trace the existing Day 2-Day 3 runtime transition as one block. The first
  failure was a host NumPy conversion used only to inspect `lai.shape`.
- After temporarily replacing that inspection with the equivalent static
  array shape, lowering reached the next boundary and failed in
  `hydrol_static_precall_template_from_slowproc`: the daily HYDROL template
  converts model-produced `veget_max`, `veget`, `frac_nobio`, and `soiltile`
  to NumPy. This boundary is state-dependent and cannot be prebuilt as forcing.
- Continuing would also require a trace-safe replacement for the daily
  accumulator's host equality test, typed STOMATE bundle/OK_LEAK inputs, and
  dynamic day/season/end-of-year control. That is a new compiled scientific
  transition API across several ownership boundaries, not a minimal wrapper
  around the accepted day scan.
- The temporary shape edit was reverted. No runtime or default change was
  retained. Because no candidate implementation survived the feasibility
  gate, the 30-day strict state/modelout A/B, annual representative-point, and
  cross-year restart acceptance gates were deliberately not run. Running
  those expensive gates without a candidate would provide no promotion
  evidence.

Evidence:

- `outputs/performance/lane_p2_multiday_block_feasibility_20260713.json`
