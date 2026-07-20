# Performance Candidate Fast Paths

This ledger records performance optimizations that were removed or kept
default-off because they changed floating-point bit patterns or did not yet
meet the current strict parity/debug requirements.

Current policy:

- Strict/debug mode remains the default for semantic closure and first-error
  attribution.
- Fast-production mode may later enable numerically equivalent but non-bit-exact
  paths after source semantics are closed and longer tolerance gates are in
  place.
- A fast candidate must not hide missing Fortran processes. It can only reorder
  or batch already source-backed algebra.

## Candidates

### STOMATE Daily Accumulator Stacked Scan

- Status: removed.
- Area: `jax_orchidee/stomate/daily.py`, runtime daily accumulator fold.
- Idea: stack 48 half-hour STOMATE entry payloads and advance daily
  accumulators with a compiled `lax.scan`.
- Observed effect: removes Python-loop structure from daily accumulator fold.
- Blocking issue: existing bit-exact regression failed by about `5.5e-17` in
  `soilhum_daily`.
- Scientific interpretation: negligible floating-point order difference.
- Current reason to keep off: it adds noise to strict semantic attribution.
- Restore condition:
  - strict source path remains available;
  - fast-production mode has tolerance-based annual/multi-point gates;
  - daily accumulator A/B report records max abs/relative drift for all daily
    fields and downstream modelout variables.

### STOMATE Daily Carbon Scan / Compiled Reordering

- Status: removed/default-off depending on variant.
- Area: STOMATE daily carbon chain.
- Idea: compile larger pieces of daily carbon process ordering or replace
  explicit host loops with scans.
- Observed effect: speed direction was useful in earlier experiments.
- Blocking issue: variants introduced about `1e-16` bit-level differences or
  XLA warnings.
- Scientific interpretation: likely acceptable after semantic closure, but not
  during first-error localization.
- Restore condition:
  - compare daily carbon internal fields and final `modelout` with documented
    tolerances;
  - run multi-year restart/handoff checks in both strict and fast modes;
  - retain strict mode for trace-backed diagnosis.

### Fast State Loop

- Status: available but default-off.
- Area: `jax_orchidee/driver/fast_state.py` and driver later-day loop.
- Idea: use a fixed-layout pytree state bundle instead of dictionary packets
  between half-hour steps.
- Current retained cleanup: cached component/field name indexes.
- Latest timing: still slower than the default packet path (`19.88 s` for 30
  days versus nearby default-off `18.12 s`).
- Current reason to keep off: structure is useful for future compiled state
  bundles, but not a performance win by itself.
- Restore condition:
  - use it as the dynamic state carrier for a coarser compiled transition;
  - prove parity against strict packet mode.

### HYDROL No-Snow Fusion Into Module Closure

- Status: removed.
- Area: `jax_orchidee/sechiba/hydrol.py`.
- Idea: include the no-snow checked-array helper in the fused HYDROL module
  closure to avoid one more small JIT dispatch.
- Validation: HYDROL unit tests and 3-day driver parity passed.
- Timing: no stable improvement (`~14.53 s` for 30 days, essentially unchanged
  from the immediately preceding retained HYDROL module-closure fusion).
- Current reason to keep off: added branch complexity without measured benefit.
- Restore condition: only revisit if snow/no-snow checks dominate after larger
  half-hour transition work.

## Retained Fast Paths

These are not candidates; they are active because they preserved current
strict checks and showed stable or defensible benefit:

- Static-JIT daily carbon path, opt-in via
  `--use-static-jit-daily-carbon on`.
- HYDROL no-snow checked-array JIT helper.
- Non-cryoturbation `littercalc_leak_core_with_controls_jit`.
- DIFFUCO PFT14 five-column insertion fusion.
- HYDROL output plus THERMOSOIL moisture post-processing fusion.
- HYDROL module/diagnostics/output/moisture fused strict JIT closure.
- Later-day HYDROL static-template use of prepared-context arrays.

## Compiled SECHIBA Day Transition

- Status: accepted and enabled by default in the production multiyear and
  paper-landpoint acceptance runners; strict/debug remains explicit.
- Switch: `use_compiled_sechiba_day=True` or
  `--compiled-sechiba-day on` in the development runners.
- Boundary: the first half-hour remains on the strict compact path to
  normalize the restart/day-end packet; the remaining 47 half-hours execute
  DIFFUCO -> ENERBIL -> HYDROL explicit snow -> CONDVEG -> THERMOSOIL in one
  `jax.lax.scan`.
- Reachability: the compiled HYDROL path executes the complete explicit-snow
  process rather than assuming the paper point is snow-free. Unsupported or
  missing-state fallback remains on the strict path.
- 30-day warmed A/B: strict `~16.47 s` (`0.549 s/day`); compiled scan after
  host-side stacked-entry unpacking `7.83-8.11 s` (`~0.261 s/day`).
- 3-day semantic A/B: all modelout fields pass `rtol=1e-10, atol=1e-8`;
  modelout maximum absolute difference is about `2.0e-15`. The largest state
  absolute difference is `3.17e-7` in `pcapa_en`, whose magnitude is about
  `1.5e6` and relative difference is `2.14e-13`.
- Restart smoke: a `1961:3 day -> 1962:2 day` run completed with zero handoff
  gaps.
- Three-point 365-day acceptance passes for `001.0-071.0`, `069.0-119.0`, and
  `319.0-057.0`, including annual output, complete state, cross-year, and
  split-restart comparisons. Compiled annual timings are `79.33-94.16 s`, or
  `2.65-3.00x` strict. See
  `outputs/performance/compiled_fast_path_acceptance/final_3point_annual_compiled.json`.
- Cross-landpoint structural acceptance uses one scan callable; later points
  add no executable variants. See
  `outputs/performance/compiled_fast_path_acceptance/reuse_3points_current.json`.
- The NumPy accumulator companion remains off because its three-point annual
  matrix exceeded the fixed fast-path gate at `319.0-057.0`; see
  `outputs/performance/compiled_fast_path_acceptance/final_3point_annual_numpy.json`.
