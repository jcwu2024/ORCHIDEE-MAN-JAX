# Thermal source and same-input recurrence diagnostic

The compact-tape replay at `d31e93c` completed in 615.589 seconds and is not
admitted. Window 5's score discrepancy is -7.37018197e-6. At displacement
1e-5 its production/observation directional errors are 1.258%/1.828%, above
the existing 1% gate. Window 1 passes; physical failures are zero. No weights
were retained and the production gradient default is unchanged.

The first recorded compact/verbose state difference includes Day-1 snowheat
973.072019405356 versus 972.6645657789619 in layer 0. Both HYDROL and DIFFUCO
mirrors carry it. By Day 4, layer-1 snowheat differs by approximately 211106.
These are differences between evaluation paths, not errors against Teacher.

## Source evidence and question

`constrained_daily_replay.py::THERMAL_FIELDS` assigns snowheat a 1e9 daily
tendency bound and snowtemp a 100 K bound. `NeuralProcessHead._thermal`
converts learned tendencies to float64; `apply_conservative_update` forms
`start + bound * normalized_tendency`. `updated_fast_state_fields` installs
these endpoints in the retained boundary. They are learned daily transitions,
not calls to the Teacher's iterative snow solver. Lahlou et al.'s implicit
root derivative is therefore not justified on this edge by current evidence.

Determine whether the initial discrepancy enters before the float64 thermal
update, and separate newly introduced same-input arithmetic differences from
amplification of different incoming states. A large physical bound alone is
not evidence of an error: measure the actual decoder slope and input scales.

## Bounded experiment

Use accepted update 3 and existing artifact 322, windows 5 and 1. Reuse the
canonical verbose and compact seven-day traces. Replay each day with the same
verbose incoming state through one-day detailed and state-only executables.
A diagnostic-only tracing hook exposes the existing raw snowhead outputs and
decoded tendencies without changing their formulas. It is restored after
tracing. Independently reconstruct the decoder and affine update on the host,
and compare the post-update and retained-tail endpoints.

At days 2, 4, and 7 of window 5, apply the observed incoming-state difference
under a single executable, either to all state, both snowheat/snowtemp mirrors,
or the complement. Compare the snow-only directional JVP with centered finite
changes at multipliers 1, 0.5, and 0.1. Record JVP primal drift separately so
finite-difference discrepancies are not automatically blamed on autodiff.
These directions measure sensitivity to the observed discrepancy, not a
parameter update or a new gradient acceptance claim. Diagnostic output
capture may itself alter compiled arithmetic; retain comparisons to original
scan outputs rather than declaring the instrumented path canonical.

Runner: `scripts/hpc/audit_conditioned_attention_thermal_source.py` and its
matching shell wrapper. One idle free gln01 V100, one process, 30-minute bound.
No optimizer, parameter retention, monitor read, Teacher generation, or paid
submission. This is a source diagnosis; it does not admit training.

Follow the measured source. Repair a demonstrated arithmetic issue and
recheck forward/gradient consistency before repeating full integration.
Do not relax tolerances, change architecture, or truncate cross-day gradients.

## Execution record

The first launch at `8aa5331` stopped before model initialization: the new
runner wrote its progress JSON before a reused initializer checked that the
output did not yet exist. `ff9310c` corrects ordering, with three local tests
covering decoder capture, one-day directional derivatives, parser/launcher
arguments and this startup lifecycle. Ruff and shell syntax checks pass.

The corrected run completed in 1328.8 seconds on free gln01 GPU 1 with a
30-minute limit. Both GPUs were idle at prelaunch. Report/log:
`runtime/outputs/diagnostics/conditioned-thermal-source-ff9310c.json` / `.log`.
No paid allocation or training update was launched.

## Completed findings

1. Same-input one-day state-only versus detailed execution reproduces the
   original Day-1 snowheat difference exactly: `[0.4074536, -0.4220055,
   0.3492460]`. Instrumented Day-1 snow endpoints equal the original verbose
   scan endpoints. The discrepancy therefore exists before recurrent drift.
2. The snowheat head emits direct `tanh(raw)` tendencies. Its Day-1 raw values
   are approximately `[9.73e-7, 7.67e-7, 2.08e-6]`, multiplied by 1e9.
   Available training normalization scales are approximately
   `[9.20e-7, 1.79e-6, 9.75e-7]`, but direct signed-unit decoding does not use
   them. The host tanh calculation agrees to roundoff. The recorded start
   plus decoded tendency times bound reproduces **all 84** snowheat/snowtemp
   retained endpoints exactly in the host float64 check. This localizes the
   change upstream of the final affine update, without identifying a specific
   neural primitive or compiler optimization as the source of rounding drift.
3. Under one identical executable at window 5 Day 4, full incoming-state
   differences change layer-1 snowheat by -211069.689; snow-only differences
   (both heat/temperature and both mirrors) change it by -211052.460. The
   complementary non-snow perturbation changes that layer by zero. This is
   direct causal evidence for this window's snow-state amplification.
4. Day-4 snow-only JVP predicts -206304.566; centered changes approach it:
   -204211.718, -205826.020, -206175.027 at multipliers 1, 0.5, 0.1. The final
   layer-specific discrepancy is about 0.063%. Day-2 tiny perturbations do
   not converge cleanly and the JVP primal itself differs from forward.
   Thus this is not an all-direction gradient acceptance or proof of a
   general JAX defect. One JVP/secant finite-mask mismatch is recorded; the
   diagnostic uses finite intersections and does not claim physical-mask
   acceptance from these comparisons.

### Teacher comparison changes the repair priority

A separate read of the same existing prepared artifact on free `test01`
requires no Teacher execution or model compilation. It writes
`runtime/outputs/diagnostics/conditioned-thermal-source-teacher322.json`.
Both windows have zero snow mass, depth, liquid water and snowheat on all
seven Teacher days; snow density is 50 and temperature 273.15 K. These are
defined, snow-free states, not missing labels. Window 5 is landpoint
`303.0-103.0`, 1963, start row 88; window 1 is `229.0-117.0`, start row 348.

The candidate instead creates nonzero snowheat immediately and reaches about
30.2 million on Day 7 in window 5 (35.3 million in window 1). A passing
score-consistency window is therefore not a scientifically valid snow-state
control. The problem includes a learned transition leaving Teacher support,
not only a compiler/output-layout discrepancy. Higher arithmetic precision
alone does not correct this off-support trajectory.

### Diagnostic caveat and correction

The `ff9310c` report's `fast_endpoint`, `affine_host_max_abs_error` and
`tail_minus_fast_max_abs_error` fields used compact canonical leaf widths to
index the full-PFT thermal vector. These fields are invalid and must not be
used as evidence of a physical update failure. The independent check above
uses actual retained endpoints and the recorded raw/decoded outputs; it
requires no GPU rerun. The runner now derives offsets from actual prediction
shapes, has a focused expanded-PFT regression test, and also records the
direct affine-to-tail check. Same-input and intervention results use canonical
state coordinates and are unaffected by this diagnostic indexing issue.

## Next action

Prioritize snow-state physical consistency and conditioning before another
compiler-layout or precision-only trial. Audit the source-backed day-end
no-snow, snowfall and melt boundaries and their v6 examples, including both
state mirrors. Use model inputs/predicted physical state to determine regimes,
never Teacher next-state values at inference. `snow3lheat_explicit` provides
an enthalpy identity, but HYDROL cleanup occurs after its evaluation, so do
not assume every saved endpoint can be reconstructed without checking that
ordering. Freeze the smallest source-supported correction before modifying
the head or canonical updater. Then perform local state/budget/gradient
checks and the original paired 7/30-day admission. Accepted recovery stays
at update 3; no new model or gradient default is admitted.
