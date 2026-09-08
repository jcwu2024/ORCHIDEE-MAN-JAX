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
