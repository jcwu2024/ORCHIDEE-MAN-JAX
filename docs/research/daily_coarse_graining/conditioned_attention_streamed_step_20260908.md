# Streamed Backward Discarded-Step Check

Status: completed and admitted for a multi-artifact integration check.

## Evidence And Purpose

Daily replay `4468c92` completed in 502.559 seconds. The maximum 5.58793
absolute state discrepancy belongs to snowheat near 1,029,050 (relative
5.43e-6). The three replay days' downstream linearized effects are about
1e-9 to 1e-8. GPP differences are at most 5.376e-7. Day-16 duality residual
1.64441e-4 reproduces; the old reverse sign failure does not. This supports
testing a repair, not claiming a proven remat root cause or better accuracy.

The next experiment checks whether a negative streamed gradient predicts and
produces descent of the existing complete 30-day science objective. Keep the
conditioned Attention architecture, trainable parameter ownership, canonical
transition, retained processes, and science loss unchanged. Use accepted
update 3 from run `45ab263`, prepared train/train artifact 442, all four of
its 30-day windows. No teacher regeneration, v5, monitor or validation reads.

## Mechanism And Matched Reference

Run canonical forward once per window, retaining daily carries. Differentiate
the existing observation loss with global four-window statistics frozen at
the current parameters. Propagate the full state adjoint backward through
each of 30 one-day VJPs and sum all parameter gradients in float64. Sum the
four window contributions; do not average them again because the global
counts already normalize the objective. Include incoming-state-dependent PFT
fractions in observations. Cache the one-day reverse executable across windows.
No whole-window reverse graph, no truncated BPTT, no optimizer or recovery write.

The negative unit summed gradient defines one direction. Evaluate the same
canonical production objective at displacements +/-1e-4 and +/-1e-5, with
original parameter dtypes preserved. These are two magnitudes along one
discarded direction, not two training updates. The opposite displacement is
the matched directional reference. Recompute all four-window bias statistics
for every finite-step loss; the frozen linearized score is only for gradients.
Record gradient dot actual dtype-rounded displacement, central differences,
per-window losses, component science errors, physical hard counts and counts
of valid science entries. Repeat the unchanged baseline once to measure
same-executable evaluation variability.

## Bounded Admission

Before GPU execution: test full 7/30-day parameter-gradient equality against
monolithic autodiff in float64 and mixed precision, four-window global-bias
equivalence with unequal valid counts, and complete runner/launcher wiring.

On real data require finite gradients, zero physical hard failures, unchanged
science counts, observation-versus-production linearized scores within 1e-6
absolute plus 1e-5 relative (greater than the already observed graph-rounding
differences), and central-difference agreement within 1% for both magnitudes.
This is an engineering derivative tolerance, not Gate D2. Both forward
displacements must reduce the true production loss by more than
max(1e-8, 10 times repeated-baseline variability). The two magnitudes check
that the conclusion is not specific to one finite displacement.

Record all evaluations before classifying. A failed numerical comparison is
a completed scientific rejection, not a launcher failure. No automatic retry
with more scales or new optimizer. Even a pass admits only a subsequent
multi-artifact integration check; it does not resume the 43-constraint training
or establish held-out accuracy. The isolated direction can worsen other
objectives; this check makes no claim about them.

## Execution

Reusable reverse owner: `daily_operator/streamed_adjoint.py`. Experiment:
`scripts/hpc/audit_conditioned_attention_streamed_step.py` and `.sh`, reusing
the qualified conditioned bootstrap and free-node wrapper. One free gln01
V100, 90-minute maximum, one process. Compiled one-day reverse must fit the
existing 24 GiB buffer envelope; the prior measured estimate was 1.50 GiB.
No paid submission. Only incremental JSON is written under runtime outputs;
accepted weights and all earlier results are preserved.

## Local Qualification

32 tests pass, covering the new full recurrent pullback at 7/30 days in
float64 and mixed precision, all four windows' global-bias gradient with
unequal valid counts, same-executable reuse, rounded finite differences,
memory rejection before execution, active nonfinite gradients, a complete
four-window runner that writes only JSON, scientific rejection classification,
and actual launcher argument coverage. Existing streamed/replay, window,
derivative-mode and flux-basis tests also pass. Ruff, CLI help, individual
shell syntax checks and whitespace checks pass. Real-v6 descent passed below.

## Completed Run

Code `c726446` was fast-forwarded to the server through an incremental Git
bundle. Run `conditioned-streamed-step-c726446` completed on free gln01 GPU 1
(actual host `ibc13b02n04`) in 890.713 seconds. All admission gates passed;
the unchanged baseline reproduced exactly. Read
`runtime/outputs/diagnostics/conditioned-streamed-step-c726446.json` and `.log`
for evidence; JSON is also under local `outputs/diagnostics/`. At displacement
1e-4, predicted loss change is -0.00088241257 and actual -0.00088227829;
loss falls from 0.1810040027 to 0.1801217244 (0.4874%). At 1e-5 the predicted
and actual changes are -8.77815e-5 and -8.78681e-5. All four windows improve
at both positive displacements and worsen at their negative counterparts.
Central-difference discrepancies are 0.1357% and 0.1488%; all hard counts are
zero and science counts are unchanged. Window-0 gradient norm is 2.63118,
versus the old production reverse's 1097.9184. The summed gradient norm is
8.82596, and the reverse executable compiled once. Accepted update-3 recovery
is unchanged; no candidate was retained.

Next: [`conditioned_attention_streamed_integration_20260908.md`](conditioned_attention_streamed_integration_20260908.md).
