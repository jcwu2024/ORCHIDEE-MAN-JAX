# Conditioned Attention Update-4 Derivative Audit

Status: bounded diagnostic frozen before execution. No training update is authorized by this audit.

## Persistent Smoke Result

Run `45ab263` completed normally in 3162.530 seconds. Update 3 was retained,
with fixed-monitor direct GPP down 0.9373% and science macro down 0.7579%.
Update 4 was rejected at every scale from 1 through 1/64; update 5 was not run.
Engineering and physical checks passed. Recovery remains at update 3.

All 40 direct-GPP artifacts improve at every tested scale. Scheduled state
and 7-day losses improve as well, but the 30-day loss increases by 0.0209239%
at full scale and 0.000272762% at 1/64. The current four-source regression
tolerance is 0.0001%. The latter increase is small, but the former is about
209 times the tolerance; this is not evidence for simply relaxing the gate.
The proposed direction predicts a 2% 30-day reduction at full scale.

The scheduled science baseline and gradient-measurement baseline match
exactly in the saved report. Static inspection finds the same samples,
statistics, and flux basis in forward and backward, with rematerialization
enabled in the reverse graph. This does not yet prove numerical derivative
agreement through the recurrent physical computation.

## Bounded Question And Method

Use the exact update-3 recovery and the rejected update-4 panel, without
retraining or recomputing the direction optimization. Reconstruct only the
seven nonzero simplex support sources: five direct-GPP artifacts and the
scheduled 7/30-day gradients. Preserve the production reduction order and
record historical versus reconstructed gradient norms and direction norm.

For each scheduled microbatch, record the reverse graph's returned scalar
alongside the same linearized scalar formed from forward statistics. The
production helper discards this reverse primal; observing it tests whether
rematerialization changes the trajectory before differentiation.

On all eight scheduled artifacts, evaluate 30-day loss at scales
0, +/-1, +/-1/4, and +/-1/16. Compare:

- historical and current zero/full-step values;
- the ideal directional prediction with the gradient dotted into the actual
  dtype-preserved parameter displacement;
- forward and backward finite changes and their central secants;
- artifact-level prediction disagreement and physical counts.

This distinguishes a mismatched replay, material parameter rounding, a
forward/reverse primal discrepancy, and a remaining derivative/finite-step
disagreement. A remaining disagreement alone does not identify a particular
physical branch or prove an autodiff bug. Numerical tolerance and model
promotion thresholds remain unchanged.

## Execution

Runner: `scripts/hpc/audit_conditioned_attention_update4_derivative.py`.
Wrapper: same basename with `.sh`. Reuses the qualified persistent launcher
arguments, JAX cache, environment, and accepted v6 prepared artifacts.
One free gln01 GPU, 90-minute timeout, incremental JSON evidence after each
artifact. No monitor, fresh validation, Teacher regeneration, optimizer,
parameter recovery write, paid queue, or 365-day evaluation.

Parent reports remain under
`runtime/outputs/training/target-oriented-coupled-gpp-conditioned-attention-persistent-45ab263/`.
Local report copies are `outputs/diagnostics/conditioned-persistent-45ab263-report.json`
and `outputs/diagnostics/conditioned-persistent-45ab263-rejected.json`.
