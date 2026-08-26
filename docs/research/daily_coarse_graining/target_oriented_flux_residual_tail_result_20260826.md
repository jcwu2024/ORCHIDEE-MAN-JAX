# Target-Oriented Flux Residual-Tail Result (2026-08-26)

## Decision

The frozen 96-window, zero-training audit passes its engineering gate, but it
does not admit a robust-loss continuation. The dominant new evidence is model
Jacobian amplification: one ordinary 7-day loss produces `92.0677%` of the
sum of all 64 window gradient norms. Loss-only soft-L1 and log1p weighting
does not materially repair the 7-day direction conflict and only partly
improves the 30-day geometry.

The next gate is a narrow zero-training gradient-path attribution on the
extreme 7/30-day maintenance-respiration windows and loss-matched controls.
It must distinguish the primary output path, shared Attention backbone, and
cross-day rollout recurrence before any loss, gradient-conditioning, or
architecture intervention is selected.

No optimizer or parameter step was taken. No checkpoint or recovery was
written. Teacher v6 was not regenerated, historical v5 was not used,
validation values and sealed test were not read, and no paid resource was
used. Unchanged continuation, 365-day rollout, Gate D2, and product acceptance
remain closed.

## Evidence Identity

- branch: `research/daily-coarse-graining`;
- executable commit:
  `b80494facbf6f3c67f287d8476029dd49ae63e1d`;
- report schema: `target_oriented_flux_residual_tail_audit_report_v1`;
- remote report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/target-oriented-flux-residual-tail-b80494f/report.json`;
- local ignored copy:
  `outputs/diagnostics/target-oriented-flux-residual-tail-b80494f/report.json`;
- report SHA256:
  `e1c094e2dba5f2eeb4ea8becfdf6623ef58fadd3e9d02518312a71d693d27d9e`;
- residual-tail contract SHA256:
  `bafcb2daa0bcfc0e7108022b44d8af4741d87a1c5e73e371104e5985819bb763`;
- update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared-manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The free-`gln01` run completed all `64/64` 7-day and `32/32` 30-day
window gradients in `2494.92 s`. All `83,755,173` gradient values per window
were finite and every physical hard-failure count was zero. JAX peak live
device memory was `2,204,030,464` bytes (`2.05 GiB`); the roughly 25-GB
allocator limit reported by the runtime was reservation capacity, not live
use.

The first execution at commit `39d4957` completed compilation but stopped on
an undeclared `1e-8` forward-versus-gradient score reproduction check. That
check was not a scientific gate. Commit `b80494f` correctly records the
floating-point reproduction differences instead: the maximum relative
difference is `1.181e-4` at 7 days and `5.888e-7` at 30 days. The accepted run
changed no score, gradient, or decision threshold.

## Concentration Result

| Quantity | 7 days | 30 days |
| --- | ---: | ---: |
| Loss effective sample size | 21.763 / 64 | 12.028 / 32 |
| Loss top-1 share | 12.811% | 20.157% |
| Loss top-5 share | 38.439% | 51.666% |
| Gradient-norm effective sample size | 1.179 / 64 | 7.001 / 32 |
| Gradient-norm top-1 share | 92.068% | 24.008% |
| Equal-window minimum cosine | -0.97003 | -0.90485 |
| Equal-window negative-window fraction | 34.375% | 28.125% |

No runtime-year group satisfies the frozen broad-regime rule. The 7-day loss
distribution is moderately concentrated, but its gradient distribution is
qualitatively more extreme. The largest-loss 7-day window contributes only
`12.811%` of total loss and has gradient norm `10.83`; it is not the
largest-gradient window.

The 7-day gradient maximum is:

```text
update 465, runtime year 1981
landpoint 081.0-107.0, start row 90
standalone loss 0.0528476000
full gradient norm 22489.3023
```

Its loss is approximately rank 12 of 64 and contributes about `2.12%` of the
total 7-day loss, yet its gradient norm is 54 times the second-largest value
and dominates the aggregate norm mass. Its component scores are GPP
`0.143739`, maintenance respiration `0.276469`, and growth respiration
`0.002623`.

The 30-day gradient maximum is also maintenance dominated:

```text
update 465, runtime year 1981
landpoint 299.0-095.0, start row 207
standalone loss 0.0349379740
full gradient norm 53.8701
```

Its maintenance score is `0.276488`, while GPP and growth scores are only
`0.003015` and `1.004e-6`. A loss-matched 30-day window at update 572 has a
maintenance score of `0.273821` but a full gradient norm of only about `2.10`.
This directly motivates path attribution rather than treating the residual
magnitude itself as the mechanism.

## Bounded-Influence Result

At 7 days, none of the six frozen soft-L1/log1p choices improves the minimum
cosine beyond the equal-window `-0.97003`; the best median cosine remains
only about `0.07`, and `35.9%` to `37.5%` of windows remain opposed. Loss-only
tail weighting is therefore insufficient for the principal failure.

At 30 days, `log1p` at one median-loss scale gives a limited improvement:
minimum cosine changes from `-0.90485` to `-0.87437` and the negative-window
fraction from `28.125%` to `18.75%`. This is useful evidence that influence
control may later be one component of a training rule, but it does not remove
the direction conflict and cannot by itself authorize training.

The report's machine action,
`admit_bounded_window_influence_candidate_for_later_zero_update_geometry`, is
the literal outcome of the preregistered rule because both a loss tail and a
gradient tail are present. The more specific measured evidence supersedes any
informal reading of that label: a bounded influence candidate may be retained
for later comparison, but robust-loss-only repair is not established.

## Next Gate

Use only the accepted update-1000 checkpoint and prepared train/train assets.
For the extreme 7/30-day maintenance windows and frozen loss-matched controls,
measure:

1. separate GPP, maintenance-respiration, and growth-respiration gradients;
2. complete parameter-family and Attention/output-path norm attribution;
3. the same field gradients under teacher-forced day-start states, so the
   local daily operator is separated from cross-day free-rollout recurrence;
4. exact reconstruction of the aggregate gradient and all finite/physical
   guards.

This is a diagnostic, not a proposed architecture edit. If the result points
to a network-local mechanism that would require changing the architecture,
the work must pause for user discussion before that edit is made.
