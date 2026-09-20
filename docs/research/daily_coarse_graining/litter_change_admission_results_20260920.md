# Physical litter-change admission results

Free GPU0 run at immutable22e7fb9 completed with exit0 in2512.68s
(41min53s), on2026-09-20 around13:40 CST. Python89790 and timeout89756 are
gone. No paid computation or new training checkpoint. Source artifacts:
`runtime/outputs/diagnostics/litter-change-admission-v1.{json,log}`;
local copies `outputs/litter-change-admission-v1.{json,log}`.
Protocol: `litter_change_admission_protocol_20260920.md`.

## Conclusion

The changed daily objective has passed bounded technical admission. Its
supervision moves the intended carbon pools in a useful direction: after
two updates, aboveground litter change RMSE decreases8.60-8.65% and belowground
12.12-12.22%, compared with only0.50-0.82% under the old daily objective.
All32 development sample/owner residual magnitudes improve. This supports
the supervision-gap hypothesis and retaining this loss mechanism for the
next bounded joint experiment; it does not establish a good carbon model.

The remaining carbon error is still very large and wrong-sign. GPP/NPP
errors increase relative to initialization in both arms. The new objective
does not improve production accuracy in this two-update test, although its
GPP/NPP final errors are slightly smaller than the matched old arm. Weight0.1
causes the litter term to dominate the daily gradient; do not promote it
unchanged to a long training recipe without considering the joint loss balance.

## What ran and what passed

Both arms loaded identical update4000 parameters AND AdamW state from the
original direct joint pool run. Each took two DAILY-ONLY full-network updates
on original TRAIN batch indices4000/4001, batch64/micro8, LR1e-5, clip1.
All109,464,375 parameters were trainable. Final AdamW step4002 in both arms.
This is128 daily draws per arm, with the same draws shared between arms;
it is not256 distinct examples or a new epoch. No R7/R30 gradients or rollout
evaluations were included. Initial physical panels are identical between arms.
The fixed8 and excluded8 panels are reused TRAIN development inputs, not
independent held-out validation.

- New complete daily gradient finite, nonzero, norm7270.7147 on first micro8.
- Isolated unweighted litter gradient finite/nonzero, norm72707.0768.
- Reverse/JVP projected values0.2485444374/0.2485443097 and
  0.7821571329/0.7821570714. Absolute discrepancies1.277e-7/6.147e-8;
  relative discrepancies0.0000514%/0.00000786%. Existing tolerances passed
  with resolved, nonzero projections; no tolerance relaxation.
- Four full daily optimizer updates completed, parameter/AdamW values finite.
- Recorded hard counters zero in gradients, training microbatches and panels.
  Budget-counter scope remains endpoint bookkeeping, not independent
  ecosystem carbon closure.
- `technical_admission_passed=true`, `full_mixed_admission=false`,
  `scientific_acceptance=false`.

## Physical development results

All RMSE values are gC m-2 day-1; smaller is better. Litter values are RMSE of
daily carbon change, not relative error in the large standing inventory.

| Panel | Quantity | Initial | Old loss after2 | New loss after2 |
|---|---|---:|---:|---:|
| Fixed8 TRAIN | Aboveground litter | 35.3753 | 35.1940 | 32.3156 |
| Fixed8 TRAIN | Belowground litter | 25.6317 | 25.4225 | 22.5242 |
| Excluded8 TRAIN | Aboveground litter | 35.4116 | 35.2354 | 32.3658 |
| Excluded8 TRAIN | Belowground litter | 27.9887 | 27.7653 | 24.5683 |
| Fixed8 TRAIN | GPP | 0.16105 | 0.21258 | 0.20329 |
| Fixed8 TRAIN | NPP | 0.13470 | 0.18549 | 0.18091 |
| Fixed8 TRAIN | Autotrophic respiration | 0.03182 | 0.03330 | 0.03105 |
| Excluded8 TRAIN | GPP | 0.17376 | 0.18273 | 0.18268 |
| Excluded8 TRAIN | NPP | 0.14186 | 0.15399 | 0.15377 |
| Excluded8 TRAIN | Autotrophic respiration | 0.06389 | 0.06236 | 0.06359 |

For context, fixed8 Teacher mean above/below changes are+0.0442/+0.1592;
candidate means after2 remain-32.2669/-22.3612. Excluded8 Teacher means are
+0.0357/+0.0958 versus candidate-26.8353/-21.5754. Improvement is real on
these samples, but far from adequate absolute accuracy.

## Loss balance and cost

On the first identical64-sample update, old daily scalar0.011519 becomes
28.820699 with the new term. Unweighted litter loss288.091797 contributes
28.809180. Gradient norms are1.15559 versus1471.73975. Candidate clipping
scales are0.00067947 and0.00017691, versus old0.86536 and0.65373.
This is evidence of a large gradient imbalance, not proof that clipping is
incorrect or that learning is impossible. Existing AdamW moments were inherited
from the original objective, so two steps also do not establish steady-state
optimization under the revised loss. The second step uses a different batch;
its larger loss cannot be interpreted as a same-data training regression.

Explicit reverse/JVP compilations took975.94/651.99s. Initial forward and
optimizer compilations add further setup time. Warm complete daily updates
took7.33/7.33s for the candidate and7.71s for the old arm's second update;
do not extrapolate41min53s as the cost of four warm updates. This includes
admission diagnostics, not a measured joint-training throughput claim.
JAX peak live memory16,892,753,920bytes=15.73GiB; allocated pool25.39GiB.
No OOM occurred. Live peak and pool reservation are different quantities.

## Next operation

Keep Attention/decoder and this separate physical loss mechanism. Specify a
short matched DAILY+R7+R30 experiment, including explicit loss weighting and
7/30-day physical carbon/production evaluation before any longer training.
Use the observed TRAIN gradient imbalance to choose/test the coefficient;
do not select it from the reused development panels or assume0.1 is final.
Near1e-4 is a reasonable starting coefficient to examine: linear rescaling
would bring the first-batch new contribution to the order of the old daily
gradient. That estimate is not yet a joint-gradient calibration or a frozen
new protocol, because recurrence gradients and vector interactions matter.
Reuse this native AD evidence; no historical custom-adjoint investigation or
full repeated JVP compilation is warranted just to change a scalar weight.
Check a bounded complete mixed update before scaling. No next job submitted.
