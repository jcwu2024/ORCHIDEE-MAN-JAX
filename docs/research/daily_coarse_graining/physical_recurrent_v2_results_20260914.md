# Physical feedback and recurrent v2 admission results

Both free gln01 tasks completed with exit0. Python106777/112477 and pipeline
111660 are absent. No paid allocation, Teacher regeneration or final-model
acceptance. Execution checkouts remain unchanged at64c0628 andb247e65.

## Single-day physical loss

`outputs/physical-feedback-v2-updates.json`: completed2066.9s, two complete
batch32/micro4 AdamW updates from fresh seed20260911, all109464375 parameters
trainable. Technical admission passes: both native/JVP projections agree,
parameters/optimizer/gradients remain finite, all physical hard counts zero.
Peak live GPU allocation9.862GiB versus20.631GiB reserved. Reverse compilation
took936.95s and JVP compilation631.20s; this is admission overhead, not steady
training update time. Both updates were clipped at threshold1.0; preclip norms
remain large. Two updates do not establish convergence or a good clip threshold.

Fixed64 excluded TRAIN-monitor outcomes from initial to update2:

| Physical RMSE | Initial | Update2 | Change |
| --- | ---: | ---: | ---: |
| Canopy daily carbon change, gC/m2 | 43396.249 | 43340.151 | -0.129% |
| Litterpart fraction | 0.500046 | 0.499805 | -0.048% |
| GPP, gC/m2/day | 3.064839 | 3.062510 | -0.076% |
| NPP, gC/m2/day | 2.613462 | 2.607992 | -0.209% |
| Autotrophic respiration, gC/m2/day | 0.521047 | 0.523608 | +0.492% |

All four litterpart coordinate means move toward their own Teacher target:
bare target0 predictions0.500211/0.500163 ->0.499878/0.499837;
PFT14 target1 predictions0.500108/0.500082 ->0.500256/0.500240.
This is the intended early direction, not successful fraction fitting.
Canopy predicted daily change remains-31352.599 versus Teacher+7.720gC/m2.
Bare canopy residual increases2.77016 ->2.77746gC/m2 even while total RMSE
falls. Therefore neither the total objective nor aggregate improvement proves
that all coordinates/science outcomes improve. The two-update parameters were
discarded; no trained checkpoint was inherited or promoted.

## New recurrent science scalar

`outputs/recurrent-science-v2-reverse.json` and `-verify.json`: native reverse
completed2013.17s; verification completed647.48s. First preselected TRAIN T7
case only, final1998 weights used as a diagnostic asset; zero optimizer updates.
All109464375 gradient values finite, norm40.0584, all physical/mask/discrete
and science-term validity counts zero. Both resolved directional checks pass:

| Seed | Reverse projection | Independent JVP | Absolute difference |
| --- | ---: | ---: | ---: |
| 20260913 | -35.27029117 | -35.27039187 | 0.00010069 |
| 20260914 | -19.24302572 | -19.24302679 | 0.000001073 |

Primal loss0.249002709 versus JVP primal0.249002622 passes the unchanged
tolerance. Native actual peak allocation2.684GiB; compiled estimated live
2.266GiB. This is one T7 trajectory's parameter AD, without AdamW state or
mixed batches; it is not the memory estimate for full mixed/T30 training.
Native gradient is preserved at runtime/outputs/diagnostics/
recurrent-science-v2-stages/native.pkl. This establishes bounded directional
evidence for the actual new scalar, not full-coordinate/finite-difference or
T30 certification, and does not claim improved rollout accuracy.

## Next operation

Retain Attention, v2 physical losses, the shared recurrent scalar and accepted
346539-day TRAIN pool. Next declare and implement bounded combined one-day +T7
optimizer updates, with fixed sampling/weights and physical reporting. Reuse
these unchanged scalar checks; test the consequential combined update boundary
and actual memory/time. Then admit T30 and specify expanded learning budget,
unique exposure and checkpoint rules. Do not rerun either completed admission
or start unchanged-loss longer training merely because total loss decreased.
Final paper training still needs a consolidated fresh/full-network protocol.
