# Selected litter mixed continuation: completed results

## Outcome and scope

The free gln01 GPU1 run completed normally (exit0) in2205.560252s, about36m46s
of runner time. Runtime code5fdfd5e. All80 additional updates completed,
selected-relative20->100 / absolute4020->4100. Checkpoints and full milestone
assessments exist at40/60/80/100. No process remains active and no new paid
job was submitted for this status collection.

Source evidence: `outputs/litter-mixed-continuation-v1.json` and `.log`;
remote prefix `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/litter-mixed-continuation-v1`.
Method: `litter_mixed_continuation_protocol_20260920.md`.

Existing Attention architecture and109,464,375 trainable parameters retained.
Each update uses64 daily draws (microbatch8), oneT7 and oneT30 window.
Joint loss is D_old +5.221716892123007e-5 L_litter +0.5R7+0.5R30;
AdamW LR1e-5, global clip1. No recalibration or new Teacher generation.
This continuation adds5120 daily draws and80 windows of each horizon;
draw counts do not imply unique samples or a full Teacher epoch.
Inherited exploratory weights remain diagnostic, not the final paper recipe.

## Physical results

Same eight development windows per horizon at every assessment. All errors
below are RMSE in gC/m2: cumulative fluxes and stock change from the common
Teacher initial state. These repeatedly consulted development panels are
not a sealed generalization test. Relative20 is the continuation starting
point, not an independently trained old-loss continuation control.

| Horizon/update | GPP | NPP | Autotrophic respiration | AGB change | BGB change | Canopy change | Nonliving change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T7/20 | 3.6040 | 2.7720 | 0.9269 | 2.0680 | 0.7178 | 3.7926 | 271.1057 |
| T7/40 | 5.8451 | 4.5110 | 1.3664 | 3.4734 | 1.0481 | 3.0517 | 109.6161 |
| T7/60 | 3.2315 | 2.5451 | 0.7617 | 1.9506 | 0.6050 | 3.7981 | 68.5181 |
| T7/80 | 2.4345 | 1.8509 | 0.6784 | 1.4023 | 0.4633 | 2.5204 | 58.8302 |
| T7/100 | 2.8137 | 2.1744 | 0.7462 | 1.6787 | 0.5037 | 2.5143 | 58.1776 |
| T30/20 | 3.9665 | 3.0811 | 1.3021 | 2.2319 | 0.8702 | 18.1893 | 2497.0089 |
| T30/40 | 16.6360 | 12.5806 | 4.1559 | 9.0868 | 3.4314 | 20.8385 | 1423.6862 |
| T30/60 | 4.8976 | 4.2743 | 1.1337 | 2.7801 | 1.5022 | 8.2543 | 1085.4264 |
| T30/80 | 5.1150 | 4.7585 | 1.0647 | 3.0269 | 1.7333 | 8.1244 | 928.0158 |
| T30/100 | 5.7692 | 4.5888 | 1.6329 | 3.1382 | 1.4652 | 12.0304 | 903.0992 |

Nonliving error decreases78.54%T7 and63.83%T30 from20 to100. T7 production,
respiration and biomass errors also improve. T30 production/biomass errors
remain above20 despite recovery from a large transient regression at40.
Canopy error improves at100 versus20 but worsens versus60/80.
There is no uniform best milestone across these physical objectives.

At100, mean predicted nonliving changes are-6.9214(T7) and-295.4815(T30),
against Teacher+33.4930 and+125.2697. Wrong-sign windows decrease from8/8
at20 for each horizon to4/8T7 and5/8T30. Residual drift remains substantial.

Daily litter-change RMSE on the fixed eight TRAIN samples decreases from
11.46->1.53 aboveground and4.91->0.13 belowground. On the eight excluded-TRAIN
diagnostic samples it decreases11.56->1.55 and5.71->0.62. At100 aboveground
mean daily change remains negative(-1.48/-1.25) versus Teacher about+0.04;
belowground means are+0.29/+0.30 versus Teacher+0.16/+0.10. Thus learning of
this term is real on these panels, but accurate stock dynamics are not yet
established. Excluded TRAIN samples do not constitute a held-out site test.

## Technical and interpretation limits

All four milestone physical gates and metric-population checks pass.
Technical continuation and physical monitoring are true; scientific
acceptance is false. The declared two-consecutive-monitor severe-regression
guard never stops the run. The normalized science screen fails at40 and
passes60/80/100; that screen is not final physical-unit acceptance.
Endpoint budget counters are not independent ecosystem carbon-flux closure.
Teacher RH totals are zero in these monitor windows and cannot establish
learning of nonzero RH. No new AD/numerical-runtime failure is indicated.

Peak allocator live bytes16,915,194,368 (15.75GiB), reserved pool27,263,710,208
(25.39GiB). Do not report the reservation as live model memory.

The selected-only continuation cannot attribute every production change
causally to the litter term: no matched old-loss80-update continuation was
run. Results also do not show that longer training cannot help. They show
that extending this fixed setup has not yet improved all required outcomes.

## Next work

Use existing saved per-window results across20/40/60/80/100 to locate T30
GPP/NPP/biomass regressions and their association with carbon-stock changes.
Determine whether a few windows dominate or the tradeoff is broad. Use that
evidence to specify a bounded loss-balance/learning experiment if warranted;
do not assume a proven gradient conflict from aggregate curves alone.
Keep Attention, the shared admitted native AD and all milestone checkpoints.
No automatic long continuation, architecture replacement, or paid submission.
