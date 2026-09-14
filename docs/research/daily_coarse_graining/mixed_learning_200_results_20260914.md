# Fresh mixed learning: 200-update results

Job15069880 completed200/200 updates on gnall/ibc13b02n01, code12eb22c,
Slurm COMPLETED/exit0:0, elapsed01:17:20. Runner elapsed4562.474s; checkpoint
forward replay passed. Source report:
`runtime/outputs/training/mixed-learning-v1/report.json`, local copy
`outputs/mixed-learning-v1-15069880.json`. No further job submitted.

Fresh109464375-parameter joint optimization used the declared D+.5R7+.5R30
objective, daily64/micro8, one window per recurrent horizon, constant lr1e-5,
clip1 and AdamW. The run consumed12800 unique one-day TRAIN samples
(372 sites,44 years,655 point-years) and400 recurrent windows. It covers
3.69% of the346539-day prepared daily pool, not an epoch. The pool itself is
4.04456% of8568025 eligible Teacher TRAIN days; daily draws in this run cover
about0.1494% of that total. Recurrent exposure is additional and may overlap
daily exposure; do not add occurrences as unique samples.

## Learning and physical accuracy

All four0/50/100/200 fixed TRAIN/development panels passed physical checks and
metric-population completeness. All development science screens still fail.
Fixed TRAIN daily loss312.456552 ->120.478521 (-61.44%); excluded64-day monitor
581.752321 ->225.464426 (-61.24%). Both continue improving between100 and200;
this does not demonstrate convergence or an overfitting failure.

Unchanged T30 development-panel RMSE:

| Quantity | Initial | Update50 | Update100 | Update200 | Initial-to200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPP, gC/m2/day | 2.103943 | 2.036606 | 1.953600 | 1.653691 | -21.40% |
| NPP, gC/m2/day | 1.825608 | 1.714912 | 1.599527 | 1.276833 | -30.06% |
| Autotrophic respiration, gC/m2/day | .347214 | .383353 | .410824 | .415484 | +19.66% |
| AGB change, gC/m2 | 37.238195 | 34.861189 | 32.418161 | 25.945374 | -30.33% |
| BGB change, gC/m2 | 17.207383 | 16.215532 | 15.156507 | 12.024198 | -30.12% |
| Canopy carbon change, gC/m2 | 81478.757 | 81477.500 | 81474.551 | 81331.491 | -.18% |
| Nonliving carbon change, gC/m2 | 88076.997 | 88075.229 | 88070.974 | 87924.527 | -.17% |

T7 also improves GPP2.560060 ->2.180327 and NPP2.253986 ->1.700949gC/m2/day,
AGB11.941939 ->8.922219 and BGB3.927923 ->2.840773gC/m2, while RA worsens
.367068 ->.505620gC/m2/day. RH Teacher totals are zero on these panels; zero
RH error does not establish learned nonzero response. The absolute AGB/BGB
normalized gates pass at T30, but GPP/NPP and other major science gates do not.
Canopy/nonliving errors remain very large despite the falling aggregate loss.

The result supports actual learning and an operational training path; it does
not establish adequate fitting, independent generalization or long-run biomass
accuracy. Poor final accuracy cannot be attributed to architecture with such
limited exposure. More data/updates are justified, but data scarcity is not
proved to be the sole cause of the RA/canopy imbalance.

## Runtime and data-volume decision

Complete update times11..200 average15.2607s, range14.9852..30.2210s. Initial
update116.47s includes first-use overhead. Peak device live17740911616 bytes
(16.52GiB), reserved26.98GiB; host peak20992569344 bytes (19.55GiB). At the
approved additive rates, elapsed-job estimate is1.288889h*(8*.07+2.20)=3.56yuan;
this is an estimate, not an account billing receipt, and is below16.56yuan cap.

The limiting data exposure was the declared200-update pilot budget, not missing
Teacher data or a need to fit the whole dataset in GPU memory. Microbatch
controls simultaneous memory; more batches primarily require more elapsed time.
The completed technical/learning evidence justifies moving beyond repeated
200-update pilots and planning a meaningful data-coverage budget.

Recommended next planning target: a complete pass through the existing346539
unique-day pool,5415 batch64 updates with21 terminal padding occurrences,
while retaining multi-day training and fixed science monitors. At the measured
mixture throughput this is about23h of updates, with startup/IO/monitor overhead
additional; budget roughly23-25h/~65yuan before a concrete resource request.
This is a planning extrapolation, not a new approved/submitted job. Expand the
recurrent metadata schedule (current index only200 pairs) and declare exact
initialization/continuation, recovery segments and monitoring before execution.
Do not relabel this exploratory continuation as the final reproducible method:
the eventual consolidated full-network protocol still defaults to declared
fresh initialization under AGENTS.md. Do not change architecture merely because
this underexposed run fails the absolute science screen. Track the worsening
RA and stagnant canopy/nonliving losses explicitly during scale-up planning.
