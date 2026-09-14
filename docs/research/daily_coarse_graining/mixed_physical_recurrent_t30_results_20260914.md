# Thirty-day mixed update results

Free gln01 GPU0 run at commit70dd976 completed normally, exit0, two fresh
full-network updates. Bash18684/Python19718 have exited. The actual T30
native/JVP directional gate and all update/physical gates pass. This admits
bounded T30 mixed optimizer execution, not trained accuracy or generalization.

Specification: `mixed_physical_recurrent_t30_admission_20260914.md`.
Report/log: `runtime/outputs/diagnostics/mixed-physical-recurrent-t30-v1-gpu.{json,log}`.
Local report: `outputs/mixed-physical-recurrent-t30-v1-gpu.json`.
First actual T30 native gradient and first mixed gradient remain in the
`.stages` directory. No reusable model checkpoint and no paid job.

## Gradients and updates

Same fresh seed20260911, all109,464,375 parameters, daily64/micro8 plus two
sequential T30 TRAIN windows per update, separate means with outer weights1+1,
one clip/AdamW step. No Teacher reset across days, data regeneration or model
change. Four fixed T30 TRAIN windows and128 one-day TRAIN rows were used.

| Direction seed | Native reverse projection | JVP projection | Relative difference versus JVP |
| --- | ---: | ---: | ---: |
| 20260913 | 18.87075925 | 18.86138742 | 0.04969% |
| 20260914 | -15.56258564 | -15.54600512 | 0.10665% |

Both projections are resolved and pass the unchanged1% relative/1e-6 absolute
tolerance. Native/JVP primals agree. This is two-direction evidence for the
actual first fresh T30 scalar, not coordinate-wise or finite-difference proof.

| Update | Same-draw mixed loss before | After | Reduction | Recurrent/daily gradient norm | Component cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 352.174983 | 351.708951 | 0.13233% | 0.028666 | 0.824494 |
| 2 | 829.246070 | 828.261168 | 0.11877% | 0.004470 | 0.696876 |

All component/mixed gradients are finite and nonzero; proposed parameters and
AdamW state are finite. All four fixed recurrent losses decrease across both
updates. Excluded64-day TRAIN monitor loss changes581.752321 to580.636341.
The recurrent gradient is aligned with the daily gradient but smaller in
these cases. Do not infer optimal weighting, vanishing gradients or a
T7-versus-T30 causal effect from different windows and two updates.

## Physical results

Four T30 TRAIN windows (120 days), initialization versus update2. Independent
review of all three saved panels found zero daily/monitor/recurrent hard counts.

| Quantity | Initial RMSE | Update2 RMSE |
| --- | ---: | ---: |
| GPP, gC/m²/day | 2.761164 | 2.759038 |
| NPP, gC/m²/day | 2.129386 | 2.124298 |
| Autotrophic respiration, gC/m²/day | 0.643164 | 0.646050 |
| AGB thirty-day change, gC/m² | 39.671614 | 39.559989 |
| BGB thirty-day change, gC/m² | 23.248645 | 23.208089 |
| Canopy organic-carbon thirty-day change, gC/m² | 24159.705685 | 24159.675046 |

GPP/NPP/biomass gains are tiny, autotrophic respiration slightly worsens, and canopy
error remains severe with effectively no improvement. Heterotrophic respiration
predictions and Teacher values are zero in this panel; zero error is not a
learned nonzero response. These fresh TRAIN windows differ from the earlier
T7 panel and trained development panels, so they are not an architecture or
trained-model performance comparison. Physical validity does not establish
correct carbon inventories.

## Runtime and decision

Actual-model wall time2955.2s (49.25min). T30 reverse compilation1917.6s,
JVP compilation512.1s, and daily compilation/cache loading102.2s dominate.
The first update's569.8s gradient timing includes the one-off JVP check and
must not be used as steady training cost. Second update:16.11s gradient work,
20.76s including host checks/AdamW, plus about13.6s for the full fixed panels.
This is one warm measurement, not a full training throughput benchmark.

Peak live memory16.801GiB; reserved allocator pool26.978GiB. The32GiB V100
fits this T30 mixed workload with micro8. Compiled recurrent memory is about
2.27GiB; daily micro8 remains the larger executable. This does not prove
micro16 or arbitrary larger-window memory capacity.

T7 and T30 mixed technical admissions are now complete. Proceed to implement
the declared200-update fresh/full-network learning design with daily64 plus
one T7 and one T30 per update. Its400-window index is prepared (267 sites,
44 years,395 point-years,7387 unique recurrent Teacher days), and the expanded
346,539-day daily pool is ready. Reuse unchanged AD evidence and compilation
caches; do not insert another identical two-update/JVP probe solely because
the linear mixture or sampled data changes. Persist checkpoints and assess
fixed physical/science panels at declared milestones. Resource estimate must
include startup/compilation, data IO, evaluation and checkpoint writes before
any concrete paid request. No200-update job has been submitted yet.
