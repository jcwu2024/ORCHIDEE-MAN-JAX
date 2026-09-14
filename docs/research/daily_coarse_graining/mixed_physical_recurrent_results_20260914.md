# Mixed one-day and seven-day update results

Free gln01 GPU0 admission completed at commit `9e0b7bd`, exit0, with two
complete fresh full-network updates. All physical/defined-value checks pass
at initialization and after both updates. Daily, recurrent and combined
gradients are finite and nonzero. This admits bounded mixed optimizer
execution, not useful trained accuracy, generalization or a final paper method.

Protocol: `mixed_physical_recurrent_admission_20260914.md` and its machine
manifest. Report/log: `runtime/outputs/diagnostics/mixed-physical-recurrent-v1-gpu.{json,log}`.
Local report: `outputs/mixed-physical-recurrent-v1-gpu.json`.
First mixed gradient remains in the report's `.stages` directory. No model
checkpoint was retained. Bash97713/Python99324 exited; no paid job submitted.

## Update evidence

Each update uses64 one-day TRAIN examples (micro8) plus two T7 TRAIN windows,
separate means with weights1+1, one clip and one AdamW step. All109,464,375
parameters originate from fresh seed20260911, not trained update1998.
The two updates use128 unique one-day examples and four T7 windows.

| Update | Same-draw mixed loss before | After | Reduction | Recurrent/daily gradient norm | Component cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 387.312176 | 386.822891 | 0.12633% | 0.09020 | 0.89780 |
| 2 | 1296.351845 | 1295.199115 | 0.08892% | 0.20364 | 0.91820 |

Both daily batches and all four recurrent losses decrease on fixed panels
across the two updates. Excluded64-day TRAIN monitor loss decreases581.752321
to580.629982 (0.19292%). Comparing different batches' losses directly would
misrepresent this as an increase. These small same-draw changes are not a
learning-budget evaluation. The recurrent gradient contributes and is aligned
with the daily gradient in these two cases; this does not establish optimal
weights or alignment across TRAIN. Clip scales0.00042395/0.00016111 correspond
to combined norms2358.75/6206.81. Two updates cannot establish clipping trends.

## Physical accuracy

Four T7 TRAIN windows,28 simulated days; initialization versus update2:

| Quantity | Initial RMSE | After two updates | RMSE reduction |
| --- | ---: | ---: | ---: |
| GPP, gC/m²/day | 2.017296 | 2.015263 | 0.10079% |
| NPP, gC/m²/day | 1.518265 | 1.513554 | 0.31030% |
| Autotrophic respiration, gC/m²/day | 0.540895 | 0.542591 | -0.31351% |
| AGB seven-day change, gC/m² | 6.997507 | 6.977858 | 0.28080% |
| BGB seven-day change, gC/m² | 3.585091 | 3.572653 | 0.34695% |
| Canopy organic-carbon seven-day change, gC/m² | 100517.231263 | 100506.268574 | 0.01091% |

Canopy stock remains severely inaccurate at this fresh state; valid budgets
do not establish correct inventory magnitudes. Mean endpoint bias moves from
-72212.8486 to-72204.6499gC/m². Do not compare these fresh four-window errors
with the trained update1998 model's different development panel as regression.

Heterotrophic respiration predictions and Teacher values are all zero in these
windows and the one-day monitor. Zero error is not evidence of learning a
nonzero response. On the excluded one-day TRAIN monitor, GPP RMSE changes
3.064839 to3.062582, NPP2.613462 to2.609353, and autotrophic respiration0.521047
to0.522674. This is not independent spatial/temporal validation.

## Runtime and next work

Total actual-model time1824.4s (30.41min). Recurrent backward compilation took
213.0s; daily backward compilation997.5s; initial setup/forward panels reached
518.6s. The run was dominated by compilation/setup. The second compiled update
took9.13s for gradient computation,13.58s including host checks and AdamW;
subsequent monitoring took9.7s. One warm timing sample is not a production
throughput benchmark.

Measured peak live GPU memory16.803GiB differs from reserved allocator pool
26.978GiB. T7 mixed training fits the32GiB V100 with micro8; this does not
certify T30 or micro16 mixed memory. All three panels were independently
checked from the report: no daily/monitor/recurrent physical failures.

Next extend bounded actual-update admission to T30 and measure memory/runtime,
then execute a declared larger learning budget on the expanded346,539-day TRAIN
pool with rotating TRAIN windows. Reuse completed T7 evidence and caches.
Keep fresh/full-network Attention and report canopy/respiration alongside
GPP/NPP/AGB/BGB. No new microbatch benchmark or historical streamed-reverse
repair is required. Two updates do not justify an architecture change or a
convergence claim.
