# Corrected Attention: ten-update sustained-learning result

Job **14979295** completed normally (`COMPLETED`, exit `0:0`) in **02:49:07**.
All ten new updates, logical **5--14**, were accepted and saved. There is no
stopping reason. This admits sustained learning on the fixed development
monitor; final model accuracy and independent generalization remain unproven.

## Learning outcome

Changes below are relative to the corrected, saved update-4 monitor baseline.
The monitor is fixed across updates; the changing training-batch losses are
not used as a learning curve.

| Additional updates | Direct GPP loss reduction | 7/30-day science macro loss reduction |
| ---: | ---: | ---: |
| 2 | 1.93896% | 1.59500% |
| 5 | 4.10605% | 3.23903% |
| 10 | 9.35793% | 7.34134% |

Every update decreases both monitor losses and passes both local and cumulative
transfer. All eight direct-GPP artifacts improve cumulatively (8.07--16.17%).
All 18 required science-key/horizon losses improve cumulatively. Science macro
falls from 0.33650933 to 0.31180504.

The final five updates alone reduce direct loss another **5.47676%** and science
macro another **4.23963%**, relative to the update-9 checkpoint. There is no
observed late-run plateau or progressively vanishing step. All selected scales
are 1; parameter-delta L2 ranges from 0.00014112 to 0.00090503 and ends at
0.00056925. Every step is limited by the unchanged 2% predicted-response cap,
with a different direct-GPP artifact supplying its largest relative slope.

Loss reductions are not equivalent to percentage prediction errors. For example,
30-day GPP normalized RMSE falls about 3.56%, from 1.08270 to 1.04413, and still
substantially exceeds its 0.25 accuracy threshold.

## Absolute accuracy remains open

All values below use the existing normalization, not raw physical units or
relative errors against the magnitude of GPP/biomass.
Thirty-day thresholds are RMSE <= 0.25 and absolute signed bias <= 0.07.

| Thirty-day quantity | Initial RMSE | Final RMSE | Initial signed bias | Final signed bias | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| GPP | 1.082703 | 1.044132 | 0.924059 | 0.883512 | No |
| NPP | 1.288877 | 1.243397 | 1.107005 | 1.059523 | No |
| Maintenance respiration | 0.124998 | 0.123400 | 0.021136 | 0.019703 | Yes |
| Growth respiration | 0.639294 | 0.615405 | 0.546656 | 0.521831 | No |
| Autotrophic respiration | 0.435207 | 0.419912 | 0.356849 | 0.340273 | No |
| AGB carbon | 0.023305 | 0.022470 | 0.019044 | 0.018170 | Yes |
| BGB carbon | 0.026234 | 0.025336 | 0.023180 | 0.022265 | Yes |
| Total live biomass carbon | 0.033646 | 0.032458 | 0.028898 | 0.027657 | Yes |
| Ecosystem carbon | 0.020568 | 0.020511 | 0.019674 | 0.019613 | Yes |

The same four quantities (GPP, NPP, growth and autotrophic respiration) fail
the seven-day absolute gate (RMSE <= 0.35, absolute bias <= 0.10). Biomass,
maintenance and ecosystem carbon pass both horizons, as they already did at
the baseline; these are retained passes, not new annual-accuracy achievements.
All accepted monitor physical gates pass.

## Gradient, data and execution evidence

All **960** scheduled window-score comparisons pass: 640 seven-day and 320
thirty-day windows, with complete streamed inter-day gradients. The daily
reverse compiles **once** across all ten updates. The result-summary script
recomputes all 20 local/cumulative transfer gates from the full saved panels
and obtains exactly the recorded results.

The completed run consumes 400 distinct existing prepared training artifacts,
with no artifact overlap with the eight-artifact development monitor. This
means 83,200 direct day positions, 2,560 state anchors, and 14,080 recurrent
day positions. These are sample uses, not deduplicated days or distinct sites;
the sources overlap. No new Teacher data are generated and no v5 data are used.

First update training takes 1116.62 s. Updates 2--10 have a median of 846.51 s
(14.11 min), plus about 75 s per hot monitor. Paid duration implies an estimated
additive charge of **CNY 7.78**, not a billing statement. Slurm reports eight
CPUs/one GPU and peak batch RSS 14,754,248 KiB (about 14.07 GiB host memory).
This is not a measurement of GPU tensor memory.

A throughput detail needs attention before much longer runs: hot update times
rise from 772.23 to 922.52 s, and summed horizon callback times rise from 256.03
to 374.33 s. This is not repeated daily-reverse compilation. The progress
callback currently serializes the entire accumulating report on every event;
the final JSON is 9,119,272 bytes. This is a concrete candidate for avoidable
overhead, but its contribution has not yet been measured separately from GPU
computation and filesystem contention. Reduce or measure that reporting cost
before extrapolating sustained throughput to a much longer run.

## Recovery and next work

On free `test01`, the actual recovery file was opened and confirmed to contain
completed logical update 14, ten accepted records, ten monitors, no stopping
reason, and the enabled corrected boundary. Its size is 65,796,628 bytes.
Execution commit remains `7f13f10`; the queued-job checkout restriction is now
released because the job has completed.

Remote run directory:
`runtime/outputs/training/conditioned-snow-free-continuation-14979295/`.
Resume from its **`recovery.pkl`**, not old update 3 or the starting candidate.
Local full report: `outputs/conditioned-snow-free-continuation-14979295.json`.
Summary and learning plot use the same stem with `.summary.json` and
`.learning.png`. Reproduce with:

```text
conda run -n ORCJAX python -m scripts.dev.summarize_conditioned_snow_free_continuation outputs/conditioned-snow-free-continuation-14979295.json
```

Recommended next work: keep the Attention architecture and admitted corrected
gradient/optimization rules, prepare a longer bounded continuation, and define
an additional independent evaluation panel before reading its model scores.
First address the accumulating progress-report overhead with a bounded CPU
check. Materialize the new training schedule: the current 100-group formula
cannot be extended blindly beyond logical update 23. Any reuse of train-only
artifacts must be explicit rather than claimed to be new data. A later longer
run requires a concrete resource/cost proposal and separate paid approval.

Do not rerun the already passed derivative/replay gates solely for formality.
Do not promote this previously exposed development monitor to independent
validation. Annual rollout, long holdout, Gate D2 and product acceptance remain
later gates. The evidence supports continuing this route; it does not justify
forecasting eventual accuracy or convergence time from ten updates.
