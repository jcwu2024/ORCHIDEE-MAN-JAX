# Corrected Attention continuation v2: sustained learning and independent transfer

Job `14984032` completed normally (`0:0`) on 2026-09-10, from 04:43:19 to
09:51:54 Asia/Shanghai, elapsed 05:08:35. All 20 updates were retained, ending
at logical update 34. Additive duration-based cost estimate: CNY 14.19.
Execution was `b692915`; no training or model change was made after submission.

All 20 local and cumulative development-monitor gates pass. Direct-GPP loss
falls 16.8483% from update 14, and the required science MSE/bias macro falls
12.5737%. All eight direct monitor artifacts and all 18 required science
key/horizon losses improve cumulatively. The final five updates still reduce
direct loss by 4.7658% and science loss by 3.3567%. This is no observed plateau
over the tested interval, not a convergence forecast.

All 1,920 streamed-gradient score comparisons pass. Daily reverse compiles once.
Median hot train-only update time is 753.59 seconds, excluding its subsequent
fixed-monitor evaluation. Full/compact reporting took about 43.65 seconds before
the final write across the full job. The data schedule consumed 800 artifact
exposures, 678 distinct existing artifact IDs. This does not count unique days
or unique point-years. Independent validation remains excluded from directions
and scale selection.

## Independent validation

All 24 paired windows retain identical evaluation counts and zero physical hard
failures. The existing acceptance macro normalized RMSE decreases from 0.369501
to 0.352919, a 4.4875% reduction. The frozen relative transfer gate passes.

| Split | Horizon | Macro relative change |
| --- | ---: | ---: |
| Temporal | 7 | -5.4635% |
| Temporal | 30 | -2.8865% |
| Spatial | 7 | -8.0598% |
| Spatial | 30 | -5.5365% |
| Joint | 7 | +0.1962% |
| Joint | 30 | -8.5054% |

The joint seven-day cell slightly regresses within the frozen 2% cap. Do not
claim uniform improvement or final model acceptance. Absolute GPP/NPP and
growth/autotrophic respiration still fail. Fixed-monitor and independent macro
percentages use different loss definitions and cannot be compared directly.

Recovery is
`runtime/outputs/training/conditioned-snow-free-continuation-v2-14984032/recovery.pkl`.
Local report and recomputed summary:
`outputs/conditioned-continuation-v2-14984032.json` and `.summary.json`.
The accepted checkpoint supports further exploration, not final reproducible
training acceptance under the user's 2026-09-09 principle.

## Physical-unit interpretation requested by the user

The user requested direct evidence of productivity, respiration and biomass fit,
rather than loss-reduction percentages alone. An inference-only export at
`3d94543` completed **all** existing independent windows for update 14 and
34, retaining daily Teacher and model flux/stock curves. It uses the unchanged
canonical fresh evaluator with an optional trajectory return, and verifies its
aggregate acceptance metrics against the completed run before publishing plots.
All 468 RMSE/bias/count comparisons pass. Physical hard failures are zero for
both states; each has the same 211 separately reported declared-dynamic masks.
No training, new panel selection, Teacher production, or paid job is involved.

The first free GPU-1 export stopped after four windows with CUDA graph
instantiation OOM. The shared GPU had another process consuming about 7.3 GiB
when inspected after failure. The unchanged exporter completed on idle GPU 0
with `SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.55` in 323.57 seconds,
reusing compilation caches. No scientific model or dataset change was needed.
Completed remote report:
`runtime/outputs/diagnostics/conditioned-science-curves-3d94543-gpu0/report.json`.
Local curves: `outputs/conditioned-science-curves-14984032.json`.

Completed displays: physical-unit window-average flux and endpoint-stock fits,
window stock changes, and daily examples chosen by a disclosed GPP-error rank,
with every thirty-day GPP trajectory available. Teacher agreement is distinct
from field-observation validation; good short-window stock levels do not prove
correct biomass growth or long-term drift.

All physical metrics below weight the 24 windows equally. Fluxes are window
daily means in gC m-2 land day-1. These are sample summaries, not global area-
weighted estimates or the acceptance macro.

| Variable | Teacher mean | Update-34 mean | Physical RMSE |
| --- | ---: | ---: | ---: |
| GPP | 3.5076 | 4.5303 | 2.2432 |
| NPP | 1.2164 | 1.9404 | 1.6053 |
| Maintenance respiration | 1.7829 | 1.8248 | 0.1437 |
| Growth respiration | 0.5083 | 0.7651 | 0.6000 |
| Autotrophic respiration | 2.2912 | 2.5899 | 0.6547 |
| Heterotrophic respiration | 0 | 0 | 0 |

The zero heterotrophic result is not evidence of fitting nonzero heterotrophic
respiration. GPP/NPP have substantial over- and underprediction, not a uniform
scale error. Daily curves include missed near-zero GPP periods. Do not infer
the cause or change architecture from these plots alone.

Endpoint AGB/BGB RMSEs are 21.75/13.35 gC m-2 against mean Teacher stocks of
3452.74/1469.97. But changes from each trajectory's first window day to its
last are poorly fitted: mean Teacher/model AGB changes are 6.03/14.82 gC m-2,
and BGB changes are 1.68/8.93. Change RMSEs are 21.25/13.22. The mixed 7/30-day
change means are descriptive only, not annual growth rates.

For the fixed first seven-day window (235.0-085.0, 2005), endpoint AGB is
2542.64 Teacher versus 2547.42 model, but day-1-to-day-7 gain is 2.23 versus
6.42 gC m-2. This makes the limitation of short-horizon stock-level acceptance
concrete. The model is improving, but a successful long-term biomass/productivity
surrogate is not yet demonstrated.

Figures and Chinese reading guide:
`outputs/conditioned-science-fit-14984032/README.md`. Regenerate with:

```text
conda run -n ORCJAX python -m scripts.dev.plot_conditioned_science_fit outputs/conditioned-science-curves-14984032.json outputs/conditioned-science-fit-14984032
```

Future progress reports should pair loss changes with actual Teacher/model
flux fits and biomass-change curves, declaring sample coverage, units and
the distinction between stock-level and growth accuracy.
