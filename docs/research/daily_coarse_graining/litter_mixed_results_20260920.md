# Balanced litter supervision: completed matched joint learning

## Outcome

The free recovery completed with exit0 in1718.32s (28min38s), immutable
1391a75, gln01 GPU1. Both arms have20 complete mixed updates, AdamW step4020,
and saved update2/20 checkpoints. Old20 was restored without retraining;
selected2 was restored and18 updates completed. Both calibration bundles
were reused. The preceding90min timeout remains a failed first attempt,
not silently reclassified as a successful run.

Selected litter supervision produces a substantial reduction in the measured
carbon drift: versus matched old20, nonliving-carbon change RMSE falls60.02%
atT7 and43.75% atT30. All16 individual windows improve on that stock metric.
Nevertheless, nonliving-carbon change still has the wrong sign on16/16
windows. Production/biomass effects are mixed; this is useful scientific
progress but not final-model acceptance or proof of long-run stability.

Evidence: server `runtime/outputs/diagnostics/litter-mixed-admission-v1.json`
and `.resume1.log`; local snapshots
`outputs/litter-mixed-admission-v1.completed.json` and `.resume1.log`.
Protocol: `litter_mixed_admission_protocol_20260920.md`.
First attempt: `litter_mixed_partial_results_20260920.md`.

## Physical-unit results

The following are RMSE across the same eight development windows per horizon,
in gC/m2. Flux rows compare cumulative flux over the window; stock rows compare
stock change from the common Teacher initial state. Lower is better.
These are repeatedly consulted development windows, not sealed final tests.

| Horizon/metric | Source4000 | Old20 | Selected20 |
| --- | ---: | ---: | ---: |
| T7 GPP |3.7841|3.8126|3.6040|
| T7 NPP |3.0707|2.9705|2.7720|
| T7 autotrophic respiration |0.8012|0.9442|0.9269|
| T7 aboveground biomass change |2.2991|2.2102|2.0680|
| T7 belowground biomass change |0.7788|0.7756|0.7178|
| T7 canopy carbon change |2.4441|3.0772|3.7926|
| T7 nonliving carbon change |704.61|678.15|271.11|
| T30 GPP |9.3006|4.2911|3.9665|
| T30 NPP |7.4085|2.9544|3.0811|
| T30 autotrophic respiration |2.1309|1.8033|1.3021|
| T30 aboveground biomass change |4.8326|2.2241|2.2319|
| T30 belowground biomass change |2.5673|0.7222|0.8702|
| T30 canopy carbon change |18.9956|12.3142|18.1893|
| T30 nonliving carbon change |4588.68|4439.09|2497.01|

Compared with old20, selected20 improves T30 GPP7.56% and respiration27.79%,
but worsens NPP4.29%, belowground biomass change20.50%, and canopy change
47.71%; aboveground growth is nearly equal (0.35% worse). Relative to the
common source4000, all listed T30 metrics improve, but much of the flux/
biomass improvement is also obtained by old-loss continuation. Do not
attribute the whole source-to-selected gain to litter supervision.

The mean T7 nonliving-carbon change is Teacher+33.49 versus selected-149.75
gC/m2. AtT30 it is Teacher+125.27 versus selected-1033.73. Thus the mean
signed T30 error remains-1159.00gC/m2. Large physical drift survives despite
lower RMSE. These are nonliving carbon stocks collectively, not biomass.

Daily litter-change RMSE (gC/m2/day) also falls markedly:

| TRAIN panel/owner | Source4000 | Old20 | Selected20 |
| --- | ---: | ---: | ---: |
| Fixed8 aboveground |35.3753|35.6517|11.4570|
| Fixed8 belowground |25.6317|23.7008|4.9058|
| Excluded8 aboveground |35.4116|35.6791|11.5609|
| Excluded8 belowground |27.9887|25.9021|5.7060|

This is about68% aboveground and78-79% belowground reduction versus old20.
It is not closure: fixed8 selected mean above/below daily changes remain
-11.4113/-4.7453 versus Teacher+0.0442/+0.1592. Fixed8 GPP/NPP RMSE are
0.16383/0.14590 versus source0.16105/0.13470; these daily quantities have
not uniformly improved from initialization. Excluded8 GPP/NPP improve to
0.15310/0.11148 versus source0.17376/0.14186.

## Technical gates, limitations and runtime

All40 retained update records have zero hard counters, finite full-network
parameters/AdamW and correct optimizer progress. Source/old/selected each
complete all16 recurrent development windows with zero hard counters and
complete metric populations. Both final arms pass the existing normalized
development science screen; source fails it. The runner correctly keeps
`scientific_acceptance=false`. The broad stock-level normalizations can
coexist with large physical change errors, so a screen pass is not evidence
of an adequate carbon-change model. RH Teacher totals are zero on every
window; zero RH error does not demonstrate learned nonzero respiration.
Budget counters measure endpoint bookkeeping, not independent ecosystem
carbon-flux closure. No365-day, long-held-out-point, GateD2 or product gate.

The declared coefficient remained5.221716892123007e-5, selected exclusively
from the two frozen-source TRAIN gradient decompositions before learning.
Each arm saw1280 daily draws plus20T7 and20T30 windows; both share the draws.
All109,464,375 parameters train. This saved4000 fork is diagnostic, not the
paper's independently reproducible fresh-initialization training recipe.

Recovery recurrent graph preparation took244.00/242.83s, versus the first
attempt2094.54/2112.49s. Cache entries existed and preparation was much
shorter, but still not instantaneous. Warm selected updates typically take
12.4-12.8s. One update took26s; the first recovered update includes graph
preparation and is not a warm-throughput measurement. Reused monitor forward
calls take about0.3s per7-day and1.1s per30-day window, excluding compilation,
I/O and host metric summaries. Observed device allocator peak live memory
was17.43GiB, reserved pool25.39GiB; do not confuse reservation with live use.

## Next operation

No task remains running and no new paid job was submitted. Preserve both
arms and their source. Keep Attention/decoder and the admitted loss/AD path.
The next useful experiment is a bounded selected-arm continuation toward100
updates, with fixed20-update physical monitors, before a long training run.
Specify sampling, checkpoints and stop rules first. Do not repeat old-arm
training merely to create another small comparison. Assess whether the large
remaining wrong-sign nonliving drift keeps shrinking and whether canopy,
NPP and biomass tradeoffs persist or worsen; a falling scalar loss or broad
normalized screen pass alone cannot authorize scale-up. Architecture changes
and final fresh-start training remain separate decisions.
