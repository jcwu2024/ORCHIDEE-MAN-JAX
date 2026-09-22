# Shared Transformer: first 100-update learning results

Free gln01 GPU1 wrapper87042 at4408daf completed100 updates, exit0, in
5824.47seconds (97.07minutes). It restored the declared fresh seed's update0
after the monitor API repair, not an exploratory trained checkpoint. The
architecture, losses, sampling and optimizer match `shared_learning_v1.json`.
All four fixed panels pass physical gates at0/20/100; there is no OOM or
nonfinite optimizer failure. Scientific accuracy and long-term acceptance
remain open. The previous failed wrapper83621 ran zero updates.

## Fixed held-out physical results

The table pools the same24 T30 windows (8 temporal,8 spatial,8 joint) at all
three milestones. Flux errors are30-day cumulative RMSE; stock errors are
day30 minus common day0 stock-change RMSE. All units are gC/m2. Pooled RMSE
is recomputed from physical per-window curves, not averaged across RMSEs.
These are development validation windows, not sealed tests. Improvement is
relative to untrained initialization, not an accepted strong baseline.

| Quantity | Update0 | Update20 | Update100 | Reduction0 to100 |
| --- | ---: | ---: | ---: | ---: |
| GPP |77.046|59.455|30.332|60.63%|
| NPP |74.584|39.279|28.169|62.23%|
| Autotrophic respiration |8.717|21.763|7.504|13.91%|
| AGB carbon change |51.864|28.771|20.005|61.43%|
| BGB carbon change |22.713|10.734|8.135|64.18%|
| Canopy organic carbon change |163.695|232.935|182.340|-11.39%|
| DOC change |75.89|17.37|25.69|66.1%|
| Aboveground litter change |6.02|5.76|5.77|4.2%|
| Belowground litter change |6.63|8.60|7.68|-15.7%|

The seven-day pooled GPP/NPP/AGB/BGB RMSE changes22.260/21.659/15.469/6.548
to11.137/8.470/6.388/2.166. T7 respiration worsens3.074 to3.530.
Every held-out split improves GPP/NPP/AGB/BGB relative to update0. T30 GPP
at100 is10.448 joint,30.091 spatial,41.779 temporal: substantial heterogeneity
remains. Do not generalize the favorable joint subset to the whole domain.

The fixed TRAIN-fit T30 GPP/NPP/AGB/BGB errors also improve:
53.269/47.970/35.904/12.132 to22.585/20.053/13.441/6.594. Learning is visible
both within the fixed fit panel and outside the actual parameter-update data.
It is not monotonic: TRAIN-fit T7 NPP/AGB and several other groups are worse
at100 than20. Do not select20 post hoc;100 remains the declared endpoint.

## Why this is not sufficient accuracy

For held-out T30, Teacher RMS cumulative GPP/NPP is96.55/39.46; final errors
are31.42%/71.38% of those magnitudes. Teacher RMS AGB/BGB **change** is only
16.34/5.84, compared with final errors20.01/8.13. Thus biomass-change errors
are still larger than the typical true change, even though their errors have
fallen substantially. Small percentages of a large standing biomass stock
would conceal this weakness.

Signed mean errors change from underprediction at0 to overprediction at100:
T30 GPP -54.47 to+12.69, NPP -56.24 to+16.51, AGB -38.55 to+11.83,
BGB -17.01 to+4.58. Long-run drift is not solved. Respiration's components
partly cancel: maintenance bias -9.88 and growth bias+6.05 combine toRA -3.82;
separate maintenance/growth RMSE remain11.72/9.71. RH is zero in both Teacher
and predictions throughout these panels, so zero error is not evidence of
learning a nonzero RH response. Canopy carbon and belowground litter have
not improved in held-out T30; DOC at100 regresses relative to20.

## Runtime and optimization

Actual allocator peak7.999GiB, distinct from the25.391GiB reserved pool.
Median warm update wall time29.90seconds, of which28.62seconds is measured
gradient/update work. Each update uses32 sequential daily gradients and one
native T7/T30 gradient; this is not the previous architecture's dual-GPU or
microbatched throughput. Three reverse compilation stages total1170.52seconds
(19.51minutes), substantially below the earlier cold admission but still
material. Initial64-window monitoring, including first forward compilation,
takes986.23seconds; later full panels take about172seconds.

All100 updates clip gradients; median scale0.1456. This alone does not diagnose
bad gradients or inadequate learning and does not justify changing clip1.0.
Different batches have different losses. The scientific interpretation above
uses fixed windows rather than comparing first/last stochastic batch losses.

## Next operation

Keep the shared architecture and existing joint objective. Prepare a bounded
extension with more distinct existing TRAIN artifacts and continued full-network
updates from the declared100 endpoint, retaining these same physical panels.
Freeze the next sampling order, endpoint, checkpoints and budget before launch.
This extension must be documented as an adaptive follow-up to the100-step
pilot, not retrospectively called a preregistered final paper recipe. Its aim
is to determine whether the remaining changes/flux biases shrink with more
learning and coverage; no guarantee follows from this pilot.

Do not immediately add specialized heads, retune losses from one panel, or
restart AD infrastructure work. Inspect existing canopy/litter/component
curves alongside the continued learning; these quantities are already in the
objective.365-day evaluation, long-site holdouts, GateD2 and final reproducible
training remain outstanding. No new GPU/paid task was launched during result
collection.

## Evidence

- Remote report/checkpoints: `runtime/outputs/diagnostics/shared-learning-4408daf-gpu*`.
- Local raw report: `outputs/shared-learning-4408daf-gpu.json`.
- Recomputed summary: `outputs/shared-learning-4408daf-summary.json`.
- Reproduce with `python scripts/hpc/summarize_shared_learning.py REPORT OUTPUT`.
  This reads saved curves only, with no model inference or Teacher regeneration.
