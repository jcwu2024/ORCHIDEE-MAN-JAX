# GPP litter-state feedback: result and fixed leaf follow-up

## Completed group intervention

At a136fda, the free GPU0 run completed all16 windows/296 days in905.16s,
exit0. Controls reproduce prior output; all32 arms have zero recorded hard
counts. Report `trained-gpp-state-interventions-v1.json` is saved under local
outputs/ and remote runtime/outputs/diagnostics/. This is input mediation at
fixed weights, not learned-model improvement or unique physical causality.

For T30, litter_metadata replacement removes95.314% of the free-minus-reset
GPP MSE gap; reciprocal replacement restores78.301%. Rescue is positive in7/8
windows, damage in8/8. Daily RMSE goes0.87144 to0.21293, and aggregate bias
-31.58% to+1.34%. T7 corroborates (88.56% rescue), on a different panel.
DOC has43.46% rescue but only6.76% damage; its possible interaction remains
relevant despite missing the bidirectional classification threshold.
Canopy organic-carbon rescue is0.126%, so its large inventory loss should be
treated separately from the main observed same-day GPP feedback pathway.

## Follow-up frozen before execution

Use the same checkpoint,16 windows, archived starts, forcing, canonical one-day
binding and reciprocal controls. No hybrid outputs are propagated. Keep the
same32-arm graph (15 group masks plus free/Teacher controls), now with:

- nine individual litter_metadata leaves: litterpart; lignin_struc,
  lignin_struc_above, lignin_struc_below; dead_leaves; four fuel pools;
- full litter_metadata and DOC separately to reproduce their group contrasts;
- their union, the lignin family, the fuel family, and the complement of
  litter_metadata plus DOC to expose interactions and remaining effects.

These follow-up masks intentionally overlap; they are not a state partition.
Report continuous rescue/damage fractions and window signs for every contrast
using the previous fixed diagnostic rule. The all-fields and DOC controls are
also compared to their saved numerical effects on review. No acceptance is
claimed from a local hybrid-state intervention alone. Do not remove fields
from model inputs before understanding their intended physical/network route.

The runner's optional `--grouping litter_detail` changes only host-built input
masks; same32-arm array shape, one-day graph, losses/AD and model are unchanged.
Reuse the a136fda GPU runtime preflight72.1s; actual new launcher expansion and
real-array CPU shapes are checked once. Free GPU cap40m; no paid submission,
Teacher generation, parameter update or checkpoint. Independent source audit
and canopy stock-change loss design proceed concurrently.

Next deliverable: identify leaf/family, its normalization and GPP context path,
then specify the smallest supported training/input correction. The stock-change
loss is separately specified using existing train/train statistics and needs
training-only gradient/update calibration before becoming a training method.

## Execution record

At a952bb1, seven targeted local tests and Ruff pass. Actual test02 shell
expansion and all16 real-array shapes pass (183.1s). Free GPU0 was idle;
the40m forward-only run is launched as PID18414 from
`runtime/checkouts/trained-feedback-v1`, which stays fixed while running.
Log/report prefix: `runtime/outputs/diagnostics/trained-gpp-litter-detail-v1`.
The numerical one-day32-arm graph is unchanged, so its completed real-GPU
preflight is reused. No paid allocation or trainable update is involved.

## Completed leaf result

The a952bb1 run finished normally in646.2818s, all16 windows/296 days,
exit0. PID18414 is absent. All controls reproduce, all32 arms' hard counts
are zero. Full litter and DOC contrasts reproduce the preceding group run.
`outputs/trained-gpp-litter-detail-v1.json` is the durable raw report.

T30 (eight windows,240 days):

| Replacement | Excess GPP MSE removed | Reciprocal damage / excess MSE | Positive rescue/damage windows |
| --- | ---: | ---: | --- |
| litterpart alone | 95.1045% | 78.6935% | 7/8;8/8 |
| Full litter metadata | 95.3142% | 78.3014% | 7/8;8/8 |
| DOC alone | 43.4600% | 6.7641% | 5/8;4/8 |
| Full litter metadata + DOC | 98.1422% | 94.2508% | 8/8;8/8 |
| Lignin family | 0.1964% | 0.3633% | 5/8;6/8 |
| Fuel family | 0.9270% | 0.0646% | 3/8;4/8 |

Here excess MSE is free MSE minus daily-Teacher-reset MSE. It is not the
fraction of all GPP error, a variance decomposition or an additive causal
allocation. Litterpart rescue changes RMSE0.871443->0.216585 gC/m2/day and
aggregate bias-31.5805%->+1.1335%; reciprocal damage RMSE0.774460 and
bias-19.8282%. T7 corroborates litterpart rescue87.7969% and damage112.7535%
on its different panel. DOC interaction is horizon/panel dependent: adding
DOC to litter rescue helps the T30 aggregate but slightly worsens T7.

The archived-state audit resolves the training/input mismatch:

- Actual neural input normalization comes from process statistics, not the
  output loss scales. Its3854 mean/scale coordinates equal the corresponding
  base state statistics. The fresh DOC tail does not affect litterpart.
- PFT14 litterpart input mean0.99993773, scale0.0078909125. T30 Teacher
  starts/endpoints are1 for both litter coordinates; predicted reset endpoints
  average0.66768/0.67572. Recurrent input extrema reach81.35 standard deviations.
- The actual fast-output normalization is pooled across compact PFT slots:
  mean0.5,scale0.5. Its absolute unit-interval process loss uses scale1;
  next-state litterpart scales are also1. Thus a modestly sized native-unit
  endpoint loss permits a severe displacement in the next day's input units.
  This is a diagnosed loss/recurrence mismatch, not an arithmetic bug in AD.

The script `scripts/dev/summarize_trained_litterpart_feedback.py` reproduces
these readings from existing archives/statistics with no model execution.
Its terminal-state statistics are not a new raw typed fast-label readback.
The independent input/source route audit is
`litter_gpp_input_route_audit_20260913.md`.

Next: implement separately reported litterpart feedback-scale supervision
alongside canopy stock-change supervision, keep the Attention and normalization
unchanged, calibrate gradient balance on training-only fresh samples and verify
actual updates before another learning run. The new combined proposal is
`feedback_and_canopy_loss_amendment_20260913.md`. No hybrid Teacher replacement
enters inference or a reported model. These are train-domain day-disjoint
diagnostics, not independent-site/year or sealed-test acceptance.
