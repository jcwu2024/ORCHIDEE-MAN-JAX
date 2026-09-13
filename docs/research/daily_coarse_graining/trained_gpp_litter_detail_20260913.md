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
