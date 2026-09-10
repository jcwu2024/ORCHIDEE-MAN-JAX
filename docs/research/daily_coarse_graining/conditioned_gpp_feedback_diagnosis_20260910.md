# Update-34 direct GPP versus recurrent error

The diagnosis reuses job 14984032 and its completed 24-window trajectory export.
No model execution, training, new Teacher production, or new sample selection
is required. The executable analysis is `scripts/dev/analyze_conditioned_gpp_feedback.py`.
Outputs are `outputs/conditioned-gpp-feedback-14984032/analysis.json` and
`gpp_feedback.png`.

## Comparable measurements

The original direct-GPP report scores individual days, whereas the science
acceptance GPP report scores window sums normalized by horizon times a fixed
scale. Those RMSEs cannot be compared directly. Recover the already-frozen
GPP scale from physical window-mean RMSE divided by normalized acceptance
RMSE, reproducing the signed bias and the same scale across all 12 paired
stratum/horizon/version cells. Then compare direct and free-rollout **daily**
GPP on the same 444 days in gC m-2 land day-1. All days receive equal weight;
this differs from the earlier plot's equally weighted 24 window means.

| Checkpoint | Teacher-state daily RMSE | Free-rollout daily RMSE | Teacher-state bias | Free-rollout bias |
| --- | ---: | ---: | ---: | ---: |
| Update 14 | 1.89265 | 2.43412 | +1.08092 | +1.45736 |
| Update 34 | 1.74793 | 2.33039 | +0.89865 | +1.27221 |

Direct RMSE improves about 7.65%; free-rollout RMSE improves about 4.26%.
Both direct prediction and recurrence remain material problems. This comparison
does not identify the individual state/process responsible for recurrence, and
RMSE differences are not an additive causal error decomposition.

| Update-34 stratum | Days/window | Teacher-state RMSE | Free-rollout RMSE |
| --- | ---: | ---: | ---: |
| Temporal | 7 | 2.23938 | 2.27882 |
| Temporal | 30 | 2.19912 | 2.59701 |
| Spatial | 7 | 1.56889 | 1.67231 |
| Spatial | 30 | 1.05563 | 2.75792 |
| Joint | 7 | 2.86306 | 2.87967 |
| Joint | 30 | 1.30263 | 1.39424 |

Each cell contains four windows. The spatial/30 discrepancy is substantial,
but does not justify selecting or training on those held-out targets. Horizon
groups use different windows, so their contrast is not a controlled 7-to-30-day
growth-of-error experiment.

## Training side and coverage

Actual training records cover every runtime year **1961-2004**, with 800 artifact
exposures and 678 distinct artifact IDs. Each artifact contains multiple sites;
artifact count is not location count. Fixed monitor artifact IDs are excluded
from this stage's updates, but come from train/train splits and are used for
development decisions. They are not the independent validation panel.

The fixed monitor's direct loss decreases from 1.03024 to 0.85666. On the actual
40 training packages used at update 34, the update lowers direct loss from
0.63672 to 0.63116. Residual training-side error remains. Different updates use
different packages, so their raw loss sequence cannot establish a fixed-set
learning curve or a train/validation generalization gap. Training loss is
0.5*(active-PFT normalized MSE + squared mean bias); it is not the land-area
validation daily RMSE. Evidence supports continued learning, not a proof that
undertraining is the sole cause or that longer training must succeed.

## Carbon accumulation and net biomass change

Use only the 12 thirty-day windows here. Units are gC m-2 land over 30 days.

| Quantity | Teacher mean | Update-14 mean | Update-34 mean | RMSE 14 -> 34 |
| --- | ---: | ---: | ---: | ---: |
| GPP sum | 91.2187 | 139.6393 | 133.9414 | 69.3475 -> 65.9055 |
| NPP sum | 23.7668 | 56.9508 | 52.8802 | 47.8868 -> 45.5019 |
| Maintenance respiration sum | 57.1171 | 60.1675 | 59.9702 | 5.3387 -> 5.1488 |
| Growth respiration sum | 10.3349 | 22.5210 | 21.0911 | 18.1095 -> 17.2434 |
| Autotrophic respiration sum | 67.4519 | 82.6885 | 81.0613 | 21.7885 -> 20.7640 |

With a common day-zero state, model-minus-Teacher endpoint stock equals error
in the full-window net stock change: `(M_H-S_0)-(T_H-S_0)=M_H-T_H`.
AGB net-change RMSE improves 30.5028 -> 29.4153, and BGB 19.8301 -> 18.6226.
Their current biases remain +15.6678/+13.2398. Initial stock values were not
exported, so net-change levels cannot be reconstructed from these files; only
their error can. Earlier first-record-to-last-record growth plots omit day 1.

Changing a stock loss's label from endpoint to net change, with the same initial
state and denominator, would be algebraically identical and would not improve
training. Growth-sensitive rescaling would be a real loss change requiring its
own train-only evidence. No such change is made from this small validation panel.

## Decision

Continue the admitted Attention/GPP-plus-carbon-feedback method for a bounded
40-update stage (35-74), preserving the full 7/30-day objective and existing
weights/scales. Do not switch to direct-only training, change the backbone,
or reweight losses merely from these plots. Review actual accumulated carbon
and net-growth errors alongside the existing relative and absolute metrics.
The reused 24-window panel is now development validation, not a new independent
test of a method designed without seeing these results. No sealed test or
365-day gate is opened. See `conditioned_continuation_v3_plan_20260910.md`.
