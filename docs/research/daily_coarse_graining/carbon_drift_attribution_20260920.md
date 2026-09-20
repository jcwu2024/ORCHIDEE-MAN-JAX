# Saved-result attribution and bounded endpoint probe

This follows interrupted job15279139. No new training, weights, loss, Teacher
data or architecture change. Inputs are the saved report through update4643,
with complete science monitors through4000. Native AD is not under revision.

## Proven from saved results

At4000, mean T30 nonliving-carbon error is-2147.91gC/m2. Canopy contributes
-16.41; litter+particulate+DOC contribute-2131.50,99.24% of the signed deficit.
The latter already has-179.65gC/m2 error on day1. It accumulates thereafter,
but trajectories alone cannot distinguish repeated one-day bias from feedback.
Fixed-TRAIN T30 shows the same problem: noncanopy error-5113.20 at day30.

Single-day panels start from the identical Teacher state. Consequently endpoint
error equals daily-change error, even where the saved report lacks explicit
change fields. At4000:

| Panel | Samples | Litter bias | DOC bias | Particulate bias |
| --- | ---: | ---: | ---: | ---: |
| Fixed TRAIN daily |64|-68.16|-6.31|+0.21|
| Excluded TRAIN daily |64|-204.21|-8.02|+0.21|

Units gC/m2. These are different samples from the recurrent windows, so they
identify a shared one-day weakness but do not numerically apportion the T30
error into litter/DOC/particulate contributions. Litter dominates these daily
deficits and its MAE equals absolute bias: the errors are systematically low.
The problem is present on TRAIN, not just unseen-site transfer.

T30 NPP mean cumulative residual at4000 is+3.6584 = GPP+4.4193 minus RA+0.7609.
RA residual itself is maintenance-0.7039 plus growth+1.4648. Negative maintenance
error raises NPP, while positive growth-respiration error offsets some excess
GPP. NPP residual is not a separate arithmetic bug: saved per-day identities
close to4.4e-16, cumulative to2.4e-14. Across16 development windows, daily NPP
RMSE improves in8 and worsens in8 from2500 to4000; both fixed recurrent TRAIN
windows also worsen. T30 development mean absolute daily NPP error at4000 is
0.1488 in days1-10 versus0.2864 in days21-30 (2500:0.1645/0.1822).
This is consistent with recurrent drift, not proof of overfitting or LR error.

The fixed daily losses still decline (TRAIN0.01663 ->0.01163; excluded
TRAIN0.03893 ->0.03166). Changing update draws make individual step losses
non-comparable; updates2501-4000 include24 daily losses>10,21>100, and23 raw
gradient norms>1e5. Clipping bounds updates, but saved scalar magnitudes cannot
establish conflicting gradients, faulty samples, or the cause of the rebound.
No retrospective checkpoint selection or new accuracy claim follows.

## What the current supervision and budget check actually mean

Active recurrent `STOCK_KEYS` are AGB/BGB/canopy only
(`daily_operator/recurrent_science_loss.py:32,240`). There is no direct
recurrent litter, DOC or particulate-stock term. They are indirectly supervised
by daily next-state anchors (coefficient0.1; standing-stock scales) and process
labels (overall0.01). Typed transfer labels have an additional0.25 auxiliary
weight inside that term. They are not wholly unsupervised, but the explicit
multi-day objective does not measure their drift.

The current canonical adapter encodes each pool decrease as an external output
and each increase as an external input, with external share1 and zero internal
destination shares (`neural_process_head.py:123-143`). These are endpoint
bookkeeping terms, not identified ecosystem-to-atmosphere/export fluxes.
`conservative_update.py:58-75` checks internal cancellation and discarded input.
For fully defined pools in this representation, the residual is identically
zero regardless of how large the predicted pool removal is. Example: start100,
outgoing fraction0.1, incoming0 ->next90, external bookkeeping output10 and
residual0. It does not assert that respiration or a measured export is10.
`model.py:85-104` reports this pre-tail residual; it is not whole-day ecosystem
closure after the retained tail. Existing zero counters must be described with
this scope. Positivity also cannot rule out excessive depletion.

The active head uses sigmoid(raw) outgoing fractions and mean-centered softplus
incoming amounts (`output_transform.py:42-50`); near-zero fresh logits therefore
start near0.5 outgoing fraction, with an incoming mean of a different scale.
The endpoint update is start*(1-outgoing)+incoming. Large stocks can consequently
have a strong initial depletion tendency. This is a plausible mechanism for
the learning difficulty, not evidence of the trained4000 fraction values.
Do not change the decoder or force ecosystem delta=NPP-RH: deposition, DOC
imports/exports and other boundary terms require source-backed accounting.

## Next bounded forward diagnostic

Read update4000 without changing its weights; use the first8 fixed TRAIN daily
and first8 excluded TRAIN daily samples from the original plan. No gradients,
optimizer, checkpoint writes, Teacher generation or new data selection. Bind
the existing canonical assembler/head/tail with the declared execution policy.
For all five carbon owners retain predicted/Teacher outgoing fractions,
start*outgoing amounts, incoming amounts, pre-tail endpoints and post-tail
states. Preserve active/defined masks, PFT fractions and canopy's land-area
convention. Save small per-sample records incrementally.

The question is whether the one-day litter deficit is excessive outgoing,
insufficient incoming, or a discrepancy introduced by the retained tail.
This is a diagnostic subset, not a representative performance or scientific
acceptance panel. If insufficient, extend existing component extraction to the
same saved T7/T30 windows, rather than training speculatively.

Use one idle free gln01 GPU,60min ceiling, separate diagnostic checkout and
output. No paid submission authorized by this plan. A production-runtime
forward is required; local synthetic tests do not claim actual-model success.

Launched on free gln01 GPU0 after both GPUs were observed idle, immutable
`b2092a4` in `runtime/checkouts/mixed-carbon-attribution-v1`. WrapperPID82679,
timeout82694, Python82727;60min maximum. Actual target-shell dry-run and four
local analysis/endpoint tests pass. Output/log:
`runtime/outputs/diagnostics/mixed-carbon-endpoints-4000-v1.{json,log}`.
The diagnostic subsequently completed exit0,16/16samples in652.59s. Excessive
litter outgoing is confirmed; the tail does not add the deficit on these
inputs. See `carbon_endpoint_results_20260920.md`; no process remains active.

After attribution, specify one focused corrective candidate (e.g. physical
litter-change supervision if the drain is weakly penalized), preserve Attention,
sampling and scientific outputs, then verify the changed loss/gradient and a
bounded number of optimizer updates before any paid learning experiment.
Do not combine decoder, LR, data and loss changes into an unidentifiable trial.

## Artifacts and checks

`scripts/hpc/analyze_mixed_carbon_drift.py` reconstructs day-zero changes and
decomposes saved trajectories. Outputs under
`outputs/mixed-learning-review-pool-20260920/carbon-drift.{json,md,png}`.
Two local identity/missing-day tests pass; Ruff passes and plot was inspected.
Source attribution independently checked by Sol/high; paired productivity
analysis by Luna/high. Graph tools were unavailable, so exact source reads and
bounded rg replaced graph lookup without a graph-completeness claim.
