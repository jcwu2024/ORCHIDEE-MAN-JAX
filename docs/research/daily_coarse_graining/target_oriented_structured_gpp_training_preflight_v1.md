# Structured GPP Training Preflight V1

Date: 2026-08-29

Status: frozen before free-GPU gradient calibration and one discarded update

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_training_preflight_v1.json`

Contract SHA256:
`6fa0253794938c4400b1e2ac028e818f177ca9c9632edd0fbb0f562520562b62`

## Purpose

This contract opens the smallest training action after the accepted real-v6
admission. It keeps all `93,504,164` retained parent parameters frozen and
differentiates only the `12,879,363` parameters in
`target_oriented.structured_gpp`. It does not reuse the old complete one-day
state/process objective: most of that loss belongs to the frozen parent and is
not repairable by changing GPP alone.

## Direct GPP Warm-Start Objective

Every prepared artifact already contains 208 distinct Teacher-v6 days: 32
one-day samples, 56 Teacher day starts inside eight seven-day windows, and 120
Teacher day starts inside four thirty-day windows. The direct anchor evaluates
all 208 days independently and uses exact sufficient statistics for

```text
0.5 * normalized GPP MSE
  + 0.5 * normalized signed GPP bias squared.
```

Only Teacher-defined, active, non-bare PFT values with positive vegetation
fraction contribute. The frozen train/train `daily.fast_gpp` scale is used.
There is no recursive candidate state in this direct term, so it teaches the
day-local photosynthesis mapping rather than allowing GPP to compensate for a
different frozen error.

## Short-Horizon Objective

After warm start, the same existing rollout machinery will supply seven- and
thirty-day science gradients. Joint optimization uses the independent
component-resolved flux basis
(`GPP`, maintenance respiration, growth respiration, and heterotrophic
respiration) and independent stock basis (AGB, BGB, and nonliving ecosystem
carbon). Autotrophic respiration, NPP, total live biomass, and ecosystem
carbon remain mandatory reported and gated identities, but are not duplicated
as extra optimization weights.

The preflight evaluates one real 7/30-day rollout artifact for finite science
objectives and physical hard gates, but does not compile its gradients. The
joint gradients and their coefficients are calibrated from train-only
artifacts at the warm-start endpoint, when the specialist has first learned
the direct GPP mapping; validation values cannot tune them.

## Free-GPU Preflight

Eight reserved train/train artifacts (`352-357`, `359`, and `360`) measure
direct-GPP specialist gradients. The warm-stage clip threshold is their p90
norm. Artifact `352` also runs 7/30-day forward-only diagnostics. No optimizer
exists during this calibration.

One fresh AdamW proposal then uses artifact `252`, learning rate `4e-6`, and
the calibrated clip threshold. The proposal must decrease the same-artifact
direct GPP objective, change every declared specialist architecture component,
preserve every parent array object, and keep the complete one-day state/budget
hard count at zero.
The proposal and optimizer state are discarded. No checkpoint or recovery is
written.

## Planned Training Boundary

If preflight passes, a separate accepted execution contract may open 250
direct-GPP warm-start updates followed by 250 joint updates. The two stages use
disjoint prepared-artifact ranges and exclude calibration/evaluation groups.
This document does not authorize paid training, validation reads, 365-day
rollout, Gate D2, sealed test, or product acceptance.
