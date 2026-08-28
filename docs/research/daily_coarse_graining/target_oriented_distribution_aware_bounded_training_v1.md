# Distribution-Aware Bounded Training V1

Date: 2026-08-28

Status: frozen; free one-update admission accepted; paid stage awaits separate
resource approval

Machine contract:
`manifests/coarse_graining/target_oriented_distribution_aware_bounded_training_v1.json`

Contract SHA256:
`0c60924ee12d0ea555b082f76990ca55e34f86f67effb052f5eec44c01210051`

## Purpose

This is the first bounded fit of the accepted 24-coordinate distribution-aware
maintenance adapter. It retains the update-1000 coordinate-aligned Attention
parent and trains only the new `9,750,529`-parameter adapter. It is intended to
test the source-informed mechanism against absolute short-horizon science, not
to establish a small relative win over a weak baseline.

## Data And Objective

The bounded stage uses 250 existing Teacher-v6 prepared train/train artifacts
selected in ascending order from `252..511` after excluding calibration and
evaluation updates `352..360` and `465`, one per optimizer update. Every effective batch retains 32 one-day
samples, eight 7-day windows, four 30-day windows, and 208 Teacher days. The
eight calibration artifacts and eight evaluation groups are disjoint from
training and from each other. Teacher data is not regenerated; historical v5,
validation values, and sealed test remain prohibited.

Every update combines two gradients over exactly the same adapter subtree:

```text
combined_gradient = free_rollout_gradient + 0.1 * teacher_guard_gradient
```

The free term is the accepted component-resolved one-day/process plus 7/30-day
science objective. The guard is the equal macro mean of Teacher-day-start
maintenance respiration, growth respiration, and NPP at 7/30 days. Its
coefficient `0.1` is immutable evidence from report SHA256
`bb55388f...addf7a474`, not a validation-tuned hyperparameter.

## Optimizer

Use a fresh AdamW state with beta values `0.9/0.95`, weight decay `0.01`, 25
warmup updates, peak learning rate `1e-4`, and cosine decay to `3e-5` at update
250. The global-gradient clip threshold is `2.692620468992339`. It is the p90
of the eight calibration triangle upper bounds
`free_norm + 0.1 * guard_norm`, so no additional calibration run is needed.

Before paid training, one ephemeral artifact-252 update must pass on free
`gln01`. It must update all adapter leaves, preserve every parent parameter by
identity, remain finite, and have zero free and guard physical hard failures
before and after. It writes no checkpoint or recovery.

That admission passed at commit `5ea0033`; report SHA256 is
`a32b2c17...c7734e`. The combined gradient norm was `2.0161371`, no clipping
occurred, all adapter leaves changed, and all retained parent parameters were
exactly unchanged.

## Evaluation And Decision

The paid stage, if separately authorized, evaluates the candidate before and
after update 250 on train/train groups `251, 358, 465, 572, 679, 786, 893,
1000`. Promotion requires all of:

- at least 20% median maintenance-score reduction at both horizons;
- no median regression in GPP, growth and autotrophic respiration, NPP, AGB,
  BGB, total live biomass, or ecosystem carbon;
- all absolute target-oriented keys passing after evaluated-value-weighted
  aggregation across the eight groups at both 7 and 30 days;
- zero physical hard failures.

Relative improvement alone is not promotion. A relative mechanism pass with
an absolute-gate failure can only justify a bounded continuation review. A
relative mechanism failure closes unchanged adapter-only continuation.

The contract authorizes no paid submission by itself. Validation, 365-day
rollout, long held points, Gate D2, sealed test, and product acceptance remain
closed until the short-horizon science gate passes.
