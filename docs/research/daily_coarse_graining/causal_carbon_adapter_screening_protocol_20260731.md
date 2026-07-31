# Experiment C Screening Protocol

Date frozen: 2026-07-31

This document freezes the post-training model-selection screen for Experiment
C before any candidate validation output was available. Training metrics,
validation metrics, and the sealed test split cannot be used to revise this
protocol.

## Evidence Population

The screen evaluates every admitted model-selection window in all three
predeclared slices:

- temporal: known landpoints in held-out years;
- spatial: held-out landpoints in training years;
- joint: held-out landpoints in held-out years.

The sealed test split remains unread. A smoke run may limit the number of
shards per slice, but it is never promotion-eligible. The full report is
classified only after every expected window and predicted day is present.

The evaluated transitions are:

- Day 1 with Teacher state feedback;
- Day 7 with free recursive rollout;
- Day 30 with free recursive rollout.

The existing all-window evaluator, canonical retained STOMATE transition, and
restart-split probe are reused without changing their numerical definitions.
Primary and guard metrics use every lead in their declared horizon. Global
state RMSE and causal-interface error use the terminal lead.

## Relative Gates

Each candidate metric is divided by the matched control metric from the same
slice, horizon, feedback mode, and field. If the control is exactly zero, only
an exactly zero candidate passes.

| Metric | Required candidate/control ratio |
| --- | ---: |
| Day-1 causal interface, fieldwise normalized Huber | `<= 1.02` |
| Day-1 primary state, fieldwise normalized Huber | `<= 0.98` |
| Day-7 primary state, fieldwise normalized Huber | `<= 0.95` |
| Day-30 primary state, fieldwise normalized Huber | `<= 0.95` |
| Global terminal normalized state RMSE | `<= 1.02` |
| Signed flux-bias Huber | `<= 0.90` |
| Signed stock-tendency-bias Huber | `<= 0.90` |
| Litter and DOC guards, fieldwise normalized Huber | `<= 1.05` |

The causal interface contains GPP, 12 maintenance-respiration values, 96
`carbon_32l` values, and 32 `deepC_peat` values, grouped into four
equal-status fields. Primary state fields are NPP, growth respiration,
maintenance respiration, biomass, LAI, `carbon_32l`, and `deepC_peat`. Flux
bias covers NPP and both respiration fields. Stock-tendency bias covers
biomass, LAI, `carbon_32l`, and `deepC_peat`.

There are exactly 165 relative gates:

- 12 interface gates;
- 9 global terminal-state gates;
- 63 primary-state gates;
- 27 flux-bias gates;
- 36 stock-tendency-bias gates;
- 18 litter/DOC guard gates.

All 165 must pass. Aggregate improvement cannot compensate for a failed
field, slice, or horizon.

## Exact And Structural Gates

For both arms, each of the three selected slice/horizon cells must have exact
zero counts for:

- unexpected continuous-state defined-status mismatches;
- unexpected fast-boundary defined-status mismatches;
- discrete-state mismatches;
- nonfinite defined continuous-state values;
- nonfinite defined fast-boundary values;
- negative values in source-declared nonnegative carbon stocks.

This produces 108 exact hard gates. Declared dynamic defined-status behavior
continues to be measured but is not relabeled as an unexpected hard error.

Both arms must also pass:

- bit-exact 30-day versus `15+15` restart-split prediction;
- canonical state as the only cross-day network memory;
- the source-backed retained daily transition.

This produces six structural gates.

## Execution And Stop Rules

The evaluator must verify the frozen training execution, completed reports,
checkpoint hashes, exact update budget, protocol hash, training protocol hash,
model-selection inventory, and resume identity. It assembles each arm from
the same frozen parent and that arm's verified adapter checkpoint.

The execution order is:

1. Complete both 4,096-update training arms.
2. Run one free `gln01` smoke with one shard per slice.
3. If and only if the smoke is complete and valid, run the paid all-sample
   screen on `gnall`.
4. Classify once from all declared gates.

A rejected result stops Experiment C. It does not authorize threshold changes,
validation-weight search, confirmation seeds, 365-day tuning, sealed-test
inspection, or fallback to the rejected Experiment B candidate. A passed
screen authorizes three-seed confirmation; it does not by itself promote a
user-facing daily surrogate.
