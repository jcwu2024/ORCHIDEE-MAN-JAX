# Target-Oriented Coupled-GPP Objective Calibration V1

Date: 2026-09-01

Status: frozen before zero-update train-only GPU calibration

## Purpose

The accepted coupled candidate controls only `daily.fast_gpp`, but GPP errors
reach 41 next-day slow-carbon coordinates in 16 Markov fields. This contract
turns that accepted dependency inventory into a field-grouped state loss and
calibrates it jointly with direct GPP and 7/30-day science losses before any
optimizer exists.

## State Anchor

The state anchor reads only the accepted 41 global coordinates from
`next_state.other`. It does not also read the duplicated `carbon` inventory
view. Each field receives its own normalized MSE and signed-bias-squared loss;
coordinates are averaged within a field and active fields are then averaged
equally. Thus the 11 reachable `biomass` coordinates cannot outweigh a
single-coordinate field merely by count.

All 41 exact-nonzero coordinates remain included. The reachability report's
normalized `1e-12` threshold is descriptive and does not remove the two
smaller valid dependencies. Scales come from the existing v6 train/train
next-state statistics with the already accepted unresolved-scale amendment.

## Four Gradient Sources

The calibration reuses prepared train/train artifacts `352-357`, `359`, and
`360`. Every artifact contributes:

1. direct GPP over 208 Teacher days;
2. the reachable one-day state anchor over 32 Teacher transitions;
3. component-resolved science loss over eight recursive 7-day windows; and
4. component-resolved science loss over four recursive 30-day windows.

The short-horizon flux basis is GPP, maintenance respiration, growth
respiration, and heterotrophic respiration. The independent stock basis is
AGB, BGB, and nonliving ecosystem carbon. Derived NPP, total autotrophic
respiration, total live biomass, and ecosystem carbon remain reported physical
metrics and later absolute gates.

## Coefficient Rule

Direct GPP is the fixed reference with coefficient `1.0`. Before seeing the
new gradients, the target gradient-norm ratios are frozen as:

- reachable state anchor: `0.5`;
- 7-day science: `0.25`; and
- 30-day science: `0.25`.

The three non-reference coefficients are the median across eight artifacts of
`target ratio * direct gradient norm / source gradient norm`, bounded to
`[1e-4, 100]`. They are frozen once and are not recalibrated during training.
The p90 of the eight calibrated combined-gradient norms supplies the later
clip candidate. Gradient cosines are reported but do not alter coefficients.

This allocates a total target gradient budget equal to direct GPP across the
cross-day state and two rollout sources. It prevents direct GPP from being
discarded while giving the newly identified feedback path material weight.

## Parameter And Execution Boundary

Only `target_oriented.structured_gpp` and
`target_oriented.coupled_gpp_carbon_feedback` are differentiated, for exactly
15,960,195 parameters. The other 93,504,164 parameters remain frozen.

Every source and combined gradient must be finite, nonzero, and cover every
trainable leaf. Every objective must be finite and positive, all 16 reachable
fields must be active in every artifact, and all one-day/7-day/30-day physical
hard gates must pass. The structured checkpoint and retained parameters must
remain unchanged.

The execution creates no optimizer, performs no update, and writes no
checkpoint. It does not regenerate v6, use v5, read fresh or sealed targets,
reconstruct 48 forcing records, or execute a 48-step state scan. Passing this
calibration opens only a separately frozen discarded four-source optimizer
proposal preflight.

The machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_objective_calibration_v1.json`,
canonical SHA256
`20986e2d4fdefdddf2c89bd5a8ab83f64c680804d5b2b5b497c011bc34dcb825`.
