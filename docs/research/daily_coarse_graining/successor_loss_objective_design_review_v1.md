# Successor Loss-Objective Design Review v1

Status: **mechanism and train-only scales accepted; zero-update gradient
preflight next; no training authorized**.

Date: 2026-08-14

The user approved the mechanism after review. The objective foundation and
scale fitting implementation are complete, and the train-only scale asset
passed on free `test01`. See
[`successor_named_biomass_tendency_scale_acceptance_20260814.md`](successor_named_biomass_tendency_scale_acceptance_20260814.md).

The subsequent zero-update real-v6 gradient preflight also passed. It freezes
one shared AGB/BGB tendency coefficient of `0.05440473497107039`; the weighted
AGB/NPP and BGB/NPP median gradient ratios are `2.2888` and `0.4369`, with
finite complete candidate gradients and zero hard failures. Training remains
closed pending review of the maintenance guard and validation-cell majority.
See
[`successor_named_biomass_gradient_preflight_20260814.md`](successor_named_biomass_gradient_preflight_20260814.md).

## Evidence To Explain

At the accepted update-1000 Attention state:

- maintenance gradients have stable negative cosine with NPP, AGB, and total
  live biomass;
- four-point mixed sampling mitigates none of those conflicts;
- frozen terminal-stock gradients are four to six orders smaller than NPP and
  maintenance gradients;
- nevertheless, 100 updates with mixed sampling improve Attention validation
  by 4.35% over matched single sampling and 0.73% over update 1000.

A useful next objective must therefore strengthen named biomass learning and
state explicitly how maintenance conflict is handled. Merely changing batch
composition or multiplying the existing terminal-stock scalar cannot do both.

## Rejected Shortcuts

1. Do not tune weights against the 48-window validation panel.
2. Do not multiply terminal-stock loss by an arbitrary `1e4`-`1e6`; this may
   repair magnitude but leaves the measured negative gradient directions.
3. Do not reactivate the historical `process_increment_v2`. It supervised all
   3,854 anonymous state increments in a rejected direct-state model and
   failed every required seen-condition improvement gate.
4. Do not begin with PCGrad or another multi-gradient optimizer. Five separate
   backward passes per update would multiply cost before a cheaper scientific
   objective has been tested.
5. Do not remove the one-day named-process anchor or weaken state, budget,
   restart, and mask hard gates.

## Proposed Mechanism

Call the proposed candidate `named_biomass_tendency_v1`.

It keeps the current process head, conservative updater, retained exact tail,
one-day anchor, cumulative GPP/NPP/growth/heterotrophic objectives, optimizer,
and data boundary. It changes only long-horizon stock and maintenance use:

```text
delta_AGB(H) = AGB_after_H_days - AGB_day_start
delta_BGB(H) = BGB_after_H_days - BGB_day_start
```

The candidate compares predicted and Teacher `delta_AGB` and `delta_BGB` at
`H = 1, 7, 30`, using horizon- and field-specific robust scales computed only
from train/train v6 trajectories. AGB and BGB are the exact source-partitioned
quantities already derived by the frozen long-horizon objective. Total live
biomass remains an exact `AGB + BGB` reported metric and guard, not a third
redundant optimized stock term.

Long-horizon maintenance respiration is removed from the jointly summed
rollout gradient and becomes a frozen no-regression guard. Maintenance remains
directly supervised by the unchanged one-day named-process anchor. This is an
explicit priority decision: the research goal is long-run productivity and
biomass, while maintenance is not allowed to improve by driving those primary
outputs away from the Teacher.

This differs materially from `process_increment_v2`: it does not predict or
weight anonymous state deltas, does not touch all state columns, and does not
change stock ownership. It supervises two named scientific tendencies after
the accepted conservative transition and exact retained processes.

## Matched Controls

Use a staged comparison so a failed objective does not trigger unnecessary
architecture training:

1. **Objective control**: Attention with the current frozen loss and mixed
   sampling.
2. **Objective candidate**: the identical update-1000 Attention parameters,
   mixed samples, optimizer, update count, and validation panel, changing only
   to `named_biomass_tendency_v1`.
3. **Architecture control, conditional**: graph MLP with the candidate
   objective, run only if the Attention objective candidate passes. This arm
   is required before claiming that Attention itself contributes beyond the
   repaired objective.

The prior single-sampling arm is not repeated in this loss experiment. Its
role was already answered by the equal-sample A/B; repeating it would not
isolate the objective change.

## Preparation Gate

Before any update:

- derive AGB/BGB tendency scales from train/train data only and bind their
  source manifest, field definitions, counts, and SHA256;
- prove exact zero loss for Teacher-versus-Teacher trajectories;
- prove that summing AGB and BGB tendencies reproduces total-live-biomass
  tendency to float64 tolerance;
- run local forward/loss/gradient and one-day state/budget checks;
- on disjoint train/train groups, require finite candidate gradients, zero
  hard counts, and AGB/BGB median gradient norms each between `0.1x` and `10x`
  the NPP median norm after the frozen candidate coefficient rule;
- verify that no validation values, sealed-test values, optimizer updates, or
  checkpoints are used by coefficient preparation.

Failure at this gate stops the mechanism. Do not tune the interval or weights
using validation performance.

## Bounded Training Gate

If preparation passes, run a no-checkpoint 100-update Attention A/B on free
`gln01`, using point-years disjoint from the earlier 16-group diagnosis and an
unchanged fixed validation panel. Freeze exact thresholds before launch. The
recommended gate shape is:

- zero numerical, state, mask, discrete, and budget hard failures;
- candidate primary composite at least 0.5% better than the current-loss
  Attention control;
- NPP, AGB, BGB, and total-live-biomass aggregate errors no worse, with a
  declared majority of the six slice/horizon cells improved;
- maintenance-respiration aggregate error no worse than a predeclared guard
  relative to the objective control;
- report gradient clipping and component norms, but do not make clipping alone
  a pass/fail criterion;
- write no checkpoint or recovery.

The exact maintenance guard and required cell majority are discussion items;
they must be frozen before implementation of the training runner. A passing
Attention objective A/B permits the conditional graph-control run. It does
not open 365-day, held-point, Gate D2, sealed-test, or product acceptance.
