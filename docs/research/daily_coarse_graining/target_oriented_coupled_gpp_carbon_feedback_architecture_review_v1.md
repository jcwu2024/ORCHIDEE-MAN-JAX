# Coupled GPP/Carbon-Feedback Architecture Review V1

Date: 2026-08-31

Status: user-approved architecture; local zero-training implementation is
accepted, while optimizer updates and training remain unauthorized

## Evidence And Design Question

The broad structured-GPP checkpoint still has two independent defects:

1. Teacher-state cumulative GPP fails `5/6` fresh cells, so the day-local GPP
   map does not generalize sufficiently.
2. Cross-day state feedback is material in all six fresh cells. On the frozen
   train-only component audit, carrying only
   `slowproc_stomate_previous_step_state` captures median `96.06%/81.95%` of
   the 7/30-day cumulative-GPP free-to-full-Teacher score gap.

The second result does not mean that 2,386 slow-process state coordinates
should become direct neural outputs. Those coordinates are produced by the
canonical constrained update and retained exact daily STOMATE processes.
Replacing that boundary would discard the strongest physical structure in the
current operator.

The design question is therefore narrower: can a coordinate-preserving view
of the day-start slow-process carbon state improve the GPP map, while 7/30-day
training constrains the exact recurrent carbon path created by GPP itself?

The current GPP specialist does not completely ignore carbon state. It sees
the accepted parent's frozen source tokens, but every continuous state field
has already been reduced to four statistics: mean, standard deviation,
minimum, and maximum. It also sees generic factorized-axis summaries. These
summaries discard which biomass part, age class, carbon pool, or PFT coordinate
carried a value. The proposed adapter is materially different because the GPP
query can select the original carbon coordinates conditionally instead of
receiving only a pooled description of them.

## Recommended Candidate

Working name:
`target_oriented_coupled_gpp_carbon_feedback_attention_v1`.

Retain unchanged:

- the accepted 45-token native-forcing Attention encoder;
- the potential, limitation, and bounded-correction GPP factorization;
- the explicit final `vcmax25` factor;
- the distribution-aware maintenance mechanism and all other parent outputs;
- the conservative updater and canonical retained exact tail;
- the absence of cross-day neural hidden state, 48-record forcing
  reconstruction, and a 48-step state scan.

Add one day-start carbon-feedback adapter:

1. Tokenize every defined continuous coordinate owned by
   `slowproc_stomate_previous_step_state` from the accepted Markov contract.
   The current PFT14 shape contains 2,386 coordinates.
2. Each token contains its normalized value, defined status, field identity,
   physical-axis roles, and non-PFT coordinate position. PFT slot identity is
   not embedded, preserving PFT permutation equivariance.
3. Each active-PFT GPP query performs two cross-attention blocks over global
   carbon coordinates and coordinates belonging to the same PFT. There is no
   quadratic self-attention over the 2,386-token bank.
4. Three small residual projections inject the resulting carbon context into
   the existing potential, limitation, and correction queries. Their final
   kernels start at scale `1e-3`, so the inherited update-1000 function is
   perturbed only slightly while gradients reach the new adapter.

The adapter uses the same width (`384`), eight heads, and source/axis mask
semantics as the accepted Attention family. Its new parameter budget is at
most 8 million and total candidate size must remain below the existing
120-million hard engineering cap. The exact parameter count must be frozen
after implementation and before a training contract.

## Output And State Use

The candidate still emits only one new specialist quantity:

```text
daily.fast_gpp[compact_pft]
```

It does not emit a next-day carbon state and does not add a learned cross-day
memory. The output follows the existing nonnegative factorization and remains
exactly zero for bare soil, inactive PFTs, and zero-fraction PFTs.

The recurrent path is physical and explicit:

```text
day-start slow-process carbon state
  -> carbon-coordinate cross-attention conditions daily.fast_gpp
  -> conservative daily update
  -> retained exact STOMATE processes
  -> next-day slow-process carbon state
  -> next day's GPP query
```

This is how the candidate addresses state feedback without relearning or
overwriting retained exact processes.

## Why Not Open More Outputs Initially

The Teacher-GPP bridge shows that correcting GPP is sufficient to repair the
observed downstream growth-respiration and NPP failures on the frozen fresh
panel. Opening maintenance, hydrology, temperature, litter, and DOC outputs at
the same time would allow compensation through unrelated process errors and
would make a failure difficult to interpret.

Water carry remains a monitored secondary path because it improves all three
7-day train-only groups and misses the frozen materiality threshold only
narrowly (`9.13%` versus `10%`). It is protected and reported, but it does not
receive a new trainable adapter in v1. A later water extension requires a
separate material interaction result or failure attribution from this
candidate.

## Trainable Parameters

The first stage trains:

- the existing `target_oriented.structured_gpp` subtree loaded from the broad
  update-1000 checkpoint;
- the new carbon-feedback adapter.

All 93,504,164 retained parent coordinates remain frozen. Non-GPP process
outputs must remain bit-identical to the parent for the same day-start input.

## Train-Only Objective

The exact coefficients and schedule must be frozen after zero-update gradient
calibration, but the objective inventory is fixed before implementation:

1. broad direct daily GPP normalized MSE plus signed-bias loss on previously
   unseen v6 `train/train` days;
2. a one-day next-state anchor restricted to
   `slowproc_stomate_previous_step_state` coordinates that have a finite,
   nonzero derivative with respect to `daily.fast_gpp` through the complete
   operator, grouped by source field rather than raw coordinate count;
3. 7/30-day cumulative GPP, maintenance respiration, growth respiration,
   total autotrophic respiration, and NPP;
4. 7/30-day terminal AGB, BGB, total live biomass, nonliving ecosystem carbon,
   and ecosystem carbon;
5. the existing exact carbon identities and zero-tolerance physical hard
   gates.

The one-day reachability inventory is a zero-training source/dependency audit,
not a data-driven architecture search. It prevents thousands of carbon state
coordinates that GPP cannot affect from diluting the state anchor.

The direct-GPP term remains mandatory. It prevents a long-horizon optimizer
from compensating for other frozen errors by making daily GPP physically
wrong.

## Sampling And Monitor

Direct days should continue from the unconsumed part of the accepted broad
schedule rather than repeat updates `1-1000`. Rollout windows must be a new
target-blind `train/train` schedule with no overlap with its monitor.

The existing 128-point-year direct monitor samples only 16 days per
point-year and misses calendar bins in 125 of 128 cases. Before training, the
replacement monitor should select 48 target-blind days per point-year, four
from each of 12 noleap calendar bins. It remains train-only and is excluded
from optimizer batches.

A separate fixed train-only 7/30-day monitor reports all science and water
protection keys. Neither monitor replaces the later fresh 96-window
evaluation.

## Matched Mechanism Control

The matched diagnostic control has the same carbon-coordinate tokens, width,
depth, input masks, residual injection points, parameter budget, output
factorization, objective, data, optimizer, and seed. It replaces
content-dependent carbon cross-attention with token-local MLP transforms and
fixed field-balanced pooling.

This control answers only whether content-dependent carbon-coordinate
selection matters. It is not a promotion baseline. The candidate must first
pass absolute science gates; if it fails, the control is not trained. A
relative win over the control cannot rescue a candidate that fails the
absolute gates.

## Bounded Admission And Stop Rule

Before any optimizer update:

- exact parameter ownership and count;
- finite eager/JIT forward, complete direct/one-day/7-day/30-day losses, and
  gradients;
- nonzero gradients in both cross-attention blocks, every residual injection,
  and the inherited GPP branches;
- finite nonzero GPP sensitivity to `vcmax25` and zero sensitivity to
  forbidden maintenance/allocation/mortality parameter channels;
- exact inactive-PFT isolation and strict PFT permutation equivariance;
- non-GPP parent outputs bit-identical for the same input;
- one complete day with finite state and exact water/carbon budgets;
- a free-`gln01` zero-update real-v6 admission measuring forward, loss,
  gradients, state/budget integrity, time, and peak memory.

A later training-preflight contract must separately authorize one discarded
optimizer proposal for gradient balance and clipping calibration. The current
architecture contract authorizes no optimizer construction or update.

The paid training ceiling must be set from that measured hot step. Milestones
are required at updates `250`, `500`, and at the frozen maximum, with early
stop when any physical hard failure occurs or two consecutive milestones fail
to improve both direct GPP and the 7/30-day train-only science macro. No
open-ended epoch extension is allowed.

Only a last-passing checkpoint may read the already frozen fresh 96-window
panel once. Promotion still requires every existing absolute 7/30-day key and
all hard gates; a small relative improvement is not enough. Failure closes the
candidate and returns to mechanism attribution, not a learning-rate or epoch
search.

## User Decision And Local Closure

The user approved all three decisions:

1. retain the current structured-GPP Attention and add coordinate-level
   carbon-state cross-attention rather than replacing the whole backbone;
2. keep `daily.fast_gpp` as the only newly controlled output and train carbon
   feedback through the exact retained tail rather than directly predicting
   STOMATE state;
3. use the pooled-MLP arm only as a post-success mechanism control, with
   absolute science gates controlling promotion.

The machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_carbon_feedback_v1.json`
at SHA256
`c7f81436ff6ba1179b01adad0f76b47ddd2aab68208e97719c2c5464f3e7e74d`.
It authorizes local implementation and a free-`gln01` zero-update admission
only. The accepted local implementation evidence is recorded in
`target_oriented_coupled_gpp_carbon_feedback_local_acceptance_20260831.md`.
