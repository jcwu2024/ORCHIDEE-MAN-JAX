# Coupled GPP Localized Jump Trace V1

## Question

The exact-zero audit proves that artifact `550` alone causes the update-3
small-step 7-day regression. This trace asks where that discontinuity first
appears in the seven-day rollout: a discrete state, a defined-status mask, a
continuous state leaf, or a daily science flux/stock.

## Direction Reconstruction

The audit loads the exact parameter-only recovery retained after logical
update 2 and the rejected update-3 proposal. It reuses the proposal's frozen
minimum-norm simplex weights. Only the 26 sources with nonzero weight are
recomputed: 24 individual direct-GPP artifacts, the scheduled reachable-state
mean, and the scheduled 30-day mean. Zero-weight rows are represented as exact
zero rows in the original 43-source order before combining the direction.

The reconstructed combined norm must match the rejected proposal within
`rtol=1e-5, atol=1e-10`. The treatment is the historical smallest scale
`1/64`, giving parameter displacement norm
`0.00012123395861015631`. Parameters exist in memory only.

## Bounded Trace

The audit first evaluates the eight seven-day windows inside artifact `550`
under exact-zero control and the `1/64` treatment. It selects the window with
the largest positive objective displacement, then returns the complete seven
daily transition results for that one matched window.

For each rollout day it reports:

- discrete-state and defined-mask mismatch counts;
- maximum absolute and training-scale-normalized continuous-state change,
  with the owning Markov state leaf and coordinate;
- changes in all eight reported science flux trajectories: GPP, maintenance,
  growth, autotrophic and heterotrophic respiration, NPP, NEP, and NEE;
- changes in the seven science stock trajectories.

The first material day is the earliest day with a discrete or defined-mask
change, or a continuous-state change of at least `1e-3` training scale. The
trace is diagnostic and does not assume in advance that the responsible path
is discrete.

## Execution Boundary

This is one free-`gln01` run with a 90-minute timeout. It does not train,
persist parameters, change the Attention architecture, regenerate v6, read
v5, inspect fresh validation or sealed test, reconstruct half-hour forcing,
or execute a 48-step candidate scan. Paid execution, changed losses or gates,
365-day rollout, Gate D2, and product acceptance remain closed.
