# Coupled GPP/Carbon-Feedback Local Acceptance

Date: 2026-08-31

Status: accepted local zero-training implementation; real-v6 GPU admission is
next

Architecture contract SHA256:
`c7f81436ff6ba1179b01adad0f76b47ddd2aab68208e97719c2c5464f3e7e74d`

## Implemented Boundary

The candidate retains the update-1000 structured-GPP mechanism and adds a
coordinate-preserving carbon-state adapter. The adapter reads all 2,386
continuous coordinates in 107 fields owned by
`slowproc_stomate_previous_step_state`. It embeds normalized value, defined
status, field identity, and 17 non-PFT physical-axis roles and positions.
Absolute PFT position is not embedded.

Each compact-PFT GPP query uses two shared-weight cross-attention blocks over
global coordinates plus coordinates owned by the same PFT. Three residual
projections, initialized at scale `1e-3`, enter the inherited potential,
limitation, and correction queries. There is no quadratic self-attention over
the carbon bank.

The candidate still newly controls only `daily.fast_gpp`. The conservative
updater, distribution-aware maintenance output, and canonical retained exact
STOMATE tail are unchanged. There is no neural next-state output, cross-day
hidden state, half-hour forcing reconstruction, or subdaily state scan.

## Parameter Ownership

- inherited candidate: `106,383,527` parameters;
- new carbon-feedback adapter: `3,080,832` parameters;
- complete candidate: `109,464,359` parameters;
- first-stage trainable structured-GPP plus adapter: `15,960,195` parameters;
- frozen retained parent: `93,504,164` parameters;
- engineering ceiling: `120,000,000` parameters.

The pooled-MLP arm remains a post-success mechanism control. It is not a
promotion baseline and is not implemented or trained at this gate.

## Local Evidence

Five new unit gates pass:

- exact contract-derived coordinate layout and exact parameter counts;
- finite eager/JIT outputs, complete direct loss, and all adapter gradients;
- nonzero gradients in both cross-attention blocks and all three residual
  injections;
- nonzero selected-carbon-state effect, strict PFT permutation equivariance,
  inactive-PFT isolation, complete single-day state, and exact water/carbon
  budgets;
- nonzero `vcmax25` sensitivity and exact zero sensitivity to maintenance,
  allocation, and mortality parameter channels.

The five pre-existing structured-GPP tests also pass after the residual-hook
refactor. Ruff, Python compilation, canonical contract-hash verification, and
`git diff --check` pass.

These are synthetic/local plumbing gates. They do not prove behavior on the
accepted update-1000 checkpoint or real v6 data, and they are not scientific
promotion evidence.

## Next Gate

Run one free-`gln01` zero-update admission against the accepted broad
update-1000 checkpoint and one train/train v6 day. It must load the frozen
checkpoint, initialize only the new adapter, and measure real-v6 eager/JIT
forward, complete direct loss, adapter and inherited structured-GPP gradients,
PFT/parameter ownership, one complete daily transition, budgets, time, and
peak live device memory.

The admission must not create an optimizer, retain a parameter proposal, write
a checkpoint, read fresh targets, regenerate v6, use v5, reconstruct 48
forcing records, or run a 48-step state scan. Objective coefficients,
GPP-reachable state inventory, discarded optimizer proposal, and paid training
belong to later separately frozen gates.
