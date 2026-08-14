# Direct-Direction Trust-Region Audit, 2026-08-15

## Decision

Admit global parameter step norm `0.001` for a bounded 10-update
direct-direction smoke. This is the first realized update mechanism in the
current objective study that decreases all four component losses in all four
train/train diagnostic groups.

The result supports the raw minimum-norm common-descent mechanism locally. It
does not yet show that ten consecutive updates generalize to independent
7/30-day rollout windows, and it does not promote the Attention checkpoint.

## Protocol

The frozen contract SHA256 is
`03c7d98ae25eb6f16d7d45714e6614732a65125335988bb1a0b0caa354b25343`.
Starting from the accepted update-1000 Attention parameters, the audit
recomputes the minimum-norm convex combination of unit component gradients in
the same four accepted groups. It applies virtual parameter deltas exactly
parallel to that raw direction at global norms
`0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1`.

No AdamW moments, coordinate-wise adaptive scaling, weight decay, or gradient
clipping are used. The seven proposed parameter sets are discarded. No
checkpoint, recovery, validation value, sealed-test value, new Teacher data,
or historical v5 data is written or read.

## Result

The free-`gln01` run at commit `55c30a3` completed with valid evidence:

- runtime report:
  `outputs/training/successor-direct-trust-region-55c30a3/report.json`;
- report SHA256:
  `ccc4e1fd29ddebf2d55976aec0282c62a14c814c6b148e1d862f9513379e1fbd`;
- elapsed Python time: `826.02 s`;
- peak JAX device bytes: `12,934,680,576`;
- hard failures and nonfinite values: zero;
- optimizer updates and retained proposals: zero.

Step norm `0.001` passes all four groups. Its median relative changes are:

| Component | Median relative change |
|---|---:|
| One-day anchor | `-0.0116%` |
| NPP | `-0.1652%` |
| Aboveground biomass tendency | `-0.3081%` |
| Belowground biomass tendency | `-0.0317%` |

The smaller norms improve every group but fail the preregistered minimum
median decrease for at least one component. Step norm `0.003` is already
outside the shared trust region: three groups improve, while one group's
largest component regression is `19.43%`, exceeding the global `1%` bound.
Larger norms are rejected more strongly. Therefore `0.001`, rather than the
largest superficially improving step in any one group, is the selected shared
norm.

## Next Gate

Freeze and run a free-`gln01`, 10-update direct-direction smoke from the same
update-1000 Attention parameters. Recompute the raw minimum-norm direction at
each scheduled mixed batch and apply a parameter delta of global norm `0.001`
directly. Do not use AdamW, momentum, adaptive coordinate scaling, weight
decay, or clipping.

The smoke must retain no checkpoint and must compare the initial and final
parameters on the accepted non-test 7/30-day validation subset. It may only
open a longer training design if consecutive updates remain finite and
bounded and the research-primary NPP/biomass rollout metrics improve without
an unacceptable maintenance-respiration regression. The 365-day rollout,
long-held-point validation, Gate D2, sealed test, and product gates remain
closed.
