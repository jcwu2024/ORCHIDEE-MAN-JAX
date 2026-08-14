# Realized AdamW Step Audit, 2026-08-15

## Decision

Reject both inherited-state and reset-state AdamW as realizations of the raw
minimum-norm common-descent direction. Do not run an optimizer-reset training
smoke and do not continue the rejected 10-update mechanism.

This is an optimizer-handoff result, not an Attention-architecture rejection.
No parameter or optimizer update was retained.

## Protocol

The frozen contract SHA256 is
`9fbfd2d16479597231ed3541d720b08ee942596bfa6a5fb4c424703841a1d695`.
The audit reuses the four accepted train/train feasibility groups, the
update-1000 Attention parameters and AdamW state, and the scheduled
update-1001 learning rate. For each group it computes the raw minimum-norm
direction, then compares it with the negative proposed parameter delta from:

- AdamW with the inherited update-1000 first and second moments;
- AdamW with both moments reset to zero.

It also evaluates all four component losses after each virtual step. Proposed
parameters and optimizer state are discarded, no checkpoint or recovery is
written, and no validation or sealed-test value is read.

## Result

The free-`gln01` run at commit `6cec975` completed all four groups. Report:

- runtime path:
  `outputs/training/successor-realized-step-6cec975/report.json`;
- file SHA256:
  `4b987902e4baf40e5223ae9201be416ff8f61f64080309056302ba947bf47037`;
- elapsed Python time: `862.15 s`;
- peak JAX device bytes: `13,351,674,112`;
- hard failures and nonfinite values: zero.

The raw direction passes in `4/4` groups. AdamW does not preserve it:

| Realization | Minimum component cosines | Direction groups passed | Virtual-loss groups passed |
|---|---|---:|---:|
| Inherited AdamW | 0.00169, 0.00229, 0.00464, 0.00039 | 0/4 | 3/4 |
| Reset AdamW | 0.00632, 0.00794, -0.00003, -0.00210 | 0/4 | 2/4 |

The gate requires minimum realized cosine `0.01` and all four virtual losses
to increase by no more than `1e-6` in at least three groups. Inherited AdamW
nearly orthogonalizes the proposed parameter displacement relative to at
least one component in every group. Resetting moments does not repair this
and makes one component direction negative in two groups. Therefore old
momentum is not the sole problem; AdamW's coordinate-wise normalization and
the full optimizer transform do not realize the global minimum-norm direction.

## Next Gate

Before any further training, freeze a zero-update direct-direction trust-region
audit. It must apply virtual parameter deltas exactly parallel to the raw
minimum-norm direction, evaluate a preregistered log-spaced grid of global
step norms, and select a single shared norm only if it improves every
component in at least three of four groups with zero hard failures. It must
not use momentum, coordinate-wise adaptive scaling, or weight decay.

If no shared norm passes, close raw minimum-norm training and return to
batch-to-rollout objective/generalization design. If one passes, only a
bounded 10-update matched smoke may be designed. No architecture change is
implied or authorized.
