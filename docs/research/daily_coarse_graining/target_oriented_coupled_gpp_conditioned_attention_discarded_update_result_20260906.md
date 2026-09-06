# Coupled GPP Conditioned Attention Discarded Update Result (2026-09-06)

## Outcome

The corrected free-`gln01` preflight completed at commit `2adb422` in
`6,410.496` seconds. Its classification is
`conditioned_discarded_update_transfers_but_absolute_science_remains_open`.
The conditioned candidate admits a common 43-constraint direction, the
architecture-calibrated full step passes every train-only gate, and the same
discarded step improves every one of the eight fixed-monitor artifacts.

The server report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-conditioned-attention-discarded-update-2adb422.json`,
SHA256
`0aa30a0eb38f4e1f4da8d0a155a325cae35271b2cca60c04bf104c7e9fe9b42a`.
The local evidence copy is
`outputs/diagnostics/target-oriented-coupled-gpp-conditioned-attention-discarded-update-2adb422.json`.
No parameter or optimizer state from this audit is retained.

## Direction And Step

The minimum-norm simplex direction has cosine `0.3265653-0.4816572` with all
43 unit-gradient constraints. Its seven-constraint support contains five
individual direct-GPP artifacts plus the 7-day and 30-day science objectives.
This is substantially better conditioned than the parent architecture's
accepted artifact-aware direction, whose minimum cosine was `0.0422923`.

The new architecture's linearized-response cap selects a full parameter
delta L2 of `0.0005178796`. The first tested multiplier, `1`, passes, so no
smaller scale is evaluated. On the scheduled train-only group it changes:

| Source | Relative change |
| --- | ---: |
| Direct GPP | `-0.9168%` |
| Reachable one-day state | `-0.7218%` |
| 7-day component-resolved science | `-0.6061%` |
| 30-day component-resolved science | `-2.5039%` |

All `40/40` train-only direct-GPP artifacts improve. Their combined mean
changes by `-0.9204%`; even the least-improved artifact changes by
`-0.4148%`. The train-only 7/30-day required-science macro improves by
`1.3425%`, and every physical hard gate passes.

## Fixed-Monitor Transfer

The eight monitor artifacts are read only after direction and scale selection.
All `8/8` improve in direct GPP, with per-artifact changes from `-0.8332%` to
`-1.9838%` and a mean change of `-0.9373%`. Monitor source means change:

| Source | Relative change |
| --- | ---: |
| Direct GPP | `-0.9373%` |
| Reachable one-day state | `-1.0276%` |
| 7-day component-resolved science | `-0.7332%` |
| 30-day component-resolved science | `-0.6816%` |

The required-science macro improves by `0.7579%`. Every critical and
supporting key remains within its frozen regression cap, and all physical
hard gates pass. The result therefore transfers across the preregistered
train-only/monitor boundary rather than improving only the 40 direction
artifacts.

## Absolute Accuracy

This is a directional success, not an accepted scientific model. The frozen
absolute gate remains open on both train-only and monitor panels. GPP, NPP,
growth respiration, and autotrophic respiration miss their 7-day and 30-day
RMSE or bias limits. On the monitor, for example, normalized GPP RMSE changes
only from `1.00757` to `1.00443` at 7 days and from `0.97364` to `0.97066` at
30 days; the required limits are `0.35` and `0.25`.

Biomass and ecosystem-carbon keys already pass the short-horizon absolute
gate, but that does not establish long-run accuracy. More than one accepted
update is required before 365-day or product-level evaluation is meaningful.

## Integrity And Runtime

All `93,504,164` retained parameters and the parent update-2 recovery remain
exactly unchanged. The runner creates no optimizer, checkpoint, recovery, or
trainable output. It reads only accepted Teacher-v6 `train/train` artifacts,
does not regenerate Teacher data or use v5, and performs neither forcing
reconstruction nor a 48-step state scan.

The JAX allocator peak is `11,982,794,240` bytes under a
`25,559,728,128`-byte limit. Most wall time is one-time XLA compilation of
the 7-day and 30-day reverse paths; after compilation, repeated artifact
evaluations complete in seconds.

## Decision

Accept the conditioned Attention candidate for a bounded persistent-training
smoke, with the absolute promotion gate unchanged. The smoke must begin from
an identity-exact serialization of this same conditioned update-2 state,
recompute its common direction and calibrated step after every accepted
update, use disjoint train-only panels, and stop on the first monitor-transfer
failure. It remains a training diagnostic, not model promotion.

Fresh validation, paid execution, 365-day rollout, Gate D2, sealed test, and
product acceptance remain closed.
