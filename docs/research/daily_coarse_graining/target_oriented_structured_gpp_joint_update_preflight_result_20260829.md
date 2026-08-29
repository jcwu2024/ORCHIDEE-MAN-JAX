# Structured GPP Joint-Update Preflight Result

Date: 2026-08-29

Status: completed and rejected; paid joint training remains closed

Execution commit:
`e9aa41e10ef924b511c6ab0b9524263d61524a7a`

Preflight contract SHA256:
`5d255e49f7add44c4948a3e737fe6edd72999b6fbd6a37439dc787cb38c0c5d5`

Report SHA256:
`c63d2d4e1844f6df4b1f87cf63b2c59554b1567df6b0a8e19851ff9b7abccfc6`

## Result

The one-proposal free-`gln01` preflight completed on train/train artifact 512
(runtime year 1973). All direct, 7-day, 30-day, and combined gradients are
finite and cover all `12,879,363` specialist values and 129 leaves. Every
specialist component changes by a finite nonzero amount in the ephemeral
proposal. All pre/post physical hard-failure counts are zero, all
`93,504,164` retained parent values remain the same array objects, and the
warm-start checkpoint hash is unchanged.

The proposal is rejected because it does not preserve the 30-day objective:

| Source | Before | After | Relative change |
| --- | ---: | ---: | ---: |
| Direct GPP | `0.18350344` | `0.18328849` | `-0.1171%` |
| 7-day science | `0.02789089` | `0.02783728` | `-0.1922%` |
| 30-day science | `0.00512526` | `0.00513836` | `+0.2558%` |

The weighted science term decreases `0.1282%`, and the complete calibrated
joint objective decreases `0.1224%`. Those aggregate improvements do not
override the preregistered requirement that all three sources decrease.

## Cause

This is not primarily an optimizer-only failure. Artifact 512 has materially
different raw gradient geometry from the eight coefficient-calibration
artifacts:

| Pair | Raw gradient cosine |
| --- | ---: |
| Direct versus 7-day | `0.5966` |
| Direct versus 30-day | `-0.4502` |
| 7-day versus 30-day | `-0.9134` |

The frozen combined-gradient norm is `3.5030`, below the clip threshold, so
the clip scale is exactly `1.0`. After inherited AdamW momentum and
preconditioning, cosines between each source gradient and the negative
parameter step are `0.01932`, `0.02004`, and `-0.01667`. The 30-day source is
therefore a realized ascent direction, consistent with its observed loss
increase.

The calibrated coefficients remain correctly computed from their frozen
eight-artifact sample. What fails is the assumption that the positive source
geometry on those eight artifacts is representative of every intended
single-artifact joint update. This result does not reject the architecture or
the three-source objective. It rejects entry to paid training with one
artifact per update under the frozen common-descent gate.

## Execution

The accepted execution took `806.72 s`. Direct forward/gradient took
`52.92/70.72 s`, 7-day forward/gradient took `35.37/209.16 s`, and 30-day
forward/gradient took `42.21/210.67 s`. Peak live JAX allocation was
`3,062,955,008` bytes, approximately `2.85 GiB`.

One earlier attempt at commit `b073ea7` stopped before gradient evaluation
because the new runner duplicated the pre-replacement parent count instead of
reusing the accepted constant. It produced no report, optimizer proposal, or
checkpoint. Commit `e9aa41e` removed the duplicate constant and the complete
execution above passed every engineering identity and physical gate.

No proposal, optimizer state, checkpoint, or recovery was retained. Teacher
data was not regenerated; v5, validation, and sealed-test values were not
read; no 48-record forcing reconstruction or subdaily scan was executed.

## Decision

Paid joint training remains closed. The next gate is a zero-update gradient
heterogeneity and effective-batch audit over eight equally spaced artifacts
from the planned joint-training interval. It will evaluate grouped gradient
accumulation at effective batch sizes 1, 2, 4, and 8 and identify the smallest
size, if any, whose calibrated combined gradient is a descent direction for
all three aggregate sources in every group. Only then can an accumulated
AdamW proposal be designed.

