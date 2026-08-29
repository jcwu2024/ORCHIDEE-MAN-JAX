# Structured GPP Joint-Gradient Calibration Result

Date: 2026-08-29

Status: accepted free-GPU zero-update calibration; joint coefficients and
combined-gradient clip frozen

Execution commit:
`7c547d55f54ed95eedef0cec79a328fb882f0663`

Calibration contract SHA256:
`c00ebf0242744c1583c12d1475b20021e325c4816c0391fb78a928ac874868fe`

Report SHA256:
`6ff2d03e0719b22c64012834ea3c5b134a740efa59be01d5715e5cf4c24389c0`

Warm-start checkpoint SHA256:
`ebef94aa5a98c06646e45863459bfb755dc94fc00a31646cd591f7e6c508d3f9`

## Result

The frozen train/train calibration passed on one free `gln01` V100. Each of
the eight artifacts supplied all 208 direct Teacher-GPP days, eight recursive
7-day windows, and four recursive 30-day windows. Every source gradient is
finite and nonzero over all `12,879,363` specialist parameters, and all 129
specialist leaves are nonzero for every source. All one-day, 7-day, and
30-day physical hard-failure counts are zero.

The gradient-norm ranges were:

| Source | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Direct GPP | `11.01864` | `23.18934` | `37.35352` |
| 7-day science | `0.406872` | `1.070004` | `2.208927` |
| 30-day science | `0.497732` | `1.383205` | `2.174055` |

The preregistered median ratio rule freezes these coefficients:

```text
direct GPP: 1.0
7-day science: 5.065197169020285
30-day science: 4.594433600967992
```

The resulting eight combined-gradient norms range from `16.2503` to
`48.2736`. Their p90 freezes the joint clip threshold at
`46.802861793665286`.

## Gradient Geometry

All pairwise source cosines are positive on all eight artifacts:

| Pair | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Direct versus 7-day | `0.6230` | `0.6500` | `0.6949` |
| Direct versus 30-day | `0.4849` | `0.5366` | `0.5857` |
| 7-day versus 30-day | `0.8440` | `0.8926` | `0.9500` |

The calibrated combined gradient remains strongly aligned with every source.
Its minimum cosine is `0.9376` with direct GPP, `0.8083` with 7-day science,
and `0.7005` with 30-day science. This is positive evidence that the direct
mapping and recursive science terms currently admit an ordinary shared raw
gradient. It does not yet prove that inherited AdamW moments and
preconditioning preserve that direction in the actual parameter step.

## Execution And Policy

Total wall time was `4526.34 s` (`75.44 min`). The first artifact paid the
cold compilation cost and took `4126.93 s`. The remaining seven hot artifacts
had a median total time of `28.34 s`; their typical direct, 7-day, and 30-day
gradient paths were approximately `2.0 s`, `8.1 s`, and `16.8 s`. Peak live
JAX device allocation was `3,447,174,144` bytes, approximately `3.21 GiB`.
The roughly 25 GiB visible in `nvidia-smi` was allocator reservation, not live
tensor use.

No optimizer was created, no parameter update was executed, and no checkpoint
was written or changed. The retained parent remained exactly unchanged.
Teacher data was not regenerated; v5, validation, and sealed-test values were
not read; no 48-record forcing sequence or subdaily state scan was created.

## Decision

The calibration is accepted. The next gate is one separately frozen,
discarded joint AdamW proposal on train/train artifact 512. It uses the
warm-start optimizer state at step 250, the coefficients above, and the
combined p90 clip. It must prove that the realized negative parameter step is
a descent direction for all three source gradients and that all three
same-artifact losses decrease, while retaining no model or optimizer state.
Paid joint training remains closed.

