# Rollout Turnover Ratio Gradient Stabilization

Date: 2026-07-30

## Decision

Formal Experiment B job `14419242` is rejected. The one-step control completed
all 8,192 updates, but the mixed-horizon candidate failed closed at update 302
with finite forward values and 1,885,278 nonfinite parameter gradients. No
failed update was applied.

This failure is closed at the process level by commit `90113c0`. The fix does
not alter the ORCHIDEE forward quotient. It stabilizes only the custom JVP of
the leaf-age turnover fraction when the quotient derivative cannot be
represented in float64. The candidate then passed the complete update-302
audit and 768 consecutive optimizer updates from the formal update-256
checkpoint through update 1024.

This is not promotion of the daily surrogate. A fresh matched 8,192-update
screen remains required, and the sealed test split remains untouched.

## Formal Failure

Job `14419242` ran for `01:40:11` on `ibc13b02n01` and exited with code 1.
The immutable rejected root is:

```text
runtime/outputs/training/canonical-669-rollout-stability-v2-9e8aab2
```

The control arm completed `8192/8192`. The candidate stopped at update 302:

```text
landpoint: 221.0-107.0
year: 1973
horizon: 30
loss: 0.01140532764228806
gradient norm: NaN
nonfinite parameter gradients: 1,885,278
```

All forward losses and states were finite. Unexpected status, discrete,
nonfinite-state, and negative-carbon hard counts were all zero.

## Localization

The exact candidate state was reconstructed from the formal update-256
checkpoint and deterministically replayed to `next_update=302`. The complete
batch audit found two bad rollout samples and no bad one-step anchor samples.

For sample 3, horizon-prefix bisection gave:

| Prefix | Last day | Nonfinite gradients |
| ---: | ---: | ---: |
| 2 | 328 | 0 |
| 4 | 330 | 0 |
| 8 | 334 | 0 |
| 16 | 342 | 0 |
| 23 | 349 | 0 |
| 27 | 353 | 0 |
| 28 | 354 | 0 |
| 29 | 355 | 1,885,278 |

Day 355 is therefore the first bad transition. Commit `af5eb80` added a
checkpoint-bound arbitrary-day detach diagnostic. At the Day-355 entry state,
the local retained-tail derivative was already nonfinite:

- nine nonfinite state derivatives, limited to STOMATE `biomass`, `leaf_age`,
  and `leaf_frac`;
- 12 nonfinite fast-target derivatives, all in
  `daily_interface.resp_maint_part`;
- the first checkified process frame was
  `turnover_leaf_age_fall -> _stable_ratio_for_ad_jvp`.

The leaf carbon stock was `3.705819757041253e-311`. The forward Fortran
condition is strictly `biomass > 0`, so the quotient remains part of the
source-equivalent forward transition. Its reverse derivative contains a
reciprocal of this denominator and cannot be represented in float64.

## Fix

The forward function remains exactly:

```python
return numerator / denominator
```

Only its custom JVP is guarded. For denominators below
`sqrt(float64.tiny)`, the quotient tangent is set to zero. This threshold is
derived from the float64 representation limit: below it, the squared
reciprocal needed by gradient-norm accumulation cannot be represented. Values
this small are scientifically zero carbon stocks, while all normal model
states retain the existing analytic quotient derivative.

The change is an AD numerical-boundary fix, not a Teacher semantic change and
not a projection of model state.

## Evidence

After commit `90113c0`, the Day-355 diagnostic passed:

```text
prefix-29 L_rollout nonfinite gradients: 0
prefix-29 L_rollout gradient norm: 0.9866712093
detached Day-355 parameter nonfinite gradients: 0
initial-state / retained-tail / model-state nonfinite gradients: 0 / 0 / 0
fast-target nonfinite gradients: 0
checkify error: none
```

The complete update-302 audit also passed:

```text
bad anchor samples: 0 / 256
bad rollout samples: 0 / 8
weighted gradient norm: 0.4008063674
weighted nonfinite gradients: 0
all hard counts: 0
update applied: true
```

Commit `b8c8dee` added an audited diagnostic-checkpoint fork. It verifies the
formal source identity and SHA before rebinding the unchanged parameters,
Adam state, schedule position, and history to the diagnostic Git identity.
The sequential gate then passed:

| Gate | Resume point | Applied updates | Horizon counts `1/3/7/30` |
| --- | ---: | ---: | --- |
| 512 | 256 | 256 | `220/152/92/48` |
| 1024 | 512 | 512 | `431/317/182/94` |

All 768 post-fix updates were applied with all exact hard counts zero. The
final diagnostic checkpoint SHA256 is:

```text
9a6e10e1bf7ba30b9b88720bee7ba024e698bb6adc6fced7867b67e7a1d4736d
```

Evidence SHA256 values:

```text
pre-fix Day 355: b651e248b2f9862eafa8824a06cd7db890da425d952e585e167814a84b62df41
post-fix Day 355: 95c144d993bc410c9325901d0e29232e4d86ed9ab2b1d37d98169ffa93167ad2
complete update 302: cd42db44368197174a7e88eba0b9363b8b2f8d16949f70f950070f03cdc9945a
checkpoint fork: e5a86007aa08d48eda4b9fcc3f58c5eea886ec9056ea2329eeaa504aa453ebf8
prefix 512: 00c4c6d69858d1086b36696eddcd506a79169a25973aeb7adf79fa98da7353b8
prefix 1024: 933b857929b8dcb952863fae29352372c7737c3a60b532792b7d2a7d01e7d560
```

## Next Gate

Create one clean evidence commit containing the process fix, diagnostics, and
this report. Under a new immutable output root, regenerate preflight,
train-only coefficient calibration, and execution identity, then request
approval for the formal matched `gnall` rerun. Do not resume job `14419242`,
reuse its rejected root as promotion evidence, or inspect the sealed test
split.
