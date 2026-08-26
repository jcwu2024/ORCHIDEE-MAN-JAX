# Target-Oriented Flux Gradient Variance Audit Result (2026-08-26)

## Status

The frozen zero-update audit completed successfully on free `gln01` at commit
`c5a72a4`. It used the exact update-1000 checkpoint and 32 predeclared
train/train 30-day windows. The machine decision is
`mixed_batch_and_objective_geometry`.

The result rejects an unchanged continuation whose only intervention is to
increase the four-window long-horizon batch. GPP clearly benefits from a
larger effective batch, but maintenance and total autotrophic respiration do
not converge to a common direction even at 16 windows. NPP is already
directionally coherent at four windows. The next change must therefore repair
the flux objective basis as well as its effective sampling; no paid
continuation is admitted by this result.

## Executable And Artifact Identity

- branch: `research/daily-coarse-graining`;
- executable commit: `c5a72a4de8e90552be1f4efd47bedf7b108de0f1`;
- node alias: `gln01`;
- physical host: `ibc13b02n04`;
- report path:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/target-oriented-flux-gradient-variance-c5a72a4/report.json`;
- report SHA256:
  `1de8962a64aad6a7b76cf311f1dd14932780c67691ec472fbc342c5c8eb42fbe`;
- update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared-manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The report declares `teacher_data_regenerated=false`,
`historical_v5_used=false`, `sealed_test_read=false`,
`optimizer_updates_retained=0`, and `checkpoint_written=false`.

## Gradient Variance Result

The table reports the median pairwise cosine between disjoint estimates.
Cosine `1` means the estimates point in the same parameter-space direction,
`0` means they are unrelated, and a negative value means that following one
estimate tends to oppose the other.

| Flux key | 4 windows | 8 windows | 16 windows | 32-window gradient norm |
| --- | ---: | ---: | ---: | ---: |
| GPP | -0.76684 | -0.14252 | 0.55547 | 1.51497 |
| Maintenance respiration | 0.11110 | -0.92607 | -0.76497 | 3.95811 |
| Growth respiration | 0.43138 | 0.30952 | 0.90546 | 2.10120 |
| Autotrophic respiration | 0.08756 | -0.95519 | -0.70569 | 3.12112 |
| NPP | 0.66471 | 0.60417 | 0.98395 | 14.58004 |

Only GPP satisfies the frozen larger-effective-batch rule. Only NPP satisfies
the frozen rule that four windows are already adequate. Growth respiration
becomes stable by 16 windows but does not satisfy the preregistered minimum
cosine-gain rule. Maintenance and total autotrophic respiration remain
opposed between the two disjoint 16-window halves, so more windows alone are
not an established remedy.

The all-parameter common-32-window cross-key cosines include:

| Pair | Cosine |
| --- | ---: |
| GPP vs autotrophic respiration | -0.31549 |
| GPP vs NPP | 0.02519 |
| Maintenance vs growth respiration | 0.51891 |
| Maintenance vs autotrophic respiration | 0.90577 |
| Growth respiration vs NPP | 0.97758 |
| Autotrophic respiration vs NPP | 0.72512 |

These numbers do not show a maintenance-versus-growth sign conflict. They do
show that the current aggregate basis is insufficient for the acceptance
problem: the training objective optimizes total autotrophic respiration,
whereas maintenance and growth respiration are separate absolute acceptance
keys. A low aggregate error can therefore coexist with compensating component
errors.

## Integrity And Resource Result

The audit covers exactly `83,755,173` parameters across the complete declared
partition. The auxiliary path has an exactly zero direct flux gradient, as
expected because these five flux objectives route through the primary,
state, encoder, and shared-science paths rather than auxiliary-supervision
heads.

All physical hard-failure counts are zero. The 385 declared-dynamic state-mask
mismatches are report-only `rveget` cases under the accepted policy;
unexpected mask mismatches are zero. Total runtime was `1177.998` seconds,
including `987.273` seconds for gradients and Gram matrices. Peak live device
memory was `17,935,278,080` bytes.

An earlier attempt at commit `be9c391` was stopped before producing a report
after float32 Gram products disagreed with direct norms. The accepted
`c5a72a4` implementation accumulates Gram statistics in float64 with highest
matrix-multiplication precision. The failed attempt retained no update or
checkpoint and is not scientific evidence.

## Decision And Next Gate

The minimal algebraically independent flux basis for the next objective is:

```text
GPP
maintenance respiration
growth respiration
heterotrophic respiration
```

Total autotrophic respiration, NPP, NEP, and NEE remain exact derived
identities and required absolute acceptance metrics. This replaces one
aggregate term with its two required components; it does not add a second
copy of the same information.

Before any retained update, freeze and implement this component-resolved
amendment, then run a zero-update train-only geometry audit at the exact
update-1000 state. The amended flux direction must be compared with the
one-day/process, 7-day stock, and 30-day stock gradients. A bounded virtual
step must have finite parameters and states, preserve every physical hard
gate, and avoid material regression of the already accepted biomass and
slow-carbon behavior. Only a passing result may open a bounded continuation
proposal. Architecture replacement, 365-day validation, Gate D2, sealed-test
access, and product acceptance remain closed.
