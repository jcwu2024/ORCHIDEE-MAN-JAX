# Coupled GPP Structured Precision Attribution Result (2026-09-05)

Status: accepted zero-training localization; the sampled Day-5 GPP jump is an
extremely conditioned `potential_blocks` response, not a float32 endpoint
quantization effect

Report SHA256:
`40c617ba096f5b9b50482316473d416b17dafb90b7162cbf0d59c9fffaf5b693`

Executable commit: `587c3fdb86b154d96130520d1365ab97ff63aff0`

## Execution And Integrity

The free-`gln01` GPU execution completed in `935.288` seconds with status
`passed` and classification
`structured_gpp_conditioning_persists_with_float64_weights`. It used artifact
update 550, runtime year 1976, seven-day window index 3, and Day-5 index 4.
The accepted crossed-attribution parent SHA256 is exact.

Every engineering gate passed: the historical direction and treatment
displacement reproduce, the control trace and parent `CC`/`TC` cells
reproduce, both precision paths have all 17 fixed fractions and exact own
endpoints, all factors are finite, routed GPP is nonnegative, and every
float32 ULP and interpolation-membership count conserves.

The first free launch at commit `599bb69` stopped during initialization before
historical direction reconstruction because a new guard incorrectly required
mapping insertion order rather than exact key-set identity. It produced no
JSON, parameter state, checkpoint, recovery, optimizer update, or training.
Commit `587c3fd` replaces every such parameter-tree check with exact key-set
comparison and adds a reversed-insertion-order regression test.

## Float32 Endpoint Inventory

The 14 top-level subtrees contain 129 array leaves and exactly 12,879,363
parameters. Leaf-level and subtree-level independent sums agree exactly.

| Quantity | Value |
| --- | ---: |
| Changed coordinates | `8,974,854` (`69.6840%`) |
| Unchanged coordinates | `3,904,509` |
| Exactly 1 ULP | `2,499,411` (`27.8490%` of changed) |
| Exactly 2 ULP | `1,234,221` |
| More than 2 ULP | `5,241,222` |
| Structured-tree delta L2 | `1.1840780520e-4` |
| Maximum absolute coordinate change | `7.7774748206e-6` |
| Sign crossings | `0` |

At interpolation fraction `1/2`, `72.1510%` of changed coordinates are
distinct intermediate float32 values, `20.6971%` remain at control, and
`7.1520%` have reached treatment. Thus the path is not merely a wholesale
endpoint snap caused by all changed weights being one ULP apart.

## Precision Paths

The float64 diagnostic casts only the `structured_gpp` endpoint weights and
their interpolation to float64. It retains the unchanged graph, including
the existing explicit activation casts.

| Fraction | Production float32 GPP | Float64-weight GPP |
| ---: | ---: | ---: |
| `0` | `2.7673363042` | `2.7673360823` |
| `1/2` | `2.7672266213` | `2.7672265689` |
| `10/16` | `2.7671984108` | `2.7671990274` |
| `11/16` | `4.8699010524` | `4.8699012928` |
| `3/4` | `4.8699009872` | `4.8699010618` |
| `1` | `4.8698997235` | `4.8698989473` |

Both paths make their largest adjacent change over exactly `10/16` to
`11/16`. The maximum adjacent fraction of the endpoint jump is
`1.00006622` for production float32 and `1.00006630` for float64 weights.
The float64/float32 endpoint-jump magnitude ratio is `0.9999997364`, with the
same sign. Float64 interpolation therefore reproduces, rather than smooths,
the discontinuity-like response. The frozen quantization-dominant hypothesis
is rejected for this sample.

## Top-Level Subtree Attribution

Only `potential_blocks` crosses the frozen `0.10` materiality threshold.

| Treatment subtree inserted alone | Sufficiency | Necessity capture |
| --- | ---: | ---: |
| `potential_blocks` | `0.9999997732` | `0.9999056398` |
| `potential_output` | `-0.0001096884` | `-0.0000267497` |
| `limitation_output` | `0.0000107582` | `0.0000187426` |
| `correction_output` | `0.0000053290` | `0.0000093778` |
| `correction_blocks` | `-0.0000004350` | `-0.0000007655` |
| `limitation_blocks` | `-0.0000002152` | `-0.0000005680` |
| All other eight subtrees | `0` | `0` |

Inserting treatment `potential_blocks` alone changes GPP from
`2.7673363042` to `4.8698992466`, reproducing virtually the complete
`+2.1025634193` effect. Replacing only those blocks in the treatment tree
with control reduces GPP to `2.7671379059`, removing `99.9906%` of the effect.
The result localizes the sampled pathology inside the existing potential
branch's residual attention/FFN blocks, not its final output layer, shared
forcing encoder, local input, PFT projection, or parent-hidden projection.

This is top-level causal localization. It does not identify one internal
layer, attention projection, FFN, LayerNorm parameter, or activation, and it
does not establish that the same conditioning occurs on other samples.

## Decision And Next Gate

The precision and top-level-subtree diagnostic is accepted and closed.
Increasing only stored/interpolated weight precision is not a plausible fix
for this sampled jump. The next gate is user discussion of a bounded response
to extreme conditioning in `potential_blocks`, or a different GPP potential
parameterization. No architecture edit or training continuation is authorized
by this result alone.

No optimizer, checkpoint, recovery, retained parameter update, Teacher
regeneration, historical v5 read, fresh-validation read, sealed-test read,
half-hour forcing reconstruction, 48-step scan, architecture change, paid
execution, 365-day rollout, Gate D2, or product acceptance occurred.
