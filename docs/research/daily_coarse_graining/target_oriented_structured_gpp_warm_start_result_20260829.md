# Structured GPP Warm-Start Result

Date: 2026-08-29

Status: capability gate passed; checkpoint remains non-promotion-eligible;
opens train-only 7/30-day gradient calibration

Slurm job: `14873550`

Execution commit:
`1fb8916d0d5e3c88223ddee44ab234ef33ae8c73`

Warm-start contract SHA256:
`1a3b8138b7fcef0cc28f7669d89cdfaae053244b706eebf23b3aa4efcc4818a6`

Report SHA256:
`c0e23be4a91a3f47e9411ef7baf513ade03a3e08b81578c97871f038bb07500f`

Specialist checkpoint SHA256:
`ebef94aa5a98c06646e45863459bfb755dc94fc00a31646cd591f7e6c508d3f9`

## Execution

The paid `gnall` job completed with exit code zero in `00:21:25`. All 250
preregistered optimizer updates executed and persisted. Each update consumed
all 208 existing Teacher-v6 day instances in one train/train artifact, for
52,000 day instances across the stage. Only the `12,879,363`-parameter
structured GPP specialist was trainable. All `93,504,164` retained parent
parameters remained exactly unchanged, including object identity and tree
structure.

The fixed clip threshold was `17.256861560753983`. Raw gradient norms ranged
from `0.655108` to `52.388569`, with median `13.260821`; 89 of 250 updates
were clipped. The median hot update took `2.070 s`; retained update execution
totalled `600.76 s`, while the complete report path took `1220.33 s`. Peak
live JAX allocation was `2.77 GB`. Based on the Slurm allocation of eight CPUs
and one V100 for 1285 seconds, the additive rate estimate is CPU CNY `0.20`
plus GPU CNY `0.79`, or CNY `0.99` total.

Teacher data was not regenerated, historical v5 was not used, validation and
sealed-test values were not read, no 7/30-day gradient was evaluated, no
forcing record was reconstructed, and no subdaily state scan was executed.

## Endpoint Capability

The eight evaluation artifacts were disjoint from the 250 training artifacts.
All eight improved:

| Artifact | Runtime year | Initial loss | Final loss | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 251 | 1962 | `0.768082` | `0.467731` | `39.10%` |
| 358 | 1984 | `0.818251` | `0.536745` | `34.40%` |
| 465 | 1981 | `0.397027` | `0.239266` | `39.74%` |
| 572 | 1985 | `1.432058` | `0.975222` | `31.90%` |
| 679 | 1965 | `0.356098` | `0.217571` | `38.90%` |
| 786 | 1991 | `1.825541` | `1.207225` | `33.87%` |
| 893 | 2004 | `2.854726` | `2.121751` | `25.68%` |
| 1000 | 2003 | `0.955845` | `0.686610` | `28.17%` |

The exact aggregate direct-GPP loss fell from `1.11241519` to `0.74803793`,
a `32.7555%` reduction. This exceeds the frozen minimum of `10%`, and the
individual result is `8/8` improvements against a minimum of `6/8`.

All physical hard failures are zero: prediction and state nonfinite counts,
unexpected mask changes, discrete mismatch, budget failure, and negative
stock counts are all zero. The separate count of 19 declared dynamic state
mask changes is diagnostic and is explicitly excluded from the hard-failure
sum by the frozen gate.

## Decision

The specialist has demonstrated that it can learn the direct daily GPP
mapping across held-out train/train artifacts. This is materially stronger
than a same-batch training-loss decrease, but it is not yet evidence of
recursive carbon-cycle accuracy. The update-250 checkpoint remains
`promotion_eligible=false`.

The active gate is now a separate train-only 7/30-day gradient calibration at
this retained endpoint. It must measure finite horizon objectives, physical
gates, specialist gradient coverage and norms, and the coefficient geometry
needed for a bounded joint objective. It may not retain an optimizer update or
read validation or sealed-test values. Joint training, 365-day rollout, Gate
D2, and product acceptance remain closed until their prerequisites pass.
