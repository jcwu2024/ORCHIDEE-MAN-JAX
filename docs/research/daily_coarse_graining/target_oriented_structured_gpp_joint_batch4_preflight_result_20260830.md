# Structured GPP Batch-Four Joint Preflight Result

Date: 2026-08-30

Status: completed and accepted; bounded batch-four joint training design may
proceed, but paid execution is not yet authorized

Execution commit:
`42c0e4e124afee3bf7b4b190982eb411552952a0`

Preflight contract SHA256:
`06d8c3f43beeac9c0c64dc8fd679ff57d09dc7b34df64a233df9f9d0dcc19169`

Report SHA256:
`6a574c67aa3aaa0883cbe066b3fe4372490163e7c3639cc0a172c3401cea0ac1`

## Result

Both independent batch-four proposals pass every frozen aggregate gate. Each
started from the accepted warm-start specialist and inherited AdamW state at
step 250 and independently proposed step 251.

| Artifact group | Direct GPP mean | 7-day mean | 30-day mean |
| --- | ---: | ---: | ---: |
| `512, 548, 584, 620` | `-0.1694%` | `-0.1079%` | `-0.0897%` |
| `656, 692, 728, 763` | `-0.2086%` | `-0.1038%` | `-0.1222%` |

The negative realized AdamW step retains a positive cosine with every
aggregate source gradient:

| Artifact group | Direct cosine | 7-day cosine | 30-day cosine |
| --- | ---: | ---: | ---: |
| `512, 548, 584, 620` | `0.02161` | `0.02133` | `0.01986` |
| `656, 692, 728, 763` | `0.02121` | `0.02078` | `0.01939` |

The aggregate gradient norms are `21.1003` and `34.6085`, both below the
frozen clip threshold `46.8029`; both clip scales are `1.0`. Every aggregate
source and combined gradient is finite, nonzero, covers all `12,879,363`
specialist parameters, and reaches all 129 leaves. Every specialist component
changes by a finite nonzero amount.

All one-day, 7-day, and 30-day physical hard failures are zero before and
after both proposals. Declared-dynamic mask counts remain diagnostic, as in
the accepted parent objective; unexpected masks, discrete mismatches,
nonfinite predictions/states, negative stocks, and budget failures are all
zero. Both proposals retain the `93,504,164`-parameter parent by object
identity and leave the warm-start checkpoint hash unchanged.

## Per-Artifact Evidence

Individual artifacts were preregistered as report-only because batch size 4
is the admitted update unit. Six of eight artifacts improve all three losses.
Artifact 512 improves direct and 7-day loss while its 30-day loss rises
`0.2926%`; artifact 548 improves 7/30-day loss while direct loss rises
`0.1026%`. Their four-artifact aggregate still improves every source. The
second group improves all three sources on all four artifacts.

This is the expected distinction between a stochastic minibatch update and a
per-example monotonic optimizer. It does not relax a failed group gate.

## Execution And Metadata

The free-`gln01` run completed in `1073.88 s` with exit code zero. Peak live
JAX allocation was `3,008,214,528` bytes, approximately `2.80 GiB`. It wrote
no checkpoint or recovery and retained neither proposal nor optimizer state.
Teacher data was not regenerated; historical v5, validation, and sealed-test
values were not read; no 48-record forcing reconstruction or subdaily scan
was executed.

The raw report's top-level `model.retained_parent_parameter_count` is labeled
incorrectly as `93,504,933`, the parent capacity before replacing its old
769-parameter GPP projection. The contract identity and both executed
per-group retained-tree reports correctly record `93,504,164`, 1172 leaves,
same tree structure, same array objects, and exact equality. A follow-up source
correction separates the pre-specialist parent count from the retained-parent
count. This reporting-only correction does not require repeating the complete
GPU calculation.

## Decision

Accumulated batch-four AdamW is admitted for bounded training design. The next
task is to freeze the exact artifact schedule, update count, checkpoint and
recovery policy, non-test evaluation panel, stopping gates, GPU resources,
wall-time, and worst-case paid cost. No paid job, 365-day rollout, Gate D2,
sealed test, or product acceptance is authorized by this preflight alone.
