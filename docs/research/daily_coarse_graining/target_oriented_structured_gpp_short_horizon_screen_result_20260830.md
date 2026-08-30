# Structured-GPP Independent Short-Horizon Screen Result

Date: 2026-08-30

Status: execution accepted; absolute 7/30-day science gate failed; step-498
checkpoint rejected for promotion

Executable commit:
`d41cd0c578d29943f91625dcd0ab640dbe0bc425`

Screen contract SHA256:
`2fe3febf235b3dd395947a2ca5ed4dba9926f43f895e22e202b3911d6542dcbb`

Admission report SHA256:
`f653ed4b1a44b7aff0e92add2c728df6c1f62ffefd4ba0020f241ebec9c11239`

Formal report SHA256:
`a85a8d91c259df9dcb5bc5c9dc3b39c082b028196c6a112dfe432ef8c7834a97`

## Execution

The free-`gln01` admission evaluated one frozen window in each of the six
slice/horizon cells and passed. The formal run then evaluated all 96 frozen
windows: temporal, spatial, and joint slices; 16 windows per slice at both 7
and 30 days. It completed in `322.49 s` after the admission warmed both static
horizon compilations.

Prediction, next-state, discrete-state, unexpected-mask, water/carbon budget,
and negative-stock hard failures are all zero. The 975 declared-dynamic mask
differences are reported diagnostics and are excluded by the pre-existing
hard-gate definition. The run executed zero optimizer updates and checkpoint
writes, regenerated no Teacher data, used no v5 data or sealed test, and
reconstructed neither 48 forcing records nor a 48-step state scan.

## Absolute Result

The formal science macro normalized RMSE is `0.4613408767`; no slice/horizon
cell passes every required key. Pass counts across the six cells are:

| Key | Passing cells |
| --- | ---: |
| AGB, BGB, total live biomass | `6/6` each |
| Ecosystem carbon, nonliving ecosystem carbon | `6/6` each |
| Heterotrophic respiration | `6/6` |
| Maintenance respiration | `5/6` |
| GPP, growth respiration, autotrophic respiration, NPP | `0/6` each |
| NEP, NEE | `0/6` each |

GPP is negatively biased in every cell. Its normalized RMSE ranges from
`0.7114` to `1.4214`, against limits `0.35` at 7 days and `0.25` at 30 days.
The retained exact carbon identities then propagate the GPP deficit into
growth respiration, NPP, NEP, and NEE.

## Parent-Checkpoint Context

The inherited step-1000 Attention checkpoint was previously evaluated on the
same frozen panel. Its science macro was `0.3408543729`; the current value is
`35.3484%` worse. Mean six-cell GPP normalized RMSE increases from
`0.7667242` to `1.1454848` (`+49.3998%`). Mean maintenance-respiration error
falls from `0.4733982` to `0.1925169` (`-59.3330%`).

This comparison is attribution, not a claim that the weak parent is an
acceptable baseline. It shows that the maintenance mechanism generalizes,
whereas the structured-GPP training gain does not. The train-only joint panels
improved by roughly 43-51%, but the independent panel became worse.

## Data-Coverage Interpretation

The direct-GPP warm start used 250 artifacts with 208 Teacher days each. The
joint stage used a disjoint set of 250 candidate artifacts; 248 were presented
per epoch for four epochs. Across both stages there were 258,336 direct-GPP
presentations, but at most 104,000 distinct Teacher days before any
cross-artifact duplication. This upper bound is only `1.2105%` of the
8,591,565 accepted train/train transitions.

That mismatch is a credible under-coverage explanation for a
12,879,363-parameter specialist, especially because training fit improved
while every held slice acquired a negative GPP bias. It is not yet proof that
data coverage is the only cause; architecture conditioning and loss
calibration remain possible contributors.

The subsequent target-blind schedule audit resolved the upper bound to exactly
103,399 unique days (`1.203494%` of accepted train/train). See
[`target_oriented_structured_gpp_data_coverage_result_20260830.md`](target_oriented_structured_gpp_data_coverage_result_20260830.md).

## Decision And Next Gate

Do not continue the same 500-artifact schedule, select another epoch from this
validation panel, run 365 days, or open Gate D2. The next operation is a
zero-update coverage and throughput audit for broad direct-GPP supervision
from the existing v6 train/train release. It must quantify actual unique
landpoint/year/day coverage, define a disjoint train/train monitoring panel,
and measure a practical streamed batch before any new paid training contract.

Keep the accepted coordinate-aligned Attention parent and structured-GPP
mechanism available during this audit. This result rejects the present
training/data exposure as a promotable model; it does not by itself prove that
a wholly new architecture is required.
