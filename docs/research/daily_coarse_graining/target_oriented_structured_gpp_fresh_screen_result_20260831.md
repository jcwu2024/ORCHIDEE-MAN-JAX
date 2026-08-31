# Structured-GPP Fresh Independent Screen Result

Date: 2026-08-31

Status: execution accepted; absolute 7/30-day science gate failed; broad
step-1000 checkpoint rejected for promotion

Executable commit:
`c422c37e19d3440812aed4f45391107c2b9021f7`

Screen contract SHA256:
`894a0b2dc329179566e624e7c04b3a63522588aed34feb8dc00e3187eb73e6a7`

Prepared manifest SHA256:
`43b321dd065cb7ee0d8f524af2f4ed5a2dfb56ad933a22f6d1bb33505029150e`

Validation panel SHA256:
`32c72823b7d190e8c8a3569dcbf1d57ead258ae635bdd717412fb7aa629718bb`

Admission report SHA256:
`a5b5f8ae058715e93b2d50c106b81b61798a25e5f53e7beea72deea466472e37`

Formal report SHA256:
`a9b9f6dcaf9aca349e738f096609e683342286d14d870d2c140afe2a3253798b`

## Execution

The free-CPU preparation produced all 96 frozen host artifacts. The panel has
16 windows in each temporal/spatial/joint and 7/30-day cell, excludes every
point-year consumed by the earlier panel within the corresponding slice, and
uses 96 distinct landpoints within its six cells. The admission then evaluated
one frozen window per cell on free `gln01` and passed every execution and hard
gate.

The formal free-`gln01` screen evaluated all 96 windows. Prediction, state,
discrete-state, unexpected-mask, water/carbon budget, and negative-stock hard
failures are all zero. The 960 declared-dynamic mask differences remain the
pre-existing report-only diagnostic. All 1,776 Teacher-state daily GPP values
are finite. The run performed no optimizer update or checkpoint write,
regenerated no Teacher data, used no historical v5 or sealed-test data, and
reconstructed neither 48 forcing records nor a 48-step state scan.

## Absolute Science Result

The formal science macro normalized RMSE is `0.3605389869`. No slice/horizon
cell passes all 13 required keys. Pass counts across the six cells are:

| Key | Passing cells |
| --- | ---: |
| AGB, BGB, total live biomass | `6/6` each |
| Ecosystem carbon, nonliving ecosystem carbon | `6/6` each |
| Maintenance respiration, heterotrophic respiration | `6/6` each |
| Autotrophic respiration | `1/6` |
| GPP, growth respiration, NPP, NEP, NEE | `0/6` each |

Free-rollout GPP normalized RMSE remains far outside the `0.35` seven-day and
`0.25` thirty-day limits:

| Slice | 7 days | 30 days |
| --- | ---: | ---: |
| Temporal | `0.92195` | `0.73409` |
| Spatial | `0.91279` | `0.90095` |
| Joint | `1.00839` | `0.96183` |

Growth-respiration normalized RMSE is `0.46018` to `0.55095`, and NPP is
`0.90017` to `1.32971`. The exact retained carbon identities therefore carry
the unresolved GPP error into growth respiration, NPP, NEP, and NEE. The
short-horizon stock pass is real, but it cannot compensate for failure of the
central carbon fluxes.

## Teacher-State Diagnostic

With every day evaluated from its Teacher day-start state, direct daily GPP
has aggregate normalized RMSE `0.4449608850` and normalized signed bias
`-0.1389520163`. Cell RMSE ranges from `0.16767` to `0.59120`; the joint slice
is worst at both horizons. Broad direct-GPP training therefore did not produce
adequate new-point/new-year local GPP generalization.

The direct daily and free cumulative statistics are not algebraically
subtractable because they aggregate different quantities. Their separation
does show that the failure already exists at the local process head and that
free state feedback may add further error. A matched per-day attribution is
required before deciding how much belongs to the direct GPP mapping, cross-day
state recurrence, or their interaction.

## Decision And Next Gate

Reject the broad step-1000 checkpoint for scientific promotion. Do not add
training steps, select on this fresh panel, run 365 days, open Gate D2, or
begin product acceptance. This result does not by itself reject Attention or
authorize an architecture change.

The next operation is a zero-training attribution on the same frozen 96
windows and checkpoint. It must separate Teacher-state direct GPP error from
the additional free-rollout state-feedback error, report temporal/spatial/
joint differences, and trace the signed GPP residual into growth respiration
and NPP. It may write diagnostics only and must retain every existing physical
and data-use boundary. Any architecture change suggested by that result must
be discussed with the user before implementation.
