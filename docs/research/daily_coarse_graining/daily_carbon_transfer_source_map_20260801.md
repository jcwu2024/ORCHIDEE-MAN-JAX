# Daily Carbon Transfer Source Map

Date: 2026-08-01

## Decision

The nine aggregate carbon-transfer labels are source-backed and sufficient for
an exact combined-inventory conservation diagnostic. They are not sufficient
to advance the complete daily carbon state.

Any daily-tendency successor must retain resolved source terms for pool and
layer destinations. It must not infer transfers from endpoint stock
differences or use nine scalar totals as a complete transition target.

The machine-readable authority is
`research/daily_coarse_graining/carbon_budget_ownership.py::`
`AGGREGATE_TRANSFER_LABEL_OWNERSHIP`.

## Aggregate Labels

| Label | Source availability | Conservation role |
| --- | --- | --- |
| `litter_input` | direct litter increment arrays | external inventory input |
| `litter_respiration` | direct litter/flood respiration | atmospheric loss |
| `litter_to_doc` | direct litter DOC/flood input | internal transfer |
| `poc_respiration` | exact derivation from gross POC fluxes and CUE | atmospheric loss |
| `poc_to_doc` | exact derivation from gross POC fluxes, CUE, and layer mask | internal transfer |
| `doc_to_poc` | exact derivation from gross DOC fluxes, CUE, fractions, and lignin | internal transfer |
| `doc_respiration` | exact derivation from gross DOC fluxes and CUE | atmospheric loss |
| `doc_external_input` | original routing, precipitation, and canopy-deposition inputs | external inventory input |
| `doc_export` | direct runoff, drainage, and flood export amounts | external inventory loss |

`doc_external_input` must use the original deposition inputs. Canopy-to-ground
drip is an internal movement from `interception_storage` and must not be
counted as new carbon.

All rates must be converted with `dt_days` exactly once. PFT-local stocks and
fluxes must be weighted with the source `veget_max` convention exactly once.
The aggregate report is per landpoint; the retained training labels remain
pool- and layer-resolved.

## Missing State Resolution

The following internal processes are not represented by the nine totals:

- free/adsorbed DOC equilibration;
- vertical DOC water transport;
- vertical DOC diffusion;
- cryoturbation redistribution;
- `PERMA_PEAT` redistribution.

Their existing source outputs or exact deterministic owners must remain in the
transition. A conservative decoder that omits their destinations cannot
reconstruct `DOC`, `carbon_32l`, or the source-order `deepC_peat` writeback.

## Capture Consequence

The accepted 96-day selector can be reused, but the next diagnostic asset must
capture resolved source-produced flux arrays and the original external-input
arguments. It must be emitted by a diagnostic variant of the compiled
`OK_LEAK` scan and must not change the production fast path.

The exact 13-driver capture remains a hybrid baseline and an input-completeness
probe. It is not by itself the preferred final daily architecture. On the same
96 days, compare:

1. predicted 48-step drivers plus the exact scan;
2. predicted resolved daily tendencies plus a conservative deterministic
   update;
3. direct endpoint prediction only as a rejected/control baseline.

Do not submit paid neural training until one candidate passes tiny-set overfit,
finite-gradient, conservation, nonnegativity, and 7/30-day rollout gates.
