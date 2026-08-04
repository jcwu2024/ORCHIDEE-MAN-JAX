# Daily Flux Label Inventory v1

Status: frozen Gate C1 inventory.

Date: 2026-08-04.

Machine authority:
`manifests/coarse_graining/daily_flux_label_inventory_v1.json`.

Audit implementation:
`research/daily_coarse_graining/daily_flux_label_inventory.py`.

## Result

The inventory contains 50 required water, carbon, and energy quantities. It
was audited against the real v5 Teacher contract hash
`6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442`.

| Domain | Present | Exactly derivable | Missing/non-identifiable |
| --- | ---: | ---: | ---: |
| Water | 6 | 1 | 13 |
| Carbon | 3 | 6 | 11 |
| Energy | 1 | 4 | 5 |
| Total | 10 | 11 | 29 |

Existing v5 shards contain the relevant inventory endpoints, native forcing,
daily GPP, maintenance respiration, and daily thermal context. They do not
contain daily-resolved evaporation, transpiration, runoff, drainage, internal
water transfers, OK_LEAK pool/layer transfers, or integrated energy fluxes.

An endpoint difference is a net tendency, not a unique process
decomposition. For example, one soil-moisture change can result from many
combinations of transpiration, bare-soil evaporation, vertical flow, runoff,
and drainage. The same ambiguity applies to simultaneous DOC decomposition,
export, adsorption, and vertical redistribution. These terms are therefore
classified as `missing_non_identifiable`, not fabricated as endpoint-derived
labels.

## Minimal Supplemental Capture

The 29 missing labels reduce to three diagnostic families:

1. `water_transfer_daily_v1`: daily source-resolved water amounts and
   reservoir/layer transfers;
2. `ok_leak_transfer_daily_v1`: daily resolved litter, POC, DOC, deposition,
   export, and redistribution transfers;
3. `energy_flux_daily_v1`: time-integrated net radiation, sensible, latent,
   ground, and phase-change energy.

Each family reduces values inside the compiled complete-day owner and stores
only daily tensors. It must not store a 48-step trajectory or become an
inference dependency.

The existing 669-point shards must not be regenerated before these captures
pass the non-neural replay gate on a small lifecycle matrix. Once replay
passes, supplemental labels can be produced separately and bound to the
existing shard identities.

## Gate C2 Result

Gate C2 now passes. True captured daily labels feed one non-neural constrained
updater followed by retained exact daily processes. Cold continuation,
ordinary later-day, restart-year, mask, budget, and restart-composition checks
are accepted in
[`gate_c2_constrained_replay_v1.md`](gate_c2_constrained_replay_v1.md).
