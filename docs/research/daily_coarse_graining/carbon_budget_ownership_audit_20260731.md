# Carbon-Budget Ownership Audit

Date: 2026-07-31

## Decision

The next admissible daily-surrogate architecture should predict the compact
drivers of the 48 ordered `OK_LEAK` transitions and then execute the existing
source-backed JAX `lax.scan`.

Do not build another model that independently predicts `carbon_32l`,
`DOC`, or `deepC_peat`. Do not begin paid training yet.

The current 669-point Contract-v5 data remains valid for the canonical state,
the non-carbon fast-day targets, endpoint supervision, and rollout
evaluation. It does not contain the internal transfer labels needed by an
aggregate conservation decoder, nor the 13 half-hour driver series needed by
the exact `OK_LEAK` scan. A bounded auxiliary capture is therefore required;
the 669-point Teacher dataset must not be regenerated.

## Source Ownership

Contract v5 stores 1,376 compact `OK_LEAK` values:

| Ownership class | Width | Fields |
| --- | ---: | --- |
| Independent carbon inventory | 1,222 | `litter_above`, `litter_below`, `carbon_32l`, `DOC`, `interception_storage` |
| Derived diagnostic | 64 | `deepC_peat` |
| Ratios, overlapping tracers, and partitions | 90 | lignin fractions, `litterpart`, `dead_leaves`, four fuel classes |

The independent mass-balance inventory follows the source checks:

```text
sum_PFT veget_max * (
    litter_above + litter_below + carbon_32l + free_DOC + adsorbed_DOC
)
+ interception_storage
```

`interception_storage` follows the source's unweighted canopy-deposition
accounting. `dead_leaves` and fuel classes overlap the aboveground litter
stock and must not be counted again. Lignin and `litterpart` are fractions.

`deepC_peat` is not an independent scientific stock:

1. `soilcarbon_leak` resets it from the three `carbon_32l` pools;
2. the sequential `PERMA_PEAT` branch may move carbon to deeper layers;
3. it is retained as output/restart state;
4. the next active call resets it again before decomposition.

The exact owner is
`stomate_soilcarbon.f90::soilcarbon_leak` lines 1375-1437 and the matching
JAX owner is `soilcarbon_perma_peat_redistribute`. Because the value is
formed before that half-hour's decomposition, deriving it by merely summing
the final daily `carbon_32l` would also be wrong. It must be emitted by the
ordered source process.

## Carbon Transfers

The combined inventory receives carbon through vegetation turnover and
`bm_to_litter`, TF-DOC canopy/ground deposition, and routed topsoil/subsoil
DOC. Its external losses are litter/soil/flood heterotrophic respiration and
DOC runoff, drainage, and flood export.

The following are internal transfers and must not alter the combined carbon
inventory:

- litter decomposition into DOC;
- POC decomposition into DOC;
- DOC decomposition into active, slow, and passive POC;
- free/adsorbed DOC equilibration;
- vertical DOC water transport and diffusion;
- cryoturbation;
- `PERMA_PEAT` inter-layer redistribution.

The exact source budget checks are in
`stomate_litter.f90::littercalc_leak` lines 2801-2869 and
`stomate_soilcarbon.f90::soilcarbon_leak` lines 1240-1284 and 2309-2360.

## Successor Comparison

### Aggregate Learned Transfers

This design would predict nonnegative aggregate sources, sinks, and internal
transfer fractions, then use a deterministic conservation decoder.

Advantages:

- fewer predicted values than a 48-step driver sequence;
- exact total-carbon closure can be imposed algebraically.

Problems:

- the current shards contain final stocks but none of the nine required
  transfer categories;
- endpoint-only training leaves many transfers non-identifiable;
- a daily aggregate loses source ordering, threshold crossings, layer-wise
  capacity limits, DOC adsorption/transport, and the pre-decomposition
  `deepC_peat` writeback;
- adding enough constraints to recover these semantics recreates much of
  `soilcarbon_leak` outside its audited owner.

This remains a fallback only if the exact-driver model proves unlearnable
under a frozen bounded gate.

### Learned Drivers Plus Exact Scan

The existing `_paper_compiled_ok_leak_fold` already executes all 48
source-order transitions in one `lax.scan`. It carries the 13 true mutable
states and derives litter controls, TF-DOC, decomposition activity,
32-layer moisture, DOC transfers, respiration/export, and `deepC_peat`
inside the source-backed process.

The neural output required by this design is the following 48-step driver
series:

1. `soil_mc`
2. `wat_flux`
3. `runoff_per_soil`
4. `drainage_per_soil`
5. `runoff2peat`
6. `canopy2ground`
7. `precip2ground`
8. `precip2canopy`
9. `temp_sol`
10. `tdeep`
11. `hsdeep`
12. `shumdiag_peat`
13. `resp_maint_part_radia`

Turnover, `bm_to_litter`, biomass, vegetation, SLA, root profile, parameters,
and static process controls already come from canonical state or the retained
condition contract. They are not duplicate driver labels.

This design is preferred because it:

- has no independent `deepC_peat` head;
- updates all carbon stores through the exact process owner;
- preserves sequential thresholds and within-day feedback;
- reuses the optimized whole-day JAX scan rather than introducing Python
  orchestration;
- permits exact replay checks before any network is trained.

## Existing-Data Audit

The machine audit is:

```bash
python -m scripts.dev.audit_daily_carbon_budget \
  outputs/research/daily_coarse_graining/dataset-manifest-v5/dataset_manifest.json
```

It reports:

```text
endpoint_stock_supervision: true
aggregate_transfer_supervision: false
exact_scan_driver_supervision: false
decision: capture_bounded_ok_leak_driver_auxiliary_dataset
```

The result is structural, not an inference from failed validation metrics.
The v5 shard arrays are `state_trajectory`, `fast_day_target`, native forcing,
conditions, and daily/final diagnostics. They do not store internal
`fluxtot`, DOC transport/diffusion/export, or the 13 driver series.

## Bounded Auxiliary Capture

The first asset is intentionally small and remains separate from Contract v5.
For each selected real day it stores:

- the 13 driver arrays above, each with exactly 48 source-order steps;
- source dataset, shard, landpoint, year, and day identity;
- Contract-v5 and Teacher commit hashes;
- the day-start canonical state hash;
- the existing endpoint `B_fast` hash;
- a driver-schema hash and per-array dtype/shape metadata.

The capture must be taken from the already materialized compiled entry stacks
and maintenance-respiration series. It must not rerun formulas in a data
writer. The implementation entry point is
`extract_ok_leak_driver_series`.

Initial scope:

- train only on unsealed train/train references;
- include multiple wetness, productivity, temperature, and peat conditions;
- use deterministic day selection;
- add model-selection references only for evaluation;
- do not capture all days or all 669 points until the architecture gate
  demonstrates value.

## Cheap Promotion Gate

Before paid training, all of the following must pass:

1. Captured Teacher drivers replay through the exact scan to the stored
   `OK_LEAK` endpoint within float64 process tolerance, including every carry
   field and `deepC_peat`.
2. Carbon budget closure is computed from the source inventory, with external
   sources and sinks counted once and all internal transfers cancelling.
3. No neural target or parameter group independently owns `deepC_peat`.
4. Perturbed predicted drivers cannot produce nonfinite or negative inventory
   stocks; any feasibility transform must operate on source drivers, not clip
   final stocks.
5. Forward and reverse passes through the 48-step scan are finite.
6. A tiny real-shard fit must not worsen the frozen parent's `carbon_32l` or
   DOC endpoint error and must preserve all exact mask/restart gates.

Failure of item 1 is a capture/plumbing defect. Failure of items 2-5 is an
architecture defect. Failure only of item 6 means the driver prediction is
not yet learnable at the bounded scale; it does not authorize a larger run.

The first real-day implementation check now closes item 1. For 1961 Day 2 at
train-only landpoint `001.0-071.0`, all 13 captured driver arrays and the
persisted-driver scan replay are bit-exact, all 14 `ok_leak.*` endpoints are
bit-exact, next continuous state closes at `2.84e-14`, and all discrete state
is exact. The full evidence and the isolated unrelated historical
`t2m_min_daily` drift are recorded in
[`ok_leak_driver_capture_probe_20260731.md`](ok_leak_driver_capture_probe_20260731.md).
Items 2-6 remain gates; this result does not authorize paid training.

## Implemented Guard

`research/daily_coarse_graining/carbon_budget_ownership.py` now provides:

- a complete source-provenanced ownership table;
- the exact 13-field auxiliary driver schema;
- strict 48-step extraction from real replay records;
- deterministic capture metadata and byte accounting;
- a Contract-v5 label-sufficiency audit.

The guard deliberately leaves all Teacher and neural execution paths
unchanged.
