# Daily Markov Contract v2

## Purpose

The production research boundary is:

```text
S[d] + F_native[d] + P -> S[d+1] + Y[d]
```

`S` is the sufficient cross-day prognostic state, `F_native` is the original
6-hour forcing window, `P` contains parameters and static landpoint conditions,
and `Y` contains diagnostics that are not consumed by the next day. The learned
operator represents the complete scientific day, including SECHIBA, daily
accumulation, OK_LEAK, season, STOMATE carbon processes and day-end writeback.

## State ownership

Canonical fast-state leaves come from
`docs/source_audits/sechiba_state_field_contract.yaml`. STOMATE slow-state
leaves come from `slowproc_stomate_previous_step_state` and remain cross-day
state. The adapter applies these rules:

- omit `sechiba_finalize_state`; it is a diagnostic/restart mirror, not an
  independent owner;
- keep the source-backed driver, DIFFUCO, ENERBIL, HYDROL and THERMOSOIL
  contract leaves;
- use slowproc as owner for `lai`, `frac_nobio`, `veget_max`, `veget` and
  `tot_bare_soil`, and fail if the DIFFUCO mirror differs;
- omit reset `daily_accumulators`; their same-day values are `Y[d]` and their
  next-day zero state is deterministic;
- store float leaves once in `state_trajectory`;
- store integer, logical and mask leaves in exact typed
  `state_discrete__*` trajectories;
- represent the source-defined year-start removal of `hydrol.nroot`
  explicitly. Only `S[0]` after year rebase may fill that leaf with zero; a
  missing later-day value is an error.

The contract is generated from the day-end packet because cold/restart input
packets may intentionally omit a field that the first day produces. Every
shard records the complete leaf metadata and contract SHA256.

There is deliberately no fabricated canonical `S[0]` for the 1961 cold
start. HYDROL, THERMOSOIL and STOMATE first-call owners create required state
during Day 1, so production capture uses the real deterministic bootstrap:

```text
cold-start restart/static + Day 1 forcing -> canonical S[1]
S[1] + F_native[2] + P -> S[2] + Y[2]
```

The 1961 shard therefore contains Day 2--365 (`T=364`) and begins its stored
trajectory at Day 1 end. Before promotion, its generated year-end state must
match the current accepted 1961 checkpoint with exact schema/discrete leaves
and float64 `rtol=1e-12`, `atol=1e-12`; the report retains exact mismatch count
and maximum errors. Restart years retain the ordinary Day 1 transition after
the explicit year-start rebase.

`P` is ordered and named in the contract. Every flattened group records its
shape, dtype, start/stop slice, temporal role and source. The parameter groups are
`alloc_min`, `residence_time`, `vcmax25` and `maint_resp_slope`; landpoint
static groups include hydrology tables, vertical geometry, soil/PFT
coefficients, soil class/texture/bulk density/pH, coordinates, area/continent
fraction, grid geometry, measurement heights, salinity and the periodic tide
series. Annual CO2 is an `annual_exogenous` condition, not mislabeled as
landpoint static. Global constants shared by all samples remain code/config
constants rather than repeated shard columns.

The machine-audited ownership map at
`manifests/coarse_graining/daily_markov_input_ownership_v2.json` covers every
argument of the compiled complete-day Teacher transition. Its regression gate
fails if the executable gains an argument without a state/forcing/condition
owner.

`reconstruct_state_fields` inflates the two retained PFT slots into the full
Teacher shapes, restores the five slowproc-to-DIFFUCO mirrors, resets omitted
daily accumulators and refreshes same-name finalize mirrors. Its roundtrip gate
requires exact recovery of every canonical continuous and discrete leaf.
Non-active PFT slots are preserved from the fixed packet template rather than
invented or overwritten; only PFT1 and PFT14 dynamic slices are replaced.

## Native forcing

The nine stored source variables are `Tair`, `PSurf`, `Qair`, `Wind_E`,
`Wind_N`, `Rainf`, `Snowf`, `SWdown` and `LWdown`. A day stores five source
records: the interpolation predecessor and four current 6-hour records. The
first-day cyclic source rule is preserved as `[3, 0, 1, 2, 3]`; ordinary Day 2
uses `[3, 4, 5, 6, 7]`.

Deterministic preprocessing reconstructs all 48 Teacher inputs:

- linear interpolation for atmospheric state and longwave;
- precipitation spreading and conversion to model-step amount;
- source solar-angle redistribution and 2000 W m-2 cap for shortwave;
- wind component mapping, pressure conversion and measurement heights;
- annual CO2-derived canopy concentration, salinity and tide inputs.

The paper-case regression compares Days 1, 2 and 365 against
`_paper_compiled_forcing_day` for all 14 fields at `atol=1e-12`, `rtol=0`.
Observed arithmetic-order differences are at float64 rounding scale
(approximately `1.1e-16`).

## Shard arrays

```text
state_trajectory          [T + 1, D_state] float64
state_discrete__*         [T + 1, ...] original integer/logical dtype
forcing_native            [T, 5, D_native] float64
forcing_record_indices    [T, 5] int32
parameters                [D_parameter] float64
landpoint_static          [D_static] float64
annual_conditions         [D_annual] float64
diagnostics               [T, D_output] float64
year                      [] int32
day_index                 [T] int32
```

Finite masks are derived at load time. There is no stored `forcing_48`, no
separate `day_start_state` and `teacher_target` state copy, and no duplicated
finite masks. The old measured v1 shard was 224,333,416 bytes per
landpoint-year; the v2 schema has a regression gate requiring a representative
PFT14 year to remain below one tenth of that uncompressed size. A real 1962
Day 1 packet produced 231 state leaves, 6 discrete leaves, state width 3,724,
parameter width 84, landpoint-condition width 735 and diagnostic width 90.
The dominant annual arrays extrapolate to about 11.3 MB before NPZ container
overhead.

## Mandatory gates

- `state_trajectory[d+1]` exactly equals the next captured canonical state;
- discrete state is exact and has `T+1` entries;
- forcing and diagnostics have exactly `T` entries;
- plan split validation rejects spatial or temporal leakage;
- source/contract/shard/checkpoint hashes pass aggregation;
- 1961 uses `cold_start_bootstrap`, records Day 1 as non-training bootstrap
  metadata, and passes the strict accepted year-end checkpoint gate;
- no new v1 pilot shards are generated.

The implementation authority is
`research/daily_coarse_graining/daily_markov_contract.py`.
