# Shared remaining-state source audit: OK_LEAK litter/fuel owners

Date: 2026-09-21  
Branch/HEAD audited: `research/daily-coarse-graining` / `8e17b48`  
Scope: read-only source audit of `endpoint.ok_leak_aux.dead_leaves` and
`fuel_1hr`, `fuel_10hr`, `fuel_100hr`, `fuel_1000hr`; no model, runner,
test, dataset, or job changes.

## Conclusion

The five outputs are not fluxes or independent daily summaries. They are the
absolute end-of-fast-boundary values of persistent STOMATE state carried by
the 48-call `OK_LEAK` fold:

- `dead_leaves` is a nonnegative leaf-litter tracer used to derive dead-leaf
  cover. It overlaps the litter stock; it is not an additional carbon pool.
- `fuel_1hr`, `fuel_10hr`, `fuel_100hr`, and `fuel_1000hr` are nonnegative
  size-class partitions of aboveground litter. They overlap that litter stock
  and must not be added again in carbon closure.

For the active paper lifecycle, each has a same-shaped day-start canonical
owner at
`slowproc_stomate_previous_step_state.<field>`. The fast endpoint is written
unchanged into that owner at day end and becomes the next day's initial
`OK_LEAK` carry. Therefore the scientifically appropriate learned quantity is

```text
fast OK_LEAK endpoint - day-start slowproc owner
```

with the nonnegative constraint still applied to the reconstructed absolute
endpoint. The present absolute-target representation has no field-specific
scientific justification in the audited sources. It comes from one blanket
contract-builder rule: every `daily_interface.*` and every `ok_leak.*` neural
leaf is absolute (`scripts/dev/build_successor_fast_owner_reconstruction_contract.py:477-480`).

This should not be implemented as a five-name error-threshold exception. The
same owner semantics apply to all eight neural `ok_leak` exact-scan state
leaves: `lignin_struc_above`, `lignin_struc_below`, `litterpart`,
`dead_leaves`, and the four fuel fields. The first three are bounded ratio or
partition state rather than carbon stocks, but they are also persistent
day-start owners and exact scan writebacks. The other `ok_leak` fields are
already conservative-updater outputs or an exact derivation, so they are not
part of this neural representation change.

## Teacher label semantics

The v6 label is the physical, teacher-native endpoint of the final call in the
48-step compiled `OK_LEAK` scan, before daily `StomateLpj` installs the next
day's `turnover_daily` and `bm_to_litter`:

- The Markov target inventory reads `record.ok_leak_updates` and labels it
  `"stomate_lpj:OK_LEAK half-hour fold"`
  (`research/daily_coarse_graining/daily_markov_contract.py:585-635`).
- The fold initializes these fields from the day-start slowproc state and
  advances one carry through all calls
  (`jax_orchidee/driver/orchestration.py:8476-8523`, `8672-8707`). Its source
  order explicitly keeps the previous day's `turnover_daily` and
  `bm_to_litter` fixed until after the final call
  (`jax_orchidee/driver/orchestration.py:8497-8502`).
- The frozen contract stores the five absolute slices at
  `v6.fast_day_target[1639:1659]`, with active-PFT compact shapes and
  teacher-native units
  (`manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json:5174-5375`).
- Target construction first reads the absolute retained-boundary value and
  subtracts a base only when the schema says `DAY_START_DELTA`
  (`research/daily_coarse_graining/daily_operator/successor_fast_owner_training.py:52-64`,
  `96-116`). Thus today's targets and fitted means/scales are absolute stock
  statistics by construction, not daily-change statistics.

Fortran confirms the stateful interpretation. `stomate_main` scales the
carried daily transfers to one SECHIBA step, then passes all five arrays as
`INTENT(inout)` state into `littercalc_leak` before the `do_slow` daily block
(`fortran_source/ORCHIDEE/src_stomate/stomate.f90:3288-3300`, `3587-3589`).
`dead_leaves` receives litter additions and decomposition and is then used to
recompute dead-leaf cover; each fuel class receives source-size fractions and
loses its proportional share of decomposed aboveground litter
(`fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90:2138-2317`,
`2577-2725`, `2798-2799`). All five are restart variables read and written by
`stomate_io` (`stomate_io.f90:1056-1098`, `2729-2752`). The local ownership
ledger consequently classifies `dead_leaves` as an `overlapping_tracer` and
the fuels as `overlapping_partition`, all with `exact_scan_writeback`
(`research/daily_coarse_graining/carbon_budget_ownership.py:104-138`).

## Retained tail and later consumers

The retained-tail route does not apply another litter/fuel transition to these
five values on the active paper path:

1. The predicted physical fast boundary is reconstructed and supplied to the
   retained runtime (`research/daily_coarse_graining/canonical_retained_tail.py:230-280`).
2. The runtime overlays the final `OK_LEAK` carry into slowproc before daily
   STOMATE (`jax_orchidee/driver/orchestration.py:11009-11059`).
3. After daily carbon, the runtime states explicitly that `StomateLpj` has
   installed the following day's fluxes while the `OK_LEAK` result remains the
   current day's final half-hour state
   (`jax_orchidee/driver/orchestration.py:11072-11085`).
4. `_paper_day_first_stomate_state_packet` takes `dead_leaves` and the fuels
   directly from that `OK_LEAK` result
   (`jax_orchidee/driver/orchestration.py:9021-9030`), and
   `_paper_day_end_state_packet` writes them into
   `slowproc_stomate_previous_step_state`
   (`jax_orchidee/driver/orchestration.py:9210-9238`). A multiday parity test
   asserts these exact identities (`tests/parity/test_driver_1961_orchestration.py:1989-2026`).
5. On the following day they are required model-produced STOMATE entry state
   (`jax_orchidee/driver/orchestration.py:1315-1374`, `1430-1458`) and are the
   initial carry of the next 48-call `OK_LEAK` fold
   (`jax_orchidee/driver/orchestration.py:8516-8523`). `dead_leaves` also
   determines `deadleaf_cover`; the fuel partitions participate in the next
   litter decomposition/update. Generic land-cover/fire kernels have further
   fuel consumers, but those do not introduce a second retained-tail rewrite
   in the audited paper lifecycle.

Accordingly, for these coordinates and this lifecycle,

```text
next canonical state - day-start state
== fast OK_LEAK endpoint - day-start slowproc owner
```

up to the contract's active-PFT selection and numerical identity. This is a
source-backed equality, not an assumption that retained tails generally leave
all fields unchanged.

## Why delta is preferable here

The decoder already supports the required physics. A delta field binds a
day-start state slice (`successor_fast_owner_decoder.py:105-141`), subtracts it
for training and adds it back for physical materialization
(`successor_fast_owner_decoder.py:182-225`). For nonnegative fields,
`output_transform.py` centers at `day_start + TRAIN delta mean`, uses the delta
scale, and applies softplus to the reconstructed endpoint
(`research/daily_coarse_graining/daily_operator/output_transform.py:53-88`).
Negative daily changes remain representable whenever the resulting endpoint
is nonnegative. Therefore absolute prediction is not needed to preserve the
physical bound.

The existing implementation does need an explicit owner mapping. Merely
changing the accepted contract's `value_representation` would fail:
`_day_start_owner_key` currently returns `leaf.key` except for
`final_diagnostics.temp_sol` (`successor_fast_owner_decoder.py:67-70`), but
`ok_leak.dead_leaves` must map to
`slowproc_stomate_previous_step_state.dead_leaves`, and likewise for the other
seven neural exact-scan state fields. A decoder-local temporary field spec can
instead supply this mapping without changing the external contract.

## Adopted bounded change and verified statistics gate

The bounded gate has now been run over all 16 existing prepared TRAIN
artifacts (512 samples). For every coordinate of all eight neural `ok_leak`
exact-scan endpoints, the fast target equals its corresponding
`expected_next_state.other` owner slice. The reported aggregate diagnostics
are `tail_max_abs = 0` and `tail_nonzero_count = 0`. This admits reuse of the
existing full-TRAIN `state_delta` moments for the decoder-local delta
calibration; no v6 regeneration or substitute target construction is needed.

The implementation boundary selected by the parent is narrower than changing
the accepted representation/owner contract:

1. Keep the v1 owner contract, v6 cached absolute targets, physical output
   schema, typed targets, and existing supervision/loss normalization
   unchanged.
2. Only inside the shared decoder, construct a temporary `OutputFieldSpec`
   that maps each of the eight `ok_leak` endpoint leaves to
   `slowproc_stomate_previous_step_state.<path>` and uses a day-start-delta
   learning coordinate with full-TRAIN delta calibration.
3. Pool calibration moments across PFT positions using counts, means, and
   variances together, including the between-group mean term and only valid
   active-PFT observations. This preserves PFT-permutation equivariance;
   unweighted averaging of per-PFT variances would not.
4. Invert the calibration and add the day-start owner before applying the
   endpoint's nonnegative or bounded physical transform. Pass that reconstructed
   absolute endpoint to the unchanged typed-target comparison and original loss
   normalization. Thus the target and objective remain absolute even though
   the shared head's learned coordinate is a normalized daily residual.
5. Record this decoder-internal convention in the new shared-v3 protocol. Do
   not revise the meaning of historical v1 or regenerate v6.
6. Add focused tests for owner resolution/shape, pooled-moment aggregation,
   residual inversion, nonnegative negative-delta reachability, inactive-PFT
   masking, physical endpoint equivalence, unchanged absolute supervision, and
   the eight-field fast-endpoint/day-end identity. Preserve the canonical
   updater, retained tail, all output capacity, and existing state and
   trajectory objectives.

Under those boundaries, the coordinate change is physically and
supervision-equivalent: it changes conditioning and gradient parameterization
inside the shared decoder, not the predicted physical quantity or the target
being scored. Two implementation details are essential to that conclusion:
the base must be the matching sample's day-start slowproc owner after the same
active-PFT selection/order as the endpoint, and all physical constraints must
act on `base + residual`, not on the residual alone. The 512-sample identity
gate closes the previously outstanding concern about reusing next-state delta
moments for this active lifecycle.

The user's reported TRAIN magnitudes are consistent with this source diagnosis:
absolute fuel/dead-leaf scales are orders of magnitude larger than their daily
change scales. Those statistics support urgency but are not the reason for the
representation decision; owner and update semantics are.

## Limits

- The codebase-memory MCP graph was unavailable (`Transport closed`) before
  this delegated audit. Per the parent handoff, no repeated MCP calls were
  made. Evidence here is a Verify-style targeted source fallback using exact
  Python, manifest, Fortran, and parity-test locations; graph generation and
  index coverage could not be certified.
- No Teacher shards, prepared artifacts, statistics archives, tests, or model
  executions were run by this delegated source-audit task. The subsequent
  512-sample TRAIN identity-gate result and pooled-moment decision are supplied
  by the parent task and recorded here; they were not independently recomputed
  in this audit.
- The equality between fast endpoint and next state is established for the
  active paper lifecycle implemented by the canonical retained tail. Broader
  land-cover/fire configurations may update fuel pools elsewhere and require a
  separate branch audit before reusing next-state-delta statistics.
- This audit does not decide whether other absolute `daily_interface.*`
  outputs should be deltas; many are daily summaries/fluxes without a
  persistent same-shaped owner and must remain a separate semantic class.
