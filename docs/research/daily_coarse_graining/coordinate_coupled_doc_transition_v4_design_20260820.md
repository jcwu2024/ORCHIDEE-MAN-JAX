# Coordinate-Coupled DOC Transition v4 Design

Date: 2026-08-20.

Status: approved mechanism direction; frozen before implementation. Local
implementation and local acceptance are authorized. GPU execution is not
authorized by this document.

## Decision

The next neural process-head candidate is
`coordinate_coupled_doc_transition_attention_v4`. It retains the complete
`hierarchical_coordinate_query_attention_v3` architecture except for the DOC
handoff into the `inventory_transition` stage and the two DOC inventory
endpoint heads.

The candidate remains a true daily feed-forward operator. It consumes the
canonical day-start state, native six-hour forcing, explicit PFT inputs, and
owned static/parameter channels. It emits the unchanged mechanistic-output v2
boundary, enters the unchanged conservative updater, and then enters the
canonical retained exact tail through the required `bind_daily_operator`
boundary. It has no 48-record forcing reconstruction, 48-step state scan, or
learned cross-day memory.

## Evidence Addressed

The v3 isolated litter/POC/DOC run reduced loss `98.5804%` and passed `25/28`
dominant fields. The remaining failures are DOC incoming amount,
free-to-adsorbed transfer, and advective transfer. A zero-training audit shows
that these targets are active on every one of the 364 frozen days and have
absolute daily-mean correlations from `0.8403` to `0.9597`.

The current v3 graph does not expose that relationship at the output
coordinate level. DOC reaction and transport stages are averaged within each
field and then across fields. Every inventory coordinate receives those
global stage averages, not the predicted transfer at its own PFT, soil layer,
DOC pool, and element.

## Coordinate-Aligned Transfer Bank

Each DOC inventory coordinate has axes
`[landpoint, pft, deep_layer, doc_form, doc_pool, element]`. The candidate
builds a bounded bank of five already-computed upstream DOC representations:

1. gross DOC decomposition;
2. flooded gross DOC decomposition;
3. free-to-adsorbed equilibration;
4. advective interface transfer;
5. diffusive interface transfer.

For every inventory coordinate, each source is selected at the same PFT,
deep layer, DOC pool, and element. Compact inventory PFT coordinates are
mapped through the frozen `active_pft_indices` before matching the full
14-slot transfer fields. The inventory `doc_form` remains in the query so the
same signed adsorption or transport representation can have different roles
for free and adsorbed DOC.

Each aligned token contains the upstream hidden coordinate state plus an
embedding of its predicted normalized scalar. The existing inventory-stage
cross-Attention reads these five tokens together with its existing
source-family, axis, day-start-owner, and pooled-stage tokens. Bank size is
constant per coordinate and there is no dense Attention over the 39,006
learned scalars.

## Joint Conservative DOC Transition

The v6 DOC incoming amount and outgoing fraction are a mutually exclusive
piecewise encoding of one inventory transition. Predicting them with
independent heads ignores this exact relationship. V4 therefore combines the
two inventory coordinate states and predicts one signed conservative DOC
change.

The signed change is lower-bounded smoothly by negative day-start DOC. It is
then materialized exactly as:

```text
incoming_amount = max(change, 0)
outgoing_fraction = max(-change, 0) / day_start_DOC
```

The zero-start case has zero outgoing fraction. This is not anonymous direct
next-state prediction: it is the existing source-owned conservative endpoint
encoding with one shared learned coordinate and the existing updater remains
the sole state owner. Incoming and outgoing are nonnegative, outgoing stays
in `[0, 1]`, and both cannot be active together.

## Capacity And Control

The candidate reuses the existing 384-wide, eight-head inventory Attention.
It adds five source projections, one scalar-value embedding, and one
512-wide joint transition decoder while removing the two independent DOC
endpoint projections. Expected parameter count is `59,971,257`. The
`62,000,000` limit is an engineering guard, not a scientific promotion gate;
actual activation memory must be measured by the free-GPU smoke.

No new weak control is trained. The completed v3 job `14619803` is the matched
data/seed/schedule mechanism control and is used only for attribution. V4 is
promoted only by its absolute gates; being slightly better than v3 is not
sufficient.

## Bounded Acceptance

Local acceptance requires:

- exact five-source alignment for every admitted DOC inventory coordinate;
- complete eager/JIT output identity and finite values;
- finite official loss and gradients, with nonzero gradients through every
  aligned source projection, scalar-value embedding, reused inventory
  Attention block, and joint transition decoder;
- exact DOC endpoint mutual exclusivity and reconstruction of the represented
  inventory change;
- inactive-PFT isolation and strict PFT permutation equivariance;
- one complete bound day with finite state, nonnegative inventories, and
  accepted water/carbon budgets;
- no forbidden inference path.

Only local acceptance can open one no-checkpoint free-GPU epoch. A passing
smoke can open the same 364-day, 100-epoch isolated diagnostic. That run must
reduce aggregate litter/POC/DOC loss at least `99%` and make every initially
dominant field fall at least `95%` or finish at or below `1e-4`. No checkpoint
is retained before this absolute gate passes. Multi-point training, rollout,
sealed test, Gate D2, and product acceptance remain closed.

The machine contract is
`manifests/coarse_graining/coordinate_coupled_doc_transition_v4.json`, SHA256
`264ad569dd2c71e50e747187e65f470ef36c3daeaf325c9904573d520a20313f`.
