# DOC Status And Branch-Magnitude Attention v5 Design

Date: 2026-08-20.

Status: frozen and implemented. Local forward/loss/gradient/state/budget
acceptance passed on 2026-08-21. No GPU training is authorized by this
document; the prepared 10-epoch early screen still requires explicit approval.

## Decision

The next candidate is
`coordinate_coupled_doc_status_branch_magnitude_attention_v5`. It retains the
complete v4 coordinate-aligned DOC Attention path and replaces only the final
one-scalar signed transition head and its DOC endpoint loss.

This is not a new backbone search. V4 proved that the five aligned upstream
DOC tokens repair much of DOC incoming. The zero-training geometry audit then
showed why its shared signed endpoint fails: incoming supplies `97.6209%` of
the two endpoint fields' combined zero-predictor squared error, although their
typical local Jacobians are balanced. The revised head separates status,
incoming magnitude, and outgoing magnitude while preserving the accepted
daily boundary.

## Head

For every DOC inventory coordinate, reuse the paired 512-wide v4 transition
hidden state and emit five scalars:

1. incoming status logit;
2. unchanged status logit;
3. outgoing status logit;
4. incoming magnitude coordinate;
5. outgoing magnitude coordinate.

The three status logits are initialized to the audited class prior
`[0.0748504, 0.8762663, 0.0488833]`. This controls initial physical leakage
without changing the equal-class training loss. Both magnitude kernels retain
the accepted small `1e-3` final-kernel initialization.

The incoming magnitude is decoded through the existing centered nonnegative
softplus coordinate. The outgoing magnitude uses a sigmoid centered at the
frozen output-normalization mean. With status probabilities
`p = softmax(logits)`, materialize:

```text
incoming_amount = p[incoming] * incoming_magnitude
outgoing_fraction = p[outgoing] * outgoing_magnitude
```

This soft gate deliberately avoids hard `argmax`. It keeps weight and future
physical-parameter gradients defined, keeps incoming nonnegative and outgoing
inside `[0, 1]`, and preserves exact carbon-budget accounting through the
unchanged conservative updater. The two endpoints may both be softly nonzero
while status is uncertain; their probability mass shares one simplex and the
Teacher status loss drives it toward the observed exclusive classes.

## Direct Coordinate Supervision

Teacher status is incoming when incoming amount exceeds `1e-12`, outgoing
when outgoing fraction exceeds `1e-12`, and unchanged otherwise. The accepted
v6 audit has no simultaneously active coordinate.

On incoming-active coordinates, invert the exact centered softplus
materializer and directly supervise its raw coordinate. On outgoing-active
coordinates, invert the centered sigmoid and directly supervise its raw
coordinate. Each uses mean smooth-L1 with delta `1.0`. The physical transform
Jacobian is therefore not on the supervised coordinate-gradient path; small
physical targets no longer starve their magnitude head.

The DOC composite is the equal average of:

1. equal-class macro status cross entropy;
2. incoming-active raw-coordinate smooth-L1;
3. outgoing-active raw-coordinate smooth-L1;
4. equal-status macro physical endpoint MSE under the frozen field scales.

The existing `carbon.ok_leak_inventory_endpoints` label also contains eight
non-DOC fields. Its revised value is one half their unchanged pooled loss and
one half the DOC composite. The ordinary pooled DOC endpoint contribution is
removed rather than duplicated.

## Capacity And Control

The v4 scalar head has 513 parameters. V5 replaces it with a 1,539-parameter
status head and two 513-parameter magnitude heads, for expected total capacity
`59,973,309`. The 62-million engineering ceiling remains a memory guard, not
a scientific gate.

No new long control run is authorized. Completed v4 job `14627588` is the
same-backbone, data, seed, and schedule mechanism reference. It is used only
to attribute whether the revised head removes the incoming/outgoing trade;
promotion depends solely on absolute gates.

## Acceptance Ladder

Local acceptance requires exact aligned-source coverage, finite eager and JIT
forward, valid status probability simplexes, bounded endpoints, finite
official loss and gradients, nonzero gradients through all five source
projections, status head, both magnitude heads, and inventory Attention,
inactive-PFT isolation, strict PFT permutation equivariance, and one complete
bound day with accepted state and budgets.

Only local acceptance may open a no-checkpoint early-stop screen. Review at
epoch 5 and stop immediately on any hard failure or if incoming or outgoing
loss exceeds twice its initialization. The screen may run at most 10 epochs.
At epoch 10 it must reduce aggregate litter/POC/DOC loss by at least `90%`,
both DOC endpoint losses by at least `50%`, and must not regress
free-to-adsorbed or advective transfer. Failure closes this candidate without
a 100-epoch run.

Only that screen can open the unchanged final 100-epoch absolute gate:
aggregate reduction at least `99%`, with every initially dominant field
reduced at least `95%` or ending at or below `1e-4`. Multi-point training,
rollout, sealed tests, Gate D2, and product acceptance remain closed.

The machine contract is
`manifests/coarse_graining/doc_status_branch_magnitude_attention_v5.json`,
SHA256 `46f442b2f003af80b4037ba8128e1543f1c031305ed844c9d6f30f7d4ea4142e`.

Implementation and local evidence are recorded in
[`doc_status_branch_magnitude_attention_v5_local_acceptance_20260821.md`](doc_status_branch_magnitude_attention_v5_local_acceptance_20260821.md).
