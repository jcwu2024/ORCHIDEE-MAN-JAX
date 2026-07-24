# Daily Fast-Day Teacher Contract v5

## Boundary

The learned operator replaces the 48 half-hour physical transitions, daily
accumulation and maintenance, and 48 ordered OK_LEAK transitions:

```text
S[d] + F_native[d] + P -> B_fast[d]
B_fast[d] + S[d] -> retained season/STOMATE tail -> S[d+1]
```

The retained tail remains source-backed JAX. The network receives no
landpoint identifier. Forcing interpolation and forcing-owned daily values
remain deterministic preprocessing.

For the real PFT14 paper packet, contract v5 contains:

- `S[d]`: 3,854 float64 values and 12 exact discrete values;
- `B_fast[d]`: 2,855 float64 values;
- active compact PFT slices: bare soil (PFT1) and mangrove (PFT14).

The dimensions are generated from source-backed packet metadata.

## Change From v4

Contract v4 correctly removed the duplicated 93-field finalize packet from
`B_fast`, but it removed one true fast-owned output as well. `leaf_ci` is
carried in `S[d]`, updated by DIFFUCO during the fast day, and consumed on the
next day. It is not always persistent. A real 1962 Day 2 retained-tail gate
showed that every next-state leaf closed except `leaf_ci`.

Contract v5 preserves all 2,815 v4 target columns in their original order and
appends compact `sechiba_finalize_state.leaf_ci` as the final 40 columns. No
other finalize mirror is learned.

Existing v4 Teacher shards do not require regeneration. Their canonical
`state_trajectory` already stores exact `S[d+1].leaf_ci`; the audited migration
appends that slice to each target row, rewrites hashes, and emits a new
contract and dataset identity. Old v4 statistics and checkpoints are not
compatible with v5.

## Differentiable Retained Tail

The training path reconstructs canonical state and `B_fast` with pure JAX,
then executes retained season/STOMATE and projects canonical `S[d+1]` inside a
single differentiable transition. A fixed-shape `lax.scan` supports 1-, 3-,
and 7-day unrolled objectives with both one-step `B_fast` supervision and
next-state supervision.

Forcing-owned daily values are installed by deterministic JAX formulas.
Structurally inactive bare-soil maintenance pools are not allowed to inject
undefined reverse-mode derivatives; PFT14 maintenance pools retain gradients.

The real 001.0-071.0, 1962 Day 2 gate passed with:

- maximum canonical next-state absolute error `5.68e-14`;
- zero defined-status and exact-discrete mismatches;
- 2,847/2,847 defined target gradients finite;
- nonzero target-to-next-state gradient norm.

A three-day continuation using one `lax.scan` also passed with `1.71e-13`
maximum trajectory error, zero mask/discrete mismatch, and 8,541/8,541 finite
defined target gradients. This is the current real recursive-boundary gate;
the neural operator itself has not yet passed a v5 free rollout.

The Windows CPU reverse compilation emits the known XLA
`algebraic_simplifier` 50-run warning, then completes in about 5.8 seconds.
Because compilation terminates and all numerical and gradient gates pass, this
is classified as a compiler optimization-convergence warning, not a model
semantic failure. Keep it visible and A/B it when the pinned JAX/XLA runtime is
upgraded; do not disable scientific checks to suppress it.

## Shards And Undefined Values

The shard schema is `daily_teacher_markov_year_v5`. ORCHIDEE `+/-1e20`
sentinels remain exact non-numeric values. Persistent undefined outputs are
restored from `S[d]`; the registered dynamic `diffuco.rveget` status uses the
existing flip classifier.

Teacher Day 1 still performs the real cold-start bootstrap and creates
`S[1]`; the first supervised transition is Day 2. Restart years retain the
explicit year-start rebase.

Implementation authority is
`research/daily_coarse_graining/daily_markov_contract.py`. Dataset migration
is implemented by `migrate_markov_v4_to_v5.py`; differentiable rollout is in
`canonical_multistep.py` and `canonical_retained_tail.py`.
