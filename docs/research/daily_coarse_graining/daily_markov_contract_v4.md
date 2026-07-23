# Daily Fast-Day Teacher Contract v4

## Boundary

The learned operator replaces the 48 half-hour physical transitions, daily
accumulation and maintenance, and 48 ordered OK_LEAK transitions:

```text
S[d] + F_native[d] + P -> B_fast[d]
B_fast[d] + S[d] -> retained season/STOMATE tail -> S[d+1]
```

The retained tail remains source-backed JAX. The network does not receive a
landpoint identifier and does not learn deterministic forcing interpolation.

For the real PFT14 paper packet, the compact contract is:

- `S[d]`: 3,854 float64 values and 12 exact discrete values;
- `B_fast[d]`: 2,815 float64 values;
- active PFT slices: bare soil (PFT1) and mangrove (PFT14).

The dimensions are generated from packet metadata and are regression evidence,
not hard-coded model constants.

## Changes From v3

Contract v3 incorrectly treated the complete 93-field
`sechiba_finalize_state` packet as a learned output mirror while omitting some
of its true cross-day carry from `S[d]`. Contract v4:

- removes `sechiba_finalize_state` from `B_fast`;
- carries only the fields consumed across days by
  `_SECHIBA_HALF_HOUR_CARRY_FIELDS`, including `leaf_ci`;
- keeps reset daily accumulators out of prognostic state;
- preserves exact discrete state separately;
- retains the explicit year-start `nroot` missing policy.

The carried finalize subset is not a second free prediction target. It exists
only where the production transition consumes the previous value. The full
restart/finalize packet remains reconstructable by deterministic owner code.

## Undefined Values

ORCHIDEE uses `+/-1e20` as source-defined undefined sentinels. Training and
evaluation define a numeric value as valid only when it is finite and
`abs(value) < 5e19`.

Persistent undefined outputs are restored exactly from `S[d]` rather than
regressed. A representation audit fails if an undefined value changes without
an explicitly registered source owner. The only current dynamic exception is
`diffuco_previous_step_state.rveget`: Fortran resets it to `undef_sechiba` and
writes a finite value only on the `assimilate` branch. The neural model uses a
separate binary head for that defined/undefined transition and a continuous
head for the finite value.

## Learned Representation

Every matching `B_fast` leaf is mapped to its same-owner slice in `S[d]`.
The model predicts a normalized residual around that persistence baseline.
Unmatched outputs remain mean-centered. This changes the representation, not
the Teacher labels or retained scientific transition.

Statistics use schema `daily_teacher_training_statistics_v2`. Version 1
statistics and contract-v3 checkpoints are rejected rather than silently
loaded.

## Cold Start And Shards

There is no fabricated 1961 `S[0]`. Teacher Day 1 performs the real cold-start
bootstrap and creates `S[1]`; the first supervised transition is Day 2.
Restart years begin with the explicit year-start rebase.

Shard schema `daily_teacher_markov_year_v4` stores one `state_trajectory`, one
`fast_day_target`, native six-hour forcing windows, exact discrete state,
conditions, diagnostics, and hash-linked contract metadata. It does not store
48 interpolated forcing copies, duplicate day-start/day-end state, finite
masks, or full finalize mirrors.

## Evidence Gate

Before a v4 dataset is accepted:

- compact capture must match the ordinary Teacher state, target, diagnostics,
  and final state;
- state extraction/reconstruction and retained-tail handoff must pass;
- the undefined-value representation audit must report zero unexpected
  persistence mismatches;
- all shards must share one contract hash and pass plan/source/checkpoint
  hashes;
- statistics must be regenerated as v2 from train/train shards only.

The accepted five-point v3 dataset, its v1 statistics, and its first neural
checkpoint are historical diagnostic evidence only. They are not valid inputs
to v4 training.

Implementation authority is
`research/daily_coarse_graining/daily_markov_contract.py`; training validity
and persistence mapping are implemented in
`research/daily_coarse_graining/markov_dataset.py` and
`research/daily_coarse_graining/canonical_training.py`.
