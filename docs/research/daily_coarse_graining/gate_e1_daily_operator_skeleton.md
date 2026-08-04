# Gate E1: True-Daily Operator Skeleton

Status: active implementation packet; not yet implemented.

Date: 2026-08-04.

## Purpose

Gate E1 establishes the executable boundary for the new neural research
program. It does not choose a final network by accident, train a scientific
model, or claim surrogate-gradient equivalence.

Read [`failed_architecture_lessons.md`](failed_architecture_lessons.md) before
implementation. It records the rejected state-prediction, objective,
conditioning, carbon-stock, and 48-step designs that this skeleton must not
silently recreate.

The required transition is:

```text
canonical day-start state
+ native forcing records, timestamps, durations, and mask
+ named PFT parameters and traits
+ PFT fractions/mask and non-PFT static conditions
  -> process-structured daily predictions
  -> one conservative fast-process state update
  -> retained exact daily processes
  -> canonical next-day state and diagnostics
```

The candidate graph must not reconstruct 48 interpolated forcing records,
execute 48 recurrent state transitions, or predict an anonymous next-state
vector.

## Accepted Inputs

Use the existing accepted contracts rather than defining another dataset
boundary:

- canonical day-start state and exact discrete state from
  `daily_markov_contract.py`;
- native forcing from `DailyMarkovContract.native_forcing`, including record
  time, duration, predecessor/boundary context, and an explicit record mask;
- PFT arrays assembled through `daily_pft_interface.py`:
  `pft_state`, `pft_parameters`, `pft_traits`, `pft_fraction`, and
  `active_pft_mask`;
- non-PFT static landpoint and soil fields from the named condition/static
  registries, never landpoint identity;
- only the four first-wave physical-parameter families are required in the
  first executable model. Extension must remain name-based so second-wave
  channels can be added without changing process-head semantics.

The current paper forcing contributes one predecessor record and four current
six-hour records to a day window. The implementation must use a masked
variable-record sequence interface; the number five is data-contract metadata,
not a neural architecture constant.

## Required Outputs

The neural boundary predicts process quantities, not independent stocks. Its
typed output must cover the trainable portion of the frozen 50-label inventory
in `daily_flux_label_inventory_v1.json`:

- nonnegative external inputs and outputs;
- bounded outgoing inventory fractions;
- normalized nonnegative destination shares for internal transfers;
- signed quantities represented by declared positive/negative components;
- bounded thermal tendencies and explicit defined-status/mask outputs;
- compact daily inputs required by retained season/STOMATE processes.

Labels already owned by deterministic preprocessing or retained exact daily
formulas remain outside neural heads. Diagnostics and packet mirrors are
reconstructed from primary predictions and state, never separately learned.

## Conservative Update Boundary

`constrained_daily_replay.py` is the accepted Gate C Oracle for the update
semantics. Gate E1 must separate two roles that are combined in that diagnostic
module:

1. endpoint-to-label encoding, used only to prepare or verify Teacher labels;
2. a pure-JAX inference updater consuming predicted outgoing fractions,
   incoming amounts, transfer shares, and bounded tendencies.

The inference graph must never call `encode_inventory_endpoints` or inspect a
Teacher end state. It may reuse or extract the pure update formulas
`apply_bounded_inventory_transition` and `apply_bounded_tendency`. Water and
carbon identities must close by construction; clipping a negative stock after
the update is a failed design.

The source-backed retained daily tail remains exact. Existing
`canonical_retained_tail.py` and Gate C replay code are Oracles and composition
references, but their historical anonymous `physical_fast_day_target` and
runtime monkey-patching interfaces are not the final product API. Gate E1 may
extract a clean adapter around the accepted exact daily owners; it must not
copy their formulas into a neural head.

## Package Boundary

Place new code in a dedicated package such as:

```text
research/daily_coarse_graining/daily_operator/
  types.py
  native_forcing.py
  state_features.py
  process_heads.py
  conservative_update.py
  retained_tail.py
  model.py
```

Names may change when implementation reveals a better local boundary, but the
new package must not import or subclass rejected `canonical_*`,
`structured_canonical_*`, `axis_process_*`, `causal_carbon_*`, or
`rollout_stability_*` neural models. Reusable dataset readers and accepted
Teacher/Oracle utilities may be imported deliberately.

E1 should keep encoder and process-head protocols explicit enough that a later
architecture study can compare MLP, recurrent/attention forcing encoders, and
axis-aware interaction blocks without changing the scientific IO contract.

## Implementation Sequence

1. Define stable JAX PyTrees for day input, process prediction, conservative
   update result, retained-tail input, and final result.
2. Implement native-forcing masking, timestamps/durations, unit-aware feature
   assembly, and a minimal order-sensitive encoder protocol.
3. Implement state/PFT feature grouping without flattening away PFT, soil,
   snow, water, and carbon axes.
4. Define process-head output parameterizations that enforce sign, fraction,
   partition, and thermal-bound constraints.
5. Implement the pure-JAX conservative updater and bind every predicted field
   to one frozen inventory-label owner.
6. Compose the updated fast boundary with a clean retained exact daily-tail
   adapter.
7. Add local structural and one-day Oracle tests. A zero/identity or fixed
   test head is acceptable only for plumbing tests and must be named as such.

No optimizer, training schedule, GPU launcher, paid job, or large Teacher-data
regeneration belongs in E1.

## Acceptance Tests

Gate E1 is complete only when all of the following pass:

### Contract and graph

- all inputs and outputs are named PyTrees with stable shapes and float64
  policy where the Teacher contract requires it;
- current five-record forcing and at least two other legal masked sequence
  lengths pass without changing model code;
- an executable graph audit proves no call to 48-step forcing reconstruction,
  half-hour Teacher transition, or historical fast-target decoder;
- no landpoint ID, PFT ID embedding, Teacher endpoint, or next-day target is an
  inference input.

### PFT and masks

- PFT-axis permutation equivariance passes;
- inactive and zero-fraction slots cannot affect active outputs;
- bare soil and PFT14 retain distinct legal process masks;
- no new implementation hard-codes PFT index 13 or a fixed `n_pft`.

### Physics and lifecycle

- all outgoing fractions lie in `[0, 1]`, incoming amounts are nonnegative,
  and destination shares sum to one on defined active transfers;
- synthetic water and carbon conservation tests close at float64 numerical
  tolerance without post-hoc clipping;
- feeding true Gate C labels through the new updater and retained-tail adapter
  reproduces the accepted cold-continuation, ordinary-day, and restart-year
  cases at the existing Gate C tolerances;
- exact discrete state and defined-status masks survive the transition and a
  restart roundtrip.

### Differentiability

- eager and `jax.jit` outputs agree;
- forward and reverse gradients with respect to placeholder network weights,
  day-start continuous state, native forcing, and the four first-wave physical
  parameters are finite on bounded active cases;
- inactive and threshold-adjacent cases are classified rather than silently
  treated as smooth.

These are engineering differentiability checks only. They do not satisfy Gate
D2 because an untrained or identity test head has no accepted scientific
parameter response.

### Regression

- Gate B2 PFT-interface tests, Gate C inventory/capture/replay tests, Gate D1
  evidence validation, registry validation, Ruff, Python compilation, and
  `git diff --check` pass;
- ordinary Teacher behavior and `jax_orchidee/` remain unchanged. Any necessary
  Teacher-core edit must be reviewed separately and synchronized to `main`.

## Exit And Next Gate

After E1, pause for an architecture and data-readiness review. Gate E2 then
chooses and trains bounded candidate architectures using existing admitted
Teacher data plus only the supplemental daily labels actually required by the
frozen inventory. Promotion proceeds through supervised flux fit, one-day
state/budget accuracy, 7/30/365-day free rollout, held-out landpoints and
years, and performance measurement. Gate D2 follows only after one surrogate
candidate is scientifically accepted.
