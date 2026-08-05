# Gate E1: True-Daily Operator Skeleton

Status: **accepted**. Gate E1 is complete; Gate E2 has not started.

Date: 2026-08-05.

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

## Implementation Evidence And Review Result

The Gate E1 candidate skeleton is implemented under
`research/daily_coarse_graining/daily_operator/`. The package provides named
JAX PyTrees, a masked variable-record native-forcing encoder, grouped
state/PFT features, explicitly constrained process heads, a pure conservative
inventory updater, a narrow retained-tail callable adapter, a one-day
composition root, and source/JAXPR graph audits. The included minimal linear
head and identity retained-tail adapter are named unit-test plumbing fixtures;
they are not promoted architectures and must not be trained as scientific
candidates. Real lifecycle and differentiation evidence uses the formal
canonical retained-tail adapter. The accepted review is
[`gate_e1_architecture_data_readiness_review.md`](gate_e1_architecture_data_readiness_review.md).

`tests/unit/test_daily_operator_e1.py` passes 30 local structural tests. These
cover 3-, 5-, and 7-record forcing inputs; masked padding and record order;
float64 input PyTrees; PFT permutation, bare-soil/PFT14 process masks, and
inactive-slot isolation; bounded fractions, nonnegative inputs, normalized
transfer shares, signed storage/debt and energy components; water/carbon
conservation without clipping; deterministic/exact label ownership outside
neural heads; true endpoint-label plumbing; exact masks,
discrete state and restart serialization; eager/JIT agreement; finite forward
and reverse gradients; explicit inactive/threshold-adjacent classification;
frozen-label ownership; and source/JAXPR graph audits. They also load the
frozen v5 contract and reject contradictory cached water, carbon, thermal,
litter/turnover, lignin, and process-static views. The staged five-record test
graph contains one scan of length five, no callback primitive, and no scan of
length 48.

The local real-Teacher Oracle command is:

```bash
python scripts/dev/verify_gate_e1_daily_operator_skeleton.py
```

It passes the cold-continuation, ordinary later-day, and restart-year evidence
matrix. Each case actually executes `daily_operator_transition`, the injected
true-label Oracle head, the new conservative updater, and the package-level
`CanonicalRetainedTailAdapter`. The adapter consumes predicted daily fields
and the predicted OK_LEAK boundary and does not inject the captured pre-step
boundary. All cases reproduce every continuous and discrete canonical
next-state leaf plus 26 modelout fields and 4 modelout leaves. The largest
canonical state error is below `3.94e-13`; the largest water, carbon, and
flux-side thermal residuals are `4.698463840213662e-13`,
`8.6811269284226e-9`, and
`6.984919309616089e-9`. The accepted restart roundtrip composes successfully.
Endpoint encoding occurs only before Oracle-head construction; it is absent
from `daily_operator_transition`, the Oracle `__call__`, and the audited JAXPR.
The real adapter passes eager/JIT agreement to `5.69e-14`, finite forward JVP,
and finite reverse VJP. Its graph contains only length-5 and length-14 scans,
no callback primitive, and no length-48 scan.
The generated comparison is
`outputs/research/daily_coarse_graining/gate_e1_daily_operator_skeleton/comparison.json`
with local SHA256
`83238ee006f4f705bac825a936e63241a60b0c3f35e78acf3168a5753b2d71cd`.

No optimizer, training run, paid task, Teacher-core edit, or 669-point data
regeneration was performed. The formal review confirms unique canonical input
assembly and a reusable differentiable canonical retained-tail adapter. Gate
E1 is accepted and Gate E2 has not started.

The final targeted regression run passes 70 tests covering Gate B2, Gate C,
Gate D1 evidence, the physical-parameter registry, and Gate E1. Changed-file
Ruff, Python compilation with an external temporary bytecode cache,
`git diff --check`, and the unchanged-`jax_orchidee/` check also pass.
