# Coordinate-Coupled DOC Transition v4 Local Acceptance

Date: 2026-08-20.

Status: accepted local foundation. One no-checkpoint free-`gln01` training
smoke is the active gate. This document does not authorize paid execution or
the 100-epoch diagnostic.

## Implemented Mechanism

`coordinate_coupled_doc_transition_attention_v4` preserves the v3
coordinate-query network and changes only the DOC handoff into
`inventory_transition`:

- all 896 compact DOC inventory coordinates gather five upstream DOC fields
  at the same full-PFT identity, deep layer, DOC pool, and element;
- each gathered token contains the upstream hidden coordinate and its decoded
  normalized scalar;
- the existing inventory cross-Attention consumes this fixed five-token bank;
- DOC incoming amount and outgoing fraction share one signed conservative
  transition decoder and are materialized as mutually exclusive endpoints;
- the existing conservative updater, exact reconstruction, parameter routing,
  retained tail, and all non-DOC output paths are unchanged.

The instantiated float32 parameter tree has exactly `59,971,257` parameters,
matching the frozen machine contract. It removes the two independent DOC
endpoint projections and remains below the `62,000,000` engineering guard.

## Local Gates

The focused v4 suite passes `7/7` tests in one process:

- contract and executable identity;
- exact coordinate alignment for all five sources and all 896 targets;
- exact parameter count, dtype, and joint-head ownership;
- finite positive- and negative-direction gradients through the joint
  endpoint materializer;
- complete eager/JIT 39,006-scalar learned boundary;
- canonical `bind_daily_operator` one-day transition, finite state,
  nonnegative inventories, and water/carbon budgets within `2e-12`;
- nonzero official litter/POC/DOC loss gradients through all five source
  projections, the scalar-value embedding, both reused inventory Attention
  blocks, and the joint decoder;
- all 42 litter/POC/DOC fields preserve inactive-PFT isolation and PFT
  permutation equivariance to float32 accumulation tolerance.

Related regressions pass in isolated processes so multiple large parameter
fixtures do not create an artificial local OOM:

| Suite | Result |
| --- | ---: |
| v4 focused | `7 passed` |
| qualification runner/launcher | `21 passed` |
| unchanged v3 candidate | `7 passed` |
| unchanged mechanistic-output base | `13 passed` |
| Total | `48 passed` |

Ruff, Python compilation, Bash syntax, JSON parsing, and `git diff --check`
also pass.

## Runner Boundary

The shared qualification runner and free-GPU launcher now expose
`--coordinate-coupled-doc-transition` /
`COORDINATE_COUPLED_DOC_TRANSITION=1`. The flag requires mechanistic output
and isolated `litter_poc_doc`, is mutually exclusive with the historical v3
flag, preserves logical batch size 8 through exact size-2 gradient
microbatches, and emits a distinct v4 report schema. Default and v3 paths keep
their previous behavior.

## Active Gate

Run exactly one fresh 364-day, one-epoch, 46-update smoke on one free `gln01`
GPU. It must write no checkpoint. It passes only if all updates and gradient
coverage complete, selected loss decreases, counts remain unchanged, all
values are finite, and state/budget/inventory hard counts remain zero. A pass
opens preparation of the unchanged 100-epoch absolute memorization gate; it
does not open multi-point training or rollout.
