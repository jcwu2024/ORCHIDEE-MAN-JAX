# Target-Oriented Hierarchical Coordinate Attention v1

Status: **frozen for local implementation; no GPU training authorized**.

Date: 2026-08-21.

## Decision

The first target-oriented candidate is
`target_oriented_hierarchical_coordinate_attention_v1`. It is a day-local
Transformer-style process operator, but it is not a widened copy of the
rejected DOC v4/v5 models. It replaces the old uniform label heads with a
hierarchy matched to the frozen scientific objective.

The machine contract is
`manifests/coarse_graining/target_oriented_attention_v1.json`. It binds the
accepted Teacher-v6 release, target-oriented objective, source-routed input
contract, conservative updater, and retained exact tail.

## Computation

The candidate consumes the day-start physical state, five native daily forcing
records, traits, static data, and the existing parameter routes. It does not
reconstruct 48 half-hour records, scan 48 state transitions, or carry a neural
hidden state between days.

The computation has four levels:

1. The accepted source-routed encoder builds seven process-domain tokens while
   preserving PFT, soil-layer, snow-layer, pool, part, and element axes.
2. Each domain receives additional process-specific Attention capacity. The
   five science-critical domains receive two refinement blocks; surface energy
   and finalization receive one.
3. Five science tokens connect GPP, autotrophic respiration, water stress,
   temperature, and slow carbon through explicit domain adjacency followed by
   cross-token Attention.
4. Every output coordinate queries its allowed source-family tokens, its own
   axis tokens, local start-state token, owning domain token, and admitted
   science tokens. Primary and state/budget fields use separate high-capacity
   coordinate decoders. Auxiliary fields use a separate 192-dimensional
   bottleneck and label-shared 128-dimensional heads.

This preserves the useful coordinate-aligned Attention evidence from v3-v5
without inheriting v4's joint signed endpoint or v5's soft-status-times-
magnitude materialization.

## Capacity Allocation

The inherited uniform 48-label output heads are removed. Capacity is instead
spent on 12 domain-refinement blocks, two science-to-domain cross-Attention
blocks, two science-token mixer blocks, three primary coordinate blocks, and
two state/budget coordinate blocks. Primary and state/budget fields have
independent final heads; the 35,298 auxiliary coordinates share low-capacity
heads by label.

The implemented parameter count is `83,755,173`, inside the frozen 80-100
million target interval. The 120-million ceiling is an engineering bound only,
not a promotion gate. Promotion is exclusively by the absolute target-oriented
7/30/365-day science and hard physical gates.

## Parameter Routing

`vcmax25` and `maint_resp_slope` remain explicit fast-process routes rather
than generic anonymous feature columns. `alloc_min` and `residence_time`
bypass the neural encoder and act in retained exact allocation and mortality.
No additional parameter becomes a learned input without a source-backed
ownership decision and controlled Teacher-v6 support.

## Output Boundary

All 161 learned fields remain materialized for the existing mechanistic
boundary. Their roles differ:

- GPP and maintenance respiration are direct primary science outputs.
- 93 state/budget-essential fields receive the main state transition capacity.
- 66 auxiliary fields retain typed supervision through smaller shared heads;
  their direct process loss remains downweighted by the frozen target-oriented
  objective.
- The 20 exact/diagnostic fields remain reconstructed outside the network.

Normalized predictions use the existing general constraint transform, then
the accepted conservative update and retained exact tail. No rejected v4/v5
physical endpoint decoder is reused. A `0.01` minimum normalized ratio keeps
sparse nonnegative outputs trainable when their frozen mean is zero; a
`0.0001` bounded-center floor serves the same purpose for unit-interval
coordinates without imposing a new process equation.

## Local Gate

Before any paid or free GPU training, the implementation must pass parameter
count and category routing, eager/JIT forward, finite loss and gradients for
every head family, inactive-PFT isolation, PFT permutation equivariance,
canonical binding, and a single-day state/budget check. Passing this gate only
authorizes a bounded small-scale training proposal; it does not promote the
candidate.

## Local Result

The local implementation passes its focused parameter/category, eager/JIT
forward, complete mechanistic loss/gradient, inactive-PFT isolation, strict
PFT-permutation, canonical binding, and single-day conservative budget tests.
The full learned boundary contains 161 fields and 39,006 coordinates, and all
defined eager/JIT outputs are finite. Water and carbon budget residuals remain
within `2e-12` through the bound daily transition.

This result authorizes preparation of a bounded real-v6 GPU lifecycle smoke.
It does not authorize submission, training, checkpoint retention, or promotion
without the separately approved resource and cost declaration.
