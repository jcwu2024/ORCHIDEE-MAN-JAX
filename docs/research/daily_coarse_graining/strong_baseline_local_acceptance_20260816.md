# Strong-Baseline Local Structural Acceptance

Date: 2026-08-16

## Decision

The user-approved `axis_aligned_process_expert_operator_v1` now passes its
local structural gate. This is implementation and differentiability evidence,
not evidence that the network can fit one point-year or generalize.

The executable architecture is frozen by
`manifests/coarse_graining/strong_baseline_axis_aligned_process_expert_v1.json`
at SHA256
`baf5791f0786f22e05e8db9115eeab9a77687e19250d4d9efcb5181d6a96351d`.
It contains 49,750,681 trainable float32 parameters, within the approved
35-50 million target and below the 60 million hard cap.

## Implemented Boundary

- seven independently parameterized width-384, three-block process experts;
- two width-256 native-forcing Attention blocks without forcing expansion;
- source-family masks that prevent pre-mask forcing-family leakage;
- 18 canonical physical axes, coordinate-local state/static/process-static
  summaries, and output-axis slot projections;
- one shared factorized axial self-Attention/FFN block that mixes coordinates
  within each physical axis without dense cross-axis Attention;
- two bounded directed process-message rounds initialized at gate 0.01;
- one independently parameterized final MLP per scientific label;
- equal experts, then equal labels within each expert, process loss;
- float32 neural parameters and activations, followed by the existing float64
  constrained transform, parameter routes, conservative update, and loss;
- formal `StrongBaselineProcessHead` integration with the existing named-output
  scientific adapter and `bind_daily_operator` boundary.

The output schema covers 181 fields, 48 labels, and 39,371 scalars exactly.
The contract assigns every field to one of seven experts and records 412
static-to-axis plus 294 process-static-to-axis scalar links. Full-PFT outputs
gather compact day-start state only through the declared active PFT indices.

## Local Evidence

The following command passes:

```text
conda run -n ORCJAX python -m pytest -q \
  tests/unit/test_daily_operator_strong_baseline_architecture.py \
  tests/unit/test_daily_operator_strong_baseline_contract.py
```

Result after the factorized axial completion: 14 passed in 187.43 seconds.

Covered gates include:

- exact contract hash, expert ownership, axis registry, and output coverage;
- exact parameter count and all-parameter float32 dtype;
- eager and JIT forward over all 181 fields;
- finite float64 physical outputs with correct shapes;
- finite complete expert-balanced loss, representative JVP, and VJP;
- full-PFT/compact-PFT local-state mapping;
- a local soil-layer perturbation propagates to other coordinates through the
  declared factorized axial path;
- forcing-family source isolation before expert admission;
- inactive-PFT isolation across neural PFT inputs, continuous and discrete
  state, full-PFT static fields, process-static fields, and parameter routes;
- one synthetic bound daily transition with 3,854 next-state scalars and zero
  water/carbon budget residual under the plumbing retained-tail fixture.

The shared output-transform and fast-parameter-route extractions also preserve
the previous successor implementation. The combined strong-baseline and
relevant successor/parameter-routing regression run passes 39 tests in
333.29 seconds; Ruff, Python compilation, and the free-GPU launcher syntax
check also pass.

## Explicit Non-Claims

This acceptance does not claim:

- real Teacher-v6 forward or loss evidence;
- execution through a real canonical retained-tail lifecycle;
- one-point-year memorization;
- 64-point-year learnability;
- one-day validation accuracy or 7/30/365-day rollout skill;
- promotion, Gate D2, sealed-test access, or product acceptance.

No v6 Teacher data were regenerated and no historical v5 data were read or
joined.

## Active Next Gate

The zero-update real-v6 admission runner and free-`gln01` launcher are locally
ready. Before the fit, execute them to prove one real sample through eager/JIT forward, the
expert-balanced loss, representative gradients, the canonical retained-tail
lifecycle, and measured device memory. The qualification fit must use fresh
parameters, graph-message scale zero, all valid days from one train/train
point-year, and the preregistered per-expert and primary-output reduction
gates. Failure blocks the 64-point-year and paid full-data stages.
