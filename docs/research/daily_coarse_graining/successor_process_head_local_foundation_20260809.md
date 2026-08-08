# Successor Process-Head Local Foundation

Status: **accepted local synthetic foundation; real-v6 `gln01` smoke pending**.

Date: 2026-08-09.

## Decision Boundary

The selected formal pair is:

- candidate: `source_routed_process_attention_v2`;
- matched control: `source_routed_graph_mlp_control_v2`.

Both are bound by
`manifests/coarse_graining/successor_process_head_candidate_v1.json`, SHA256
`bb8bb34972e6fdff713f06f7ce48a2faa9eb4991af9f269c9aab26d96b592b5c`.
The v1 loader also pins this hash externally, so editing the contract and
recomputing its internal hash still fails closed.

## Implemented Foundation

`daily_operator/successor_architecture.py` owns the new parameter tree,
source-family tokens, 101 field queries, source masks, process graph, matched
attention/fixed-aggregation arms, constrained decoder, and named fast
parameter factors. It does not call the closed first-pair encoder or load a
historical checkpoint.

`daily_operator/successor_process_head.py` binds the new named outputs to the
accepted `ProcessPrediction` mapping. Only that scientific mapping is reused
from `NeuralProcessHead`; the old network and parameters are not invoked.

`daily_operator/successor_training.py` owns the free daily scan, reconstruction
of next-day PFT views from predicted Markov state, extraction of five
productivity/respiration trajectories and seven carbon stocks, combination
with the unit one-day anchor, and the single bounded train/train gradient-ratio
preflight. It never reads future Teacher state during free rollout and contains
no 48-step half-hour scan.

## Local Results

- candidate parameters: `8,679,681` float64 values;
- control parameters: `8,679,681` float64 values;
- source families: `20`;
- assembled input tokens in the frozen PFT14 synthetic contract: `770`;
- output fields: `101`;
- output scalar elements: `38,530`;
- eager and JIT outputs: finite;
- attention-kernel loss VJP and JVP: finite, with nonzero VJP norm;
- PFT permutation equivariance and inactive-slot isolation: passed;
- vegetation `vcmax25` and three maintenance-coefficient gradients: nonzero;
- bare-soil fast gradients and neural gradients to `alloc_min` and
  `residence_time`: zero as required;
- conservative one-day water/carbon residuals: within `2e-12`;
- 30-day trajectory extraction and unit-anchor/long-loss composition: passed;
- train/train-only coefficient preflight resolves the frozen
  `0.10/0.15/0.10/0.15` ratios and rejects non-train input;
- inference source audit: `19` modules, zero violations.

Commands and results:

```text
conda run -n ORCJAX python -m pytest -q \
  tests/unit/test_daily_operator_successor_process_head.py
10 passed

conda run -n ORCJAX python -m pytest -q \
  tests/unit/test_daily_operator_e1.py \
  tests/unit/test_daily_operator_training_foundation.py \
  tests/unit/test_daily_operator_parameter_routing.py \
  tests/unit/test_daily_operator_long_horizon_objective.py
55 passed
```

## What This Does Not Accept

The local sample is synthetic and uses plumbing retained-tail execution for
the composition test. It does not establish cold, ordinary, or restart
agreement for the canonical retained owner; it does not use real v6
statistics or targets; and it does not measure GPU memory, throughput, or a
real 30-day gradient. No checkpoint was created.

The next gate is a no-checkpoint real-v6 smoke on free `gln01` covering both
arms, accepted statistics, forward/loss/JVP/VJP, the complete canonical
retained tail, one-day state/mask/discrete/budget checks, and a bounded
train/train rollout/coefficient probe. Bounded small fit and 7/30-day model
selection remain closed until that smoke passes. Paid GPU work, 365 days,
long-held points, Gate D2, and product acceptance remain unauthorized.
