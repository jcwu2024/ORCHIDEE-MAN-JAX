# Hydrology And Canopy Output v2 Local Acceptance

Date: 2026-08-17.

Status: local gate accepted; the subsequent zero-update real-v6 admission also
passed. The one-epoch training smoke is now the only open operation.

## Decision

The user approved the hydrology/canopy revision proposed by
`hydrology_canopy_boundary_audit_result_20260817.md`. The implementation keeps
the axis-aligned seven-expert Attention backbone and changes the output
mechanism around the conservative update. The active contract is
`manifests/coarse_graining/strong_baseline_mechanistic_output_v2.json`.

The complete boundary remains 181 fields and 39,371 scalar values. The neural
head now emits 161 fields and 39,006 scalar coordinates; 20 fields are derived
by source-backed daily-boundary kernels. The initialized model contains
49,743,001 float32 parameters.

## Implemented Mechanism

Final `nroot` is represented by a support-aware sparse simplex. A zero neural
coordinate preserves the day-start distribution exactly. A nonzero coordinate
can deactivate a previously active root layer but cannot reactivate a layer
whose day-start weight was zero. The output remains nonnegative and normalized
on each active vegetation PFT.

After the float64 conservative update, final `mc`, `mcl`, and predicted
`nroot` reconstruct `us`, `humrelv`, and `humrel` in one JAX boundary call.
Tile-level `evap_bare_lim_ns` is aggregated to grid `evap_bare_lim`.
`temp_hydro` remains learned because the audit showed a last-call phase
dependency that is not available at the daily boundary.

Canopy water is represented by one signed final `qsintveg` endpoint and three
storage/zero/debt status logits. The endpoint is capped only above by physical
canopy capacity; debt remains unbounded by signed daily `vevapwet`. One
deterministic converter maps the signed endpoint to the original four
nonnegative storage/debt transfers consumed by the conservative updater.
Those four transfers are no longer independently optimized. The canopy loss
is the equal-weight mean of normalized endpoint MSE and three-class status
cross-entropy.

`ProcessPrediction.auxiliary_outputs` carries the endpoint, its scale, and the
status logits from the neural head to the loss without changing the canonical
181 named-output boundary.

## Local Evidence

The following focused and related suites pass:

```text
tests/unit/test_daily_operator_mechanistic_output.py                 10 passed
tests/unit/test_daily_operator_output_transform.py
tests/unit/test_daily_operator_successor_process_head.py            21 passed
tests/unit/test_daily_operator_strong_baseline_qualification.py
tests/unit/test_strong_baseline_mechanistic_output_review.py
tests/unit/test_hydrology_canopy_boundary_audit.py                   15 passed
tests/unit/test_doc_diffusion_boundary_baseline_panel.py
tests/unit/test_fast_owner_delta_representation_audit.py
tests/unit/test_fast_owner_exact_reconstruction_panel.py
tests/unit/test_thermal_coefficient_exact_reconstruction_panel.py    14 passed
                                                                    60 passed
```

The tests cover schema and parameter counts, eager/JIT forward, finite complete
loss, Attention-path VJP/JVP, sparse-`nroot` identity/deactivation/gradient,
canopy endpoint-to-transfer identity, complete post-update materialization,
canonical one-day state, water/carbon budgets, and runner compatibility.
Ruff, compileall, and `git diff --check` also pass.

This local evidence by itself did not claim that the complete
49.7-million-parameter gradient had been exercised against real v6 data. The
subsequent gate established that evidence.

## Subsequent Gate

The zero-update admission on free `gln01` passed eager/JIT complete forward
equivalence, finite complete loss, a finite nonzero full parameter-tree
gradient, the canonical retained-tail transition, finite declared next state,
and exact zero water and carbon budget residuals. See
`hydrology_canopy_output_v2_real_v6_admission_20260817.md`.

The next gate is the bounded one-epoch training smoke defined there. The
64-point-year, full-v6, 7/30/365-day rollout, paid execution, Gate D2, and
product stages remain closed.
