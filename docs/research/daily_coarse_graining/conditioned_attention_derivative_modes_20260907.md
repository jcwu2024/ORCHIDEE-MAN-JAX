# Conditioned Attention Same-Graph Derivative Modes

Status: remat comparison complete; whole-window no-remat failed on memory.
Superseded for current action by the streamed-adjoint diagnostic.

## Question

Window 442/0 at accepted update 3 has production reverse directional derivative
-1097.9184, diagnostic forward tangent +2.6295 and central finite difference
+2.6274 at epsilon 1e-5. Previous forward and reverse graphs used the same
science formula but different wrappers and rematerialization settings. Do not
yet attribute this to rematerialization, a JAX defect, or a physical process.

## Fixed Comparison

Use the existing production `window_linear` scalar, exposed as
`steps["window_linearized_score"]`. Its default remains `rematerialize=True`,
so no existing training caller changes behavior. The diagnostic computes
`value_and_grad`, `jvp`, and ordinary forward evaluation from the identical
scalar function, with fixed parameters, microbatch and four-window sufficient
statistics passed identically. All totals stay frozen as in production.

Run only two modes: canonical rematerialization enabled, then disabled. In
the first mode derive the negative unit reverse gradient and record its norm
and scalar against historical values. Use this one direction in both modes.
Evaluate central differences at epsilons 1e-4 and 1e-5 with dtype-preserving
parameter perturbations. Record reverse dot products with both ideal and
actual rounded directions, primal differences, signs, relative disagreement,
compile memory estimates and stage timings. A norm/replay change itself is
evidence to investigate, not permission to claim a repair.

If disabling rematerialization restores reverse/JVP/finite-difference
agreement, isolate the recomputed boundary before changing production. If
both modes disagree, continue to a same-linearization transpose/precision
test using the saved direction. If both agree but differ from historical
production, inspect wrapper/compilation sensitivity. No outcome alone opens
training, scientific promotion or Gate D2.

## Execution Boundary

Runner and wrapper: `scripts/hpc/audit_conditioned_attention_derivative_modes.py`
and `.sh`. One free gln01 V100, 90-minute timeout, no paid queue. Reuse accepted
v6 artifacts and update-3 recovery. No regeneration, monitor, fresh validation,
optimizer, retained update, model/checkpoint write, or 365-day run.

Write incremental JSON only after the inherited initializer has completed.
Save the derived direction and frozen total statistics in a diagnostic
`.probe.pkl` beside the JSON, together with their parent identity. This file
contains no model weights and is not a recovery/checkpoint. It avoids repeated
expensive reconstruction in a follow-up diagnosis.

Twenty-two local tests pass, including complete runner orchestration and
probe serialization, compiled 30-step toy recurrence
JVP/VJP/finite differences with remat on/off, closure mode binding, actual
launcher parser coverage, bootstrap output lifecycle, and undefined-mask
handling. The real-window numerical comparison remains pending.

## Active Run

Launched at code `2cb51af` on free gln01 GPU 0, actual host `ibc13b02n04`,
wrapper PID 12150, 90-minute timeout. It failed after 5154.615 seconds when the
compiled no-remat reverse requested 46,297,116,312 bytes (43.12 GiB) of temporary
GPU memory, beyond the 32 GiB device. The complete remat scalar still gives
reverse -1097.9184265 versus JVP +2.6294964 and finite difference +2.6274486.
The no-remat case produced no derivative. It is not evidence that remat caused
the sign error. No model parameters changed; the probe direction was saved.
Check `runtime/outputs/diagnostics/conditioned-derivative-modes-2cb51af.log`
and `.json` on the server before repeating any computation. The local-only
handoff documentation commit does not change the running server checkout.

Do not rerun the whole-window no-remat configuration. The next owner is
`conditioned_attention_streamed_adjoint_20260907.md`. It consumes the saved
probe without rebuilding the failed window gradient, preserves the complete
30-day derivative chain, and compiles only a one-day reverse graph.
