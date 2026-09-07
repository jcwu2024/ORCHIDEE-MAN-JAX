# Conditioned Attention Streamed Adjoint

Status: frozen free-node derivative diagnostic. No training or model change.

## Scope And Evidence

The same-production-scalar audit reproduced a reverse/JVP sign disagreement
on artifact 442/window 0 at accepted update 3. Its whole-window no-remat
reverse failed after compilation because it requires 43.12 GiB of temporary
device memory. Do not repeat that configuration. Reuse its saved diagnostic
direction and four-window sufficient statistics from
`conditioned-derivative-modes-2cb51af.probe.pkl`.

## Full Gradient Without A Whole-Window Reverse Graph

Run the canonical forward scan once and retain its daily continuous/discrete
carries. Define each day's differentiable observations as next continuous
state, daily fast GPP, and current PFT fraction. The PFT fraction is essential:
it depends on the day-start state and weights the science outputs. Reuse the
existing trajectory extraction and linearized science loss to differentiate
the complete observation sequence, holding all four-window bias totals fixed.

Propagate the saved parameter-direction tangent through each daily transition
at the stored canonical carries. Then visit days 30 through 1. At each day,
add the next day's state adjoint to that day's loss cotangent on next-state,
apply the ordinary one-day VJP, accumulate the parameter gradient in float64,
and pass the resulting day-start-state adjoint to the preceding day. Do not
drop the future-state term: this is full BPTT, not truncated training or
independent Teacher-state differentiation. Discrete carries must be bool/int;
otherwise the diagnostic stops rather than silently dropping their derivative.

Each one-day JVP/VJP uses the same function. Record their dot-product duality
residual, parameter projection, state-gradient norm, nonfinite count and top
source-mapped state-adjoint coordinates. Compare the summed directional
derivatives with canonical central differences at 1e-4 and 1e-5, including
actual dtype-rounded parameter displacement. Record local-primal versus stored
scan discrepancies with undefined states masked. Zero-cotangent undefined
constants do not pollute diagnostic dot products, but active NaNs remain visible.

This changes graph boundaries and floating-point evaluation, not the scientific
model. An improved agreement alone does not prove rematerialization is the
cause. Interpret replay error, daily duality residuals and finite differences
together before proposing a repair. No derivative result promotes the model.

## Resource And Verification Boundary

Runner/wrapper: `scripts/hpc/audit_conditioned_attention_streamed_adjoint.py`
and `.sh`. One free gln01 GPU, 90-minute limit. No paid job, training update,
new Teacher data, monitor, validation/test exposure, or parameter/checkpoint
write. Reuse the canonical environment and prepared v6 artifact.

Before executing the compiled one-day reverse, inspect its memory analysis.
Arguments + outputs + temporary buffers - aliases must fit a 24 GiB diagnostic
envelope, leaving headroom on the 32 GiB device for live model/carry buffers.
If this does not fit, preserve the report and stop before attempting execution.
No 30-day no-remat reverse is compiled.

Persist incremental JSON after initialization and every completed day. Save
daily carry/observation/tangent/adjoint arrays in a diagnostic `.trace.pkl`
without model weights, so further process localization need not repeat the
whole diagnostic. Do not overwrite either accepted recovery or previous runs.

Local qualification: 28 tests pass. The new coverage verifies full 30-day
parameter-gradient equality with monolithic reverse for float64 and mixed
parameter precision, forward/reverse/finite-difference agreement, a nonzero
first-day contribution, memory-limit rejection, undefined-coordinate dot
products, actual launcher argument coverage, and complete runner orchestration
with diagnostic serialization. Real-v6 numerical results remain pending.
