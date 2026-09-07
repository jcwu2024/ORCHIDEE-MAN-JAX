# Conditioned Attention Streamed Adjoint

Status: full streamed diagnostic completed; narrow daily replay is next.
No training or model change.

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
with diagnostic serialization. Real-v6 numerical results are below.

## Completed Result

Run `0056141` completed on free gln01 GPU 0 in 1722.781 seconds. Baseline
hard failures are zero. Full streamed reverse projection is +2.6269377161;
forward projection is +2.6271017424. Canonical central differences are
+2.6385713650 at 1e-4 and +2.6308308773 at 1e-5, with no defined-mask changes.
The former production reverse projection -1097.9184 is not reproduced by
this path. This is derivative evidence, not improved model accuracy or proof
that remat itself caused the error.

The maximum daily duality gap is 0.000164441 on day 16. The compiled one-day
reverse estimates 1,609,245,328 bytes (1.50 GiB) for arguments, outputs and
temporary buffers; this is not measured total process VRAM. All daily state
gradients are finite. Observation loss 0.0534603538 and production loss
0.0534604043 are close but not bitwise equal. The largest JVP-primal versus
canonical state absolute error is 5.5879296 on day 2; its field and scale
must be identified before interpreting its significance.

Evidence: `outputs/diagnostics/conditioned-streamed-adjoint-0056141.json`
locally; corresponding `.json`, `.log` and `.trace.pkl` under remote
`runtime/outputs/diagnostics/`. The trace contains daily states, observations,
tangents, cotangents and adjoints, not model weights. Accepted update-3
recovery is unchanged. Access through alias cln01 with HostName
192.168.11.2 works when 192.168.11.1 times out.

## Narrow Replay

Use the same runner with `--replay-trace` (launcher environment `REPLAY_TRACE`)
to replay only days 2, 15 and 16. Day 2 has the largest raw state discrepancy;
day 16 has the largest duality residual; day 15 supplies an adjacent reference.
Reuse stored carries, direction and future adjoints; do not execute a new
30-day scan or gradient. Compare standalone forward, JVP primal and VJP primal
at identical inputs against each other and the stored canonical observations.
Report state coordinates by absolute, training-scale-normalized and downstream
cotangent-weighted differences, plus GPP differences and derivative reproduction.
Finite-intersection summaries are explicitly diagnostic and cannot replace
physical defined-mask or science acceptance. The weighted difference is a
linearized effect, not an observed full-rollout loss change.

This remains a free-node diagnostic on one GPU, bounded to 60 minutes, without
updates or checkpoint writes. Local tests include complete saved-trace runner
execution and assert that it never loads a new artifact or full-rollout evaluator.
Do not reconnect training until replay differences and actual descent using
the proposed backward path have been checked.
