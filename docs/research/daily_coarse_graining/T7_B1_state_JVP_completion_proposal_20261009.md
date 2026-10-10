# T7 B1 State-JVP Completion Proposal

## Purpose

Measure the missing native B1 initial-state JVPs under the same FP32 neural,
FP64 physics, public checkpointed T7 objective, guarded turnover, four TRAIN
windows (742/275/482/299), and strict `rtol=1e-5`, `atol=1e-6` gates used by
the completed B4 diagnostic. This is a new, independent free-node request.
The prior 3600-second grant is closed: 2809.490106 seconds were charged and
its remaining 790.509894 seconds are unavailable to this proposal.

## Preserved Evidence

The original B1 backend attempt and its failed result stay immutable. The
successful B4 JVP executable from the B4-only recovery is reused from
`T7-state-jvp-20261009-B4-recovery.result.b4.jvp.executable.pkl`; it is not
recompiled. Before use, the isolated runner must verify its binary and sidecar
identity, B4 route, JAX version, compiler options, four FP64 unit directions,
argument tree/shape/dtype ABI, and the matching B4 mathematical identity.
The saved B1 and B4 full-VJP executables remain the endpoint references.

The corrected offline analysis is
`outputs/diagnostics/shared-latent-reader-20261004/T7-state-jvp-20261009-B4-recovery.independent-analysis.json`.
Both routes' parameter-gradient self repeats pass all 66,123,713 numeric
coordinates across 355 leaves. B1's begin/end full-state comparison fails one
of 3,854 values, coordinate 2621. B4's begin/end comparison fails five of
15,416 values: `(sample, coordinate)` `(3,2621)`, `(1,509)`, `(3,434)`,
`(3,33)`, and `(2,514)`. The B1/4 selected projections passing does not waive
the full unscaled-state failure. The B4 JVP repeats are exact, but its
same-route JVP/VJP checks fail at 434, 2382, and 2621; control 200 passes.
These are numerical discrepancies, not a cause attribution or an accepted
derivative reference.

## Fixed Measurement

Compile only one new B1 JVP graph. Load the B1 VJP, B4 VJP, and saved B4 JVP
executables, each with an independent 300-second cap. Run four alternating
cycles of B1 and B4 JVPs over the four existing sample-3 directions
434/2382/2621/control200, for 32 calls total. Save four full VJP endpoint
calls (B1 and B4 before and after the JVP cycles), all primal/auxiliary
bridges, tangents, complete state gradients, timing, memory, and strict
comparisons. Keep the B1 objective divided by four and the B4 objective as a
four-case mean; compare B1 JVP values directly to B4 JVP values. Preserve all
failures. No optimizer updates, precision/tolerance changes, data changes,
training, adoption, or automatic reference selection.

## Proposed Resource Envelope

One V100 on `gln01`, at most 3600 newly funded GPU-seconds, CNY 0, with no
paid fallback. Proposed caps are 180 seconds for B1 lowering, 1800 seconds
for its single backend attempt, 2400 seconds for the route, 300 seconds for
each of the three saved executable loads, and 3300 seconds for the probe with
a protected 300-second exit reserve. This allows one B1 JVP compile, zero B4
JVP compiles, 32 JVP calls, four saved VJP calls, and zero updates. There is
no second B1 compile attempt if the proposed attempt fails or times out.

## Approval Boundary

`T7-B1-state-jvp-completion-20261009.approval.json` is a false template with
`approved=false`. CPU preparation, target-shell dry-run, and probe plan-only
checks do not approve or reserve this scope. GPU execution must reject this
template. A separate explicit approval must bind this exact resource envelope,
working directory, entry point, output path, and zero-cost scope before any
GPU occupation begins.

## Planned Isolated Artifacts

- `runtime/transfer/T7-B1-state-jvp-completion-20261009/probe_shared_latent_T7_state_jvp.py`
- `runtime/transfer/T7-B1-state-jvp-completion-20261009/run_shared_latent_T7_state_jvp.py`
- `runtime/transfer/T7-B1-state-jvp-completion-20261009/run_shared_latent_T7_state_jvp.sh`
- `outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.approval.json`
- `outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.result.json`

The completion branch must preserve the original failed B1 compile-attempt
record and all B4 recovery artifacts. Its plan-only report must show
`B1_JVP: 1`, `B4_JVP: 0`, three saved executable loads, 32 JVP calls, four
saved VJP calls, zero updates, and the caps above. This proposal makes no
claim that those checks had already passed or that execution was authorized
when this proposal was prepared. Subsequent execution is recorded below.

## Execution status

The user's subsequent explicit “批准。” approved this independent free
single-V1003600-GPU-second scope. The false template remains preserved;
`T7-B1-state-jvp-completion-20261009.approved.json` is the separate actual
approval. Seventeen focused tests passed. Deployed CPU metadata and full-child
plan-only checks passed, as did actual budget-bootstrap admission/rejection
fixtures and target gln01 Shell expansion. Ordinary startup fixes corrected
nested input receipts, the existing compiler-options nesting, old-Bash empty
array expansion and reuse of the already passed deployed runtime preflight.
No scientific precision, objective, tolerance or sampling changed.

Execution completed on free GPU 1/actual host ibc13b02n04, starting at host
22:26:56+0800 with 0 MiB/0%/no processes and host load 42.35/43.30/43.95.
The single B1 compile succeeded: lowering 143.991934s, backend 992.923233s.
B4 JVP and both VJP graphs loaded without recompilation. All 32 JVP calls and
four full VJP endpoints completed. Both routes' JVP cycles repeat exactly;
B1/4 versus B4 strictly fails at 434/2382, while 2621/control 200 pass.
Own JVP/VJP and full-state self/cross failures remain. Primal, all 355 numeric
parameter leaves/66,123,713 coordinates, auxiliary and physical repeats pass.

The independent ledger is finished within cap, charging 1780.7436698693782
GPU seconds/0.4946510194081606 card-hours/CNY 0, child exit 3; GPU is released.
The parent-observed final SSH exit 1 is distinct and its cause is not inferred.
Unused 1819.2563301306218 seconds grant no new graph. Independent NumPy-only
full-array acceptance completed on test01 CPU in 4.87128541059792s, exit 0,
with six small guards passed. [Completion results](T7_B1_state_JVP_completion_results_20261009.md)
owns the exact comparisons, artifacts, receipts, timings and limits. Zero
optimizer updates ran; no derivative reference or adoption is selected.

The accepted source archive
`T7-B1-state-jvp-completion-20261009.source.accepted.tar.gz` has SHA256
`66c99a3928509c70d78037ee348a91edb56ebc31cfd23d1d7a6e4922e07caff0`.
The pre-Shell-fix archive and first CPU identity-check error are retained.
The next first-day same-precision localization CPU preparation passed;
the [first-day run](T7_first_day_JVP_localization_results_20261010.md) completed
under its independent scope on 2026-10-10, with numerical failure and GPU released.
[NEXT_STEPS](../../NEXT_STEPS.md) owns current action and authority.
