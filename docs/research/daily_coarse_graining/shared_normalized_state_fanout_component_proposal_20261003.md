# Normalized-state fan-out component: approved window completed and closed

The bank-adjoint T7 route is rejected and closed. One new local precision
diagnostic is prepared to test whether native reverse AD loses accuracy when
the same normalized state feeds eight summary consumers. This supplies local
mechanism evidence for later throughput admission work. It has no independent
speed-benefit objective, makes no T7 causal claim and restarts no closed route.
The user approved this scientific choice and one free360s GPU1 window on
2026-10-03. It completed first+three interleaved pairs in103.711885s including
entry imports,0.028808857 free card-hours/CNY0. Local numerical checks and
actual-IR mechanism review pass, without reference or production adoption.
The window is closed; the [result](shared_normalized_state_fanout_component_results_20261003.md)
records the evidence and exact limitations.

## Concrete boundary and scientific choice

Immutable source is `c599236bcf57b745e9bafed678298010f4c3b58f`.
`strong_baseline_architecture.py::assemble_strong_baseline_context`1111–1130
normalizes the full3854 state once and sends it to the257-row field summary;
1166–1175 sends that same variable to seven140-row expert-axis summaries.
`_masked_summary`957–973 already converts each consumer input to FP32 and
performs FP32 mean/variance/min/max reductions, with the canonical finite-zero
sqrt tangent. Only the state summaries are isolated: discrete/static summaries,
linear projections and the later recurrent neural graph are outside this
component. `input_conditioning.py::standardize_continuous`12–52 controls
the original standardization and selected-tail arithmetic.

Original component: normalize once to FP32 and share with all eight original
summary consumers. Candidate: widen that exact normalized FP32 vector once to
FP64 at the named shared boundary; the existing consumer casts remain FP32.
Native AD may then accumulate cross-consumer cotangents in FP64 before narrowing
once at normalization. Internal summary reductions, masks and scalar-loss
reductions remain FP32. No custom VJP, stop-gradient, public-loss change, parameter
capacity change or production-module edit is included. Source arithmetic alone
does not prove that the GPU compiler retains this reverse boundary.

The four relevant original functions are AST-equal to the immutable Git blobs.
The prospective one-assignment overlay parses and compiles as syntax only.
The existing saved-HLO audit in `conditioning-normalization-path.json` confirms
representative F64 normalization/F32 projection and F32 summary reduction paths,
but its dynamic gather index cannot identify33/199/428 or the actual fusion
boundary. Reuse that audit; do not treat it as proof of the new candidate.

## Real prepared inputs and limited historical bridge

One consolidated NumPy `component-inputs.pkl` contains the actual initial raw
FP64[3854], mean/scale FP64[3854], tail mask, source-defined/PFT mask, field pool
[257,3854], seven-axis pool[7,140,3854], threshold16 and original case/method
identity. The4,915,365-byte small bundle is bound by exact byte count/SHA256 to
`component-input-bridge.json`; the actual loader verifies those bytes before
unpickling. CPU-only passive extraction on test01 reads the canonical prepared
context; no model or JAX imports occur. Raw-defined and post-PFT undefined
counts are both21. Pool/tail arrays, raw-defined mask and first-day active-PFT
mask compare exactly with the saved500s inputs; raw3854 matches the previously
recorded original-input fingerprint. The500s, original-FP32 JVP and existing
full-neural-FP64 identities are equal.

The runners declare the same frozen canonical context path. Its prior whole
pickle fingerprint `ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026`
is inherited, without rehashing the context. New small mean/scale fingerprints
are recorded in `component-input-bridge.json`. There is no independent per-array
mean/scale snapshot from the historical JVP execution; that historical JVP is
qualitative supporting evidence. Both new arms and their directional checks use
the same current small bundle. This cannot establish the cause of the original
T7 differences.

| Coordinate | Raw / mean / scale | Tail selected | Actual consumer membership |
| --- | --- | --- | --- |
|33|0 /1.0555922e-8 /1.5512721e-6|False|Field22; axes0/1/2/6 rows0,20|
|199|0 /0 /1|False|Field45; axes0/1/2/3/5/6 rows0,1|
|428|0 /4.6377426e-9 /4.3430424e-6|False|Field61; axes0/1/2/3/5/6 rows0,7,17|

These coordinates use the legacy FP32 standardization branch. The small33/428
scales can amplify cotangent roundoff during division;199 has scale1. Membership
and scale establish a plausible local mechanism, not attribution of T7 error.
Shared-transformer local gather normalization and later projections remain
other FP32 paths outside this component.

The scalar is explicitly artificial: eight arrays of
`sin(arange+1+consumer_index*0.125)`, computed FP64 and stored FP32, contract
with the eight summary outputs. These are not public-loss cotangents and are
not inferred from the saved global state gradients. The component returns a
scalar loss, eight FP32 summary arrays, eight row-valid Boolean masks, the
native raw-state VJP FP64[3854] and native basis JVPs33/199/428. It has no
trainable parameters; the full300-leaf parameter/science gates belong to a
separately discussed public-model route.

## Proposed bounded execution and decision rules

The reviewable command is:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/normalized-state-fanout-component-20261003/run_normalized_state_fanout_component.sh --gpu 1
```

Only one free gln01 V100 GPU1 is proposed, canonical JAX0.4.38/container and
immutable `runtime/checkouts/shared-split-width-v1`. Output is
`runtime/outputs/diagnostics/normalized-state-fanout-component-20261003`.
Whole allowance360s includes350s main and10s termination grace, at most0.1
free card-hours/CNY0. Setup, imports, preflight, compilation, artifacts, calls
and ordinary repairs share the same target-clock deadline.

The actual-runtime preflight runs once with a90s cap, informed by the prior
80.272568s preflight. The single90s component setup encloses source/preflight
validation, JAX imports/GPU initialization, input loading/device transfer and
component construction; each original/candidate
lower+backend attempt60s; each optimized-IR capture15s; each call15s. All
individual maxima are clipped to remaining whole time and cannot be added to
extend it. Component compile time is unmeasured. This deliberately bounded
window may end before all pairs; completion is not promised. It omits the two
approximately96s full-model binary loads, without claiming a steady-state gain.

There is at most one compile attempt per arm. First original and first candidate
are saved, followed by three alternating synchronized pairs if admitted and
time remains. Hard/shape/nonfinite/mask/21undefined-exact-zero/24GiB failures
save raw results and stop immediately; an original hard failure precedes any
candidate compilation. Strict comparisons use unchanged `atol=1e-6,rtol=1e-5`
for loss, summaries and all3854 raw-state gradients; masks are exact. Native
JVP/VJP consistency and within-arm drift are recorded separately. Timing is
reported without a speed gate or throughput claim.

Preserve optimized IR of both actual compiled executables, full compact text
with metadata and operands, within the same budget. No new compilation is
allowed for printing or calls. A numerical pass stays **IR review pending**:
subsequent bounded CPU review must establish the candidate cross-consumer F64
cotangent accumulation and narrowing connection. A compiler-folded or
unresolved boundary is inconclusive mechanism evidence, even if values pass.
Structural evidence, forward comparison, finite/mask/undefined-zero gates,
strict cross-arm reverse gradients, native JVP consistency and timing remain
separate result categories.

If the candidate is closer to native JVP yet fails strict original-FP32 parity,
it supplies possible mechanism evidence only. No reference is promoted, no gate
waived, and no speed candidate reopened. The next scientific question would
concern the reference/acceptance basis. Even a fully passing local component
does not verify the public T7 loss, full-network gradients or optimizer update.
Any fullT7 integration or new reference choice requires a separate decision.

An ordinary precompile interface repair may use `--resume` only within the
original deadline and with unchanged candidate policy. A successful preflight
from that same window and unchanged boundary is preserved and reused rather
than rerun; its reuse is recorded. Any compile attempt or
terminal scientific/hard result blocks resume. There is no compilation retry,
GPU sweep, paid fallback or new window implied by unused allowance.

## Prepared artifacts and checks

Policy: `scripts/hpc/normalized_state_fanout_component.py`.
Probe and deadline launcher:
`probe_normalized_state_fanout_component.py`,
`run_normalized_state_fanout_component.py`,
`run_normalized_state_fanout_component.sh`.
Preparation evidence lives under the completed500s family's
`next-component-preparation/`: consolidated bundle/metadata, small-dependency
bridge, saved-HLO source audit, source-AST/CLI records and actual test01 Bash
full/resume dry-run logs. These are preparation checks, with zero component
evaluations, GPU calls or scientific acceptance. Scientific choice and the
one360s GPU window have been approved; the scope and stop rules above remain.

Latest deployed test01 checks pass in `component-interface-preparation.json`:
actual Bash full/resume expansion, CLI, source-AST/input-bridge plan and policy
interface guards. `affected-interface-verification-test01.json` verifies the
90s setup enclosure, successful unchanged-preflight reuse, failed/changed
preflight rejection and actual deployed small-bundle byte/hash binding. These
records contain zero JAX imports, GPU/component calls or native compilation.
