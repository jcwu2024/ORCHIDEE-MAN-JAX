# Normalized-state fan-out: local mechanism and numerical gates pass

The one approved free360s window completed first results and all three
interleaved pairs, exit0, with no repair or resume. Independent NumPy agrees
with every saved3854-state/summary/mask/JVP comparison. Actual optimized IR
retains the intended candidate eight-way FP64 cotangent accumulation followed
by one FP32 narrowing into normalization; original accumulation is FP32.
This is a completed local diagnostic, with no reference/production adoption,
publicT7 causal claim or complete-update speed improvement. The window is closed.

## Executed scope and resources

Exact command:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/normalized-state-fanout-component-20261003/run_normalized_state_fanout_component.sh --gpu 1
```

The executed route uses immutablec599236, canonical JAX0.4.38/container,
gln01`ibc13b02n04`, physicalGPU1 V100, OMP4/OpenBLAS1 and24GiB admission.
GPU1 was0MiB/0% with no contexts before launch and in the terminal snapshot.
All new arrays/IR/logs/sources are downloaded under
`outputs/diagnostics/normalized-state-fanout-component-20261003/`.

First dispatch UTC1791007216.9306278; completion1791007320.6425128. Whole
dispatch-to-exit103.711884975s,0.02880885694 free card-hours/CNY0, below the
first-dispatch360s deadline1791007576.9306278. The launcher ledger starts
2.675396442s later, after entry imports, and records101.036488567s/
0.02806569127card-hours. Both origins are retained; whole accounting includes
those imports. GPU termination happened before either deadline. Unused time
does not authorize another run.

| Stage | Actual seconds/evidence |
| --- | ---: |
| Actual-runtime preflight |79.801591931; passed|
| Component source/input/module setup |4.324594822|
| Original lower+compile |5.107423672; one attempt|
| Candidate lower+compile |4.716209665; one attempt|
| Original/candidate IR capture |0.244731141 /0.080084400|
| Component native backend calls |1 original +1 candidate|
| Setup helper backend calls |23, separately recorded|
| Forbidden backend calls during values/IR |0|

The preflight runs in its separate process. Observed JAX active peak for the
component process is8,564,736B (8.168MiB), peak pool10,485,760B (10MiB), passing
24GiB. These are cumulative component-process statistics, not per-arm physical
GPU peaks or full-model memory measurements.

## Numerical result and independent validation

The actual initial raw/mean/scale/defined/PFT/field/axis pools and threshold16
come from the accepted small input bridge. Artificial FP32 sin cotangents
remain explicit; they do not represent the public recurrent loss. Original
standardization and consumer-local statistics remain unchanged, including the
legacy FP32 branch for33/199/428.

| Gate | First | Pair0 | Pair1 | Pair2 |
| --- | --- | --- | --- | --- |
| Scalar and all eight summaries |Pass|Pass|Pass|Pass|
| All3854 raw-state VJPs |0 fail|0 fail|0 fail|0 fail|
| Masks/finite/21undefined exact zeros/24GiB |Pass|Pass|Pass|Pass|
| Native JVP/VJP33/199/428, both arms |Pass|Pass|Pass|Pass|
| Within-arm strict drift |—|Pass|Pass|Pass|

`independent-numpy-verification.json` independently traverses all eight saved
results, reproduces the four cross-arm comparisons and every within-arm
comparison, checks source-pool row masks/invalid exact-zero summary rows and
all21 undefined-state gradients, and reproduces native directional errors.
It imports no JAX and executes no model. The program and NumPy judgments agree
under unchanged `atol=1e-6,rtol=1e-5`.

Forward loss/summaries and three basis JVPs are exactly array-equal between
arms. State VJPs differ at1124 coordinates in every comparison, all within
the strict allowance; within-arm maximum difference is0. The largest absolute
cross-arm difference is154.687569618 atDOC coordinate1813: original
-2.035704811e9, candidate-2.035704966e9, allowed20357.048114441. Its scale is
4.816534788e-11 and tail selection is enabled. These large artificial-loss
derivatives are not scientific observables or grounds for a reference change.

| Direction | Original VJP−JVP absolute error | Candidate error | Allowance |
| --- | ---: | ---: | ---: |
|33|0.046875|0.0390625|0.671781156|
|199|2.980232239e-8|2.980232239e-8|4.985192776e-6|
|428|0.001953125|0.001953125|0.290207211|

Both original and candidate already pass all directions here.33 is modestly
closer;199 is unchanged;428 has equal error on opposite sides. This does not
reproduce or repair the historical publicT7 JVP failure. There is no test of
parameter gradients in this parameter-free summary component.

Warm medians are2.316302ms original and2.230827ms candidate. Millisecond
component timing has no speed threshold and does not identify this summary as
a major training bottleneck. Complete updates and their sampling/loss/gradients
were not timed.

## Actual optimized-IR scope

The original/candidate compact optimized modules are820,652/823,482B.
`actual-IR-mechanism-review.json` verifies the candidate connection: eight
F32[3854] partials widen toF64, seven F64[3854] adds end atline3277,
line3278 narrows the completed sum once toF32, and lines3279/3289 consume it
in the normalization transpose. Original corresponding seven adds areF32.
The source `_masked_summary` casts and its statistics/scalar reductions remain
F32. The state-VJP result isF64 because its raw input ABI isF64; output dtype
alone would not establish wide accumulation.

The review distinguishes state VJP from the three forward JVP outputs and
follows command-buffer output reconstruction. Compiler optimization removed
unique consumer labels: the eight partials match the eight declared consumers,
but the certificate does not assign each SSA partial to a named field/axis.
It proves this actual compiled component's precision flow, not a coordinate
cause, publicT7 precision flow, general derivative accuracy or scientific gain.

The canonical context's inherited whole-pickle identity and actual raw/pool/tail
bridges are reused. There is no independent historical per-array mean/scale
snapshot from the earlier publicJVP run, so it remains qualitative evidence.
No old2GB family, context or frozen bank proof was rehashed/recomputed.

## Next action toward training throughput

Do not expand this precision route from artificial-cotangent success: the
original local component also passes. The next bounded question is whether
the originalFP32 publicT7 loss sends materially cancelling cotangents into this
same eight-summary boundary. Real upstream cotangents are not recoverable from
the saved global initial-state gradient or the existing binary's returned ABI.

Prepare one originalFP32 native-AD extraction using the shared complete7-day
public objective, unchanged weights/data/masks, and an explicit day index so
summary cotangents cannot silently sum across days. A zero-delta diagnostic
seam may expose the eight state summaries only; discrete/static summaries stay
outside the seam. Bridge zero-delta public loss/auxiliary/physical values to
the existing original executable before accepting any cotangent. Reuse public
rollout/loss logic, not a reconstructed loss or artificial observation seeds.

The extraction requires a new native AD/JIT graph; this component's5s compile
does not budget it. CPU/source preparation must establish the day selector,
input/output ABI, one-compile launcher and a historically justified publicT7
compile cap. This remains a diagnostic of an original method, not approval for
a precision/reference change. OriginalFP32 instrumentation can use the standing
free-diagnostic authorization once source/CPU checks establish unchanged public
science; it gets a distinct finite budget and uses none of the closed360s
window's remainder. If real cotangents show
no material local admission issue, stop this precision workstream and return
to structural complete-update execution optimization; do not make exhaustive
internal-gradient decomposition a prerequisite for throughput work. Paid
resources or a new precision/reference/loss/architecture choice still require
explicit approval.
