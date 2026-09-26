# Real decoder FFN precision screen

Status: forward-only and native-forward/backward precision comparisons completed on free V100s,
September26. This is not a production precision policy or a training result.
The user requires speed optimization before training. No paid allocation.

## Result and decision

Using native `DotAlgorithmPreset.F16_F16_F32` with FP32 parameter/activation
storage also lowers native-transpose matrix operands to FP16, while accumulating
and returning FP32. No custom VJP/JVP, straight-through rule or handwritten
cotangent is introduced. On the SAME saved real FFN inputs this gives:

| Layout | FP32 forward + VJP | Native-policy forward + VJP | Speed ratio |
| --- | ---: | ---: | ---: |
| micro4 | 72.415ms | 46.250ms | 1.566x |
| micro1 | 114.759ms | 95.827ms | 1.198x |

All values/gradients are finite. Native-policy relative L2 errors are0.04193%
forward,0.27757% FFN-parameter VJP,0.04524% input VJP; exact1e-5/1e-6 comparison
still fails. This is a promising arithmetic component, not a production
numerical-method admission or a full-update improvement. The10x goal is unmet.
Keep this as the precision candidate for the next bounded broader-neural
execution assessment; do not launch training or a long full recurrent compile
based on a fragment alone. Its measured absolute time cannot simply be
subtracted from the separately compiled full-model timing.

The native policy screen70060 completes in4.93s on deployed JAX0.4.38/V100.
All five tested policies compile; the selected FP32-storage preset yields six
FP16-operand/FP32-result dots including native backward. The independent tiny
device trace contains `cutlass_70_wmma_tensorop_*` and
`volta_s884gemm_fp16_64x64_ldg8_nt` kernels. This is actual Tensor-Core kernel
evidence for that capability run; it does not attribute every full-fragment
or production kernel. CUPTI activity-buffer/missing TensorFlow-hook warnings
are recorded;14 device events remain available. Trace timings are not the
unprofiled speed measurements above.

Real-policy micro4/micro1 processes70456/70703 complete in22.46/25.88s.
Reference/candidate gradient compilation is3.06/2.97s on micro4 and4.66/4.92s
on micro1. Compiler estimated live memory remains about2.799GiB/2.820GiB
respectively. Five warm repetitions per arm; FP32 precedes candidate in each
process. Local reports are `outputs/shared-ffn-algorithm-capability-20260926.json`,
`outputs/shared-ffn-algorithm-micro{1,4}-20260926.json` and the capability's
`.trace.json.gz`/`.trace-summary.json`. Remote artifacts use the same stems.
The isolated `probe_shared_ffn_algorithm.py` supports a tiny `capability` mode
and a capability-gated `fragment` mode reusing the original producer/scheduling.
Ten targeted tests pass. No task from these comparisons remains running.

## Earlier explicit-cast result and why it was insufficient

FP16 operands with requested FP32 dot results accelerate the micro4 forward,
but the forward-plus-native-VJP improvement is small. Do not integrate this
version into a full daily/T7/T30 compilation on the strength of these results.
An identified next issue is that native JAX AD still lowers every backward
matrix contraction in this experiment to FP32 operands and FP32 results.
The subsequent native-policy result above resolves this capability question;
no custom adjoint is introduced.

Both layouts process the same four saved daily samples and576 FFN calls.
Micro4 restores the sample vectorization used by the daily objective; micro1
executes those samples serially and probes the per-call shape used in a
single recurrent trajectory. It is not an actual multi-day rollout.

| Layout | FP32 forward | Mixed forward | FP32 forward + VJP | Mixed forward + VJP | Combined speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| micro4 | 21.430ms | 11.550ms | 72.240ms | 62.239ms | 1.161x |
| micro1 | 34.807ms | 33.343ms | 114.568ms | 112.137ms | 1.022x |

Warm medians are five synchronized repetitions, with FP32 measured before
mixed precision on each GPU. The two layout tests ran concurrently on separate
empty GPUs; host load is recorded in their logs. These are component results,
not a full-update speedup or an exact decomposition of forward/backward costs.
Independent component execution returns all saved activations and their input
gradients; production fusion/liveness differs, so its absolute time cannot be
subtracted from the full-model runtime to predict a new update time.

## Numerical evidence

All outputs,2,363,136 FFN parameter gradients and26,775,552 captured-input
gradient coordinates are finite. Relative L2 differences for micro4 are:

- Forward outputs:0.04193%.
- FFN parameter VJP:0.25259%, cosine0.99999681.
- Captured-input VJP:0.04676%, cosine0.99999989.

Micro1 is similar:0.04193%,0.25304%,0.04706%, respectively. The unchanged
exact-execution rtol1e-5/atol1e-6 comparison fails, as expected for a genuinely
different precision method; tolerances were not loosened. This is not a
scientific rejection of all mixed precision. VJP seeds are deterministic
synthetic sinusoids, including inactive coordinates, not the science loss.
No physical budgets, recurrent gradients, optimizer updates or learning
outcomes were evaluated.

The actual deployed JAX0.4.38 StableHLO contains24 forward dot occurrences
with FP16 operands/FP32 results and48 backward dot occurrences with FP32
operands/results (both layouts). Optimized HLO also shows FP32 backward
cuBLAS calls. This supports a concrete limitation of this implementation;
it does not quantify exclusive backward wall time. Tensor-Core instructions
have not been established by a device trace and are not claimed.

## Runtime, memory and reproducibility

Capture63001 completed0 in319.87s after the81.10s tiny deployed runtime check.
It used fresh update0 from the trusted technical-run report and artifact742,
not the profile's later fixed-update2 replay weights. Input assembly and
the unchanged neural backbone/coordinate chunks were evaluated without
physical processes, loss, optimizer or recurrent AD. One112MiB pickle retains
real inputs, both FFN parameter blocks, active masks and call/sample/expert
metadata. It is reusable and is not a new Teacher release.

Fragment68352 (GPU0,micro4) completed0 in22.11s;68500 (GPU1,micro1) completed0
in28.32s. FP32/mixed forward-plus-VJP compilation was3.16/2.46s on micro4
and4.60/3.59s on micro1; forward-only compilation was0.50–0.64s. Compiler
estimated live memory changes2.799→2.654GiB on micro4 and2.821→2.712GiB
on micro1. These are compiler liveness estimates, not GPU pool reservations.

First fragment67557 failed before backend compilation because of a first-time
module import inside JAX tracing. Explicit eager module initialization repaired
the diagnostic entry. The completed payload/preflight were reused. Review also
corrected serial-sample packing to preserve production micro4 vectorization
before either successful comparison. Nine local tests include a fresh-process
CLI and vectorized/serial FP32 forward/native-VJP agreement.

Code: `scripts/hpc/probe_shared_ffn_precision.py`,
`scripts/hpc/run_shared_ffn_precision.sh`,
`tests/unit/test_shared_ffn_precision_probe.py`.
Remote overlay: `runtime/transfer/shared-ffn-precision-20260926` against
immutable model checkout `runtime/checkouts/shared-split-width-v1` (c599236).
No production model or sampling configuration was edited.

Local reports: `outputs/shared-ffn-capture-20260926.json` and
`outputs/shared-ffn-fragment-micro{1,4}-20260926.json`.
Remote reports/logs use the same stems under `runtime/outputs/diagnostics`;
the payload is `shared-ffn-payload-20260926.pkl`. Each successful fragment
also preserves its FP32 and candidate full results before the next stage.
The import-failure report/log are retained locally with `.import-failure`
suffixes. No task from this first comparison remains running.
