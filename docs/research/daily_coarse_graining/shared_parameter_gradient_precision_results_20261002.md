# Fixed-local-S4096 kernel-gradient screen — 2026-10-02

Status: completed, rejected on strict state-gradient parity and the F4S speed
gate. All 16 parameter engineering groups passed and hard admission passed;
those results do not waive the failures. No training, optimizer updates,
recurrent validation, or production adoption has occurred. Independent
NumPy-only verification of the actual initial result and all three matched
pairs confirms these conclusions.

## Method and provenance

Only kernel tangents use `S * dot_F16_F16_F32(x,W_t/S)`, with fixed local
`S=4096`. Native JAX custom-JVP transpose produces
`dot_F16_F16_F32(x.T,S*g)/S`. Both low-dot operands are quantized; output and
accumulation are FP32. Forward values, input/state derivatives and biases keep
their FP32 operations; physics is FP64. Candidate parameter JVP is excluded
as a correctness oracle. This approximate derivative does not carry an exact
finite-precision adjoint claim.

The policy patches four `_linear` and three `_ffn` aliases in the shared
transformer/backbone/decoder execution path. Canonical tokenizer operations
remain outside this scope. Neural source provenance is
`strong_baseline_architecture.py::_linear:886–887` and `_ffn:897–898`;
no Fortran process equations are changed. The immutable execution checkout is
`c599236bcf57b745e9bafed678298010f4c3b58f`. The fixed policy SHA256 is
`00dc43791716456a1da047f6c9b9f6d47f0735e49d1f2173285e3282fefd3ab8`.

The original four samples, update-0 weights, loss, masks, architecture and
full 300-leaf/64,884,481-parameter gradient tree remain unchanged. Public F4S
also differentiates all 15,416 initial-state coordinates (four by 3854).
The saved original full-AD executable is replayed, never recompiled.

## Completed component evidence

The initial GPU shapes were illustrative component layouts. They are not
the actual F4S FFN batch layout. Five interleaved pairs gave:

| Illustrative component | Original/candidate median ratio |
|---|---:|
| `[384]→384` |0.774733|
| `[128,384]→1536` |0.986845|
| `[2,128,384]→1536` |1.072654|
| Masked FP64 scalar readout `[4,128,256]×[256,1]` |0.874752|

All four strict loss/input/bias comparisons and kernel engineering comparisons
passed. The component speed gate failed. Thirteen tiny interfaces passed all
primal/input/bias strict checks and eight kernel engineering checks; six tiny
kernel strict checks failed (579 coordinates). Native optimized V100 HLO
confirmed FP16 operands/FP32 output. The original process omitted arrays due
to an already-loaded logging source; same-seed untimed recovery preserved the
original timings and failed gate. Its peak was 203,691,264 bytes; the original
process peak was not recorded.

Source inspection plus existing real F4S forward StableHLO identified exactly
two relevant representative FFN classes:

| Actual F4S component | Original median (s) | Candidate median (s) | Ratio | Kernel relL2 | Cosine |
|---|---:|---:|---:|---:|---:|
| Decoder `[4,128,1,384]×[384,1536]` |0.000363506377|0.000331906602|1.095207|0.000193586106|0.999999981263|
| Backbone `[4,7,1,768]×[768,3072]` |0.000223821029|0.000225974247|0.990471|0.000285658267|0.999999959208|

All eight array comparisons were finite and strict; both kernel engineering
comparisons and native codegen passed. Peak memory was 215,226,112 bytes.
The aggregate component speed≥1.05 gate remained false. Independent NumPy
verification confirmed both sets of array comparisons and retained their
failed speed decisions. Shape evidence comes from decoder
`shared_decoder_macro_packed._decode_mapped_rows/decode_chunk:278` and backbone
`shared_backbone_projection_reuse.run_backbone_with_shared_source_projection:82`.
The older full F4S precision StableHLO is dimension/consumer evidence only.

## Public F4S execution

The lead continued the real F4S diagnostically under the user's standing
instruction to reach a real-model conclusion within the existing 3500-second
authorization. The component speed failure remains false. The strengthened
continuation guard preserves all interface correctness, alias restoration,
native full-JVP structure, exact policy/JAX/flags/shapes, codegen and memory
requirements. Independent source review confirmed this limited bypass.

The first backend attempt passed source-mask auditing, the 891-input ABI and
saved original reference validity, then rejected exactly
`UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_F16_F16_F32`.
The sole allowed repair added `--xla_disable_hlo_passes=operand_upcaster`.
A tiny repaired actual-shape retest passed numerical/codegen/memory checks and
retained the false component speed gate (ratios 1.108493 and 1.042488;
independent NumPy confirmation). An ordinary launcher control-flow
repair resumed from those completed assets without repeating the retest.
The second backend attempt lowered in 106.352290 seconds and compiled in
882.394850 seconds, then reached actual full results and all three warm pairs.
Its baseline flag also retains
`--xla_gpu_enable_llvm_module_compilation_parallelism=false`.

## Actual full F4S result

Loss and all 160 auxiliary coordinates passed strict comparison in the initial
result and every matched pair. Loss absolute difference was 1.41459e-8 and
relative difference 2.82391e-8. All 16 semantic parameter groups passed the
unchanged engineering limits in all four comparisons: maximum group relL2
0.000101496460 and minimum cosine 0.999999994852. The independently retained
strict all-parameter gate failed in 22,157 initial coordinates, with maximum
absolute difference 2.51830e-5 and total relL2 6.37885e-5.

The strict all-state gate failed and prevents acceptance:

| Comparison | Strict parameter failures / 64,884,481 | Strict state failures / 15,416 | State max absolute difference |
|---|---:|---:|---:|
| Initial candidate vs original replay |22,157|24|0.885466408|
| Matched warm pair 0 |22,157|21|0.885466408|
| Matched warm pair 1 |22,158|20|0.885466408|
| Matched warm pair 2 |22,158|20|0.885466408|

Independent NumPy recomputation found 16 source-defined state coordinates
failing in all three matched pairs (per-sample counts 5,4,5,2; zero overlap
with undefined coordinates). Zero-based packed indices by sample are
sample0:35,177,434,508,513; sample1:385,410,428,434;
sample2:32,34,180,505,1777; sample3:2320,2386. The saved identity and mask
metadata do not map these columns to named state fields. Their persistent
strict failure supports the stop decision without diagnosing its cause.

All values remained finite; all eleven physical hard counts were zero; source
mask invariance and all 82 actual undefined-state gradients exactly zero
passed throughout. Original executable estimated active memory was
11,348,361,292 bytes (10.568985 GiB); candidate estimate was 13,026,091,652
bytes (12.131493 GiB). Observed peak was 13,304,238,080 bytes (12.390537 GiB)
for the whole process containing both arms; it cannot be split into arm peaks
from this statistic. All satisfy the unchanged 24 GiB gate. Valid hard admission allowed the
diagnostic to complete all three pairs despite strict numerical failure.

| Matched pair | Original full derivative (s) | Candidate full derivative (s) |
|---|---:|---:|
|0|0.379496541|0.366359390|
|1|0.382573709|0.369449161|
|2|0.380763389|0.364840867|

The median original/candidate ratio was **1.039316582**, below the required
1.05. This is diagnostic daily full-derivative timing; it is not complete
optimizer-update throughput. The speed result independently fails even before
considering the state numerical rejection.

Original replay versus the saved original states had four strict failures
(max absolute 0.287239736). Original warm self-replays had 4,5,5 failures;
candidate self-replays had 6,4,5. These diagnostic differences establish replay
variation, without identifying the cause of the candidate-vs-original state
failures or waiving their strict gate. No tolerance was loosened and no new
GPU diagnosis was started.

## Runtime accounting and cleanup

The free route used one physical GPU1, Tesla V100-SXM2-32GB, on
`ibc13b02n04`/`gln01`, canonical JAX0.4.38, OMP4/OpenBLAS1. Each launcher
recorded node/load/affinity and empty GPU admission. There were two public
backend attempts and one exact known compatibility repair, with no paid jobs.

| Family stage | Wall seconds | Exit |
|---|---:|---:|
| Original preflight and illustrative component interface |113.681934|1|
| Untimed logging-array recovery |8.061156|0|
| Corrected source-backed actual-shape interface |7.231113|1|
| First public backend and repaired interface |306.935352|1|
| Second and final repaired public run |1072.437843|1|
| **Cumulative family** |**1508.347398**| |

Cumulative route time was 0.418985388 GPU card-hours, below 3500 seconds
(0.972222 card-hours), with CNY0. The repaired process reached a completed
numerical/performance failure; it did not exhaust a resource/runtime boundary.
Post-exit checks confirm PIDs and groups 63670,67427,68780,70840 absent,
no GPU compute contexts, and both GPUs at 0 MiB/0% utilization. The original
untimed recovery process peak is kept separate from its unsampled first peak.

Actual full candidate/original/pair arrays, all state replays and the repaired
candidate serialized executable are retained under `public-F4S-repaired/`.
The actual binary distinguishes this weight-gradient-only plus compatibility
combination from older forward-precision results. Accounting and source/asset
indices are `actual-accounting.json`, `source-manifest-final.json`,
`artifact-manifest.json`, and `final-cleanup-all-stages.json`.
The independent record is
`public-F4S-repaired/independent_numpy_verification.json`, reproduced by
`verify_public_F4S_numpy.py` using inert serialized-record placeholders and
blocking JAX imports. It verified all parameter/state coordinates, the actual
undefined-state mask and all three saved matched pairs without model execution.

## Acceptance and limits

The unchanged engineering gates cover all 16 parameter groups (norm≤1e-12:
maxabs≤1e-6; otherwise relL2≤0.01 and cosine≥0.9999). Strict loss/aux/state
comparisons use rtol1e-5/atol1e-6, with loss relative≤0.001; all-parameter
strict comparisons are retained independently. All finite, eleven hard
physical counts zero, exact source-mask invariance, actual undefined-state
gradients zero and estimated/observed memory≤24GiB are hard admission gates.
Undefined coordinates per actual sample are 20,20,21,21, established by a
CPU-only export of the cached declared source masks.

Numerical comparison failures with valid hard admission still require all
three bounded warm pairs. Hard admission failure stops immediately. Original
state self-drift is diagnostic and never waives matched strict state gates.
F4S speed≥1.05 is a separate requirement. F4S is a daily full derivative
measurement, not complete optimizer-update or recurrent throughput evidence.

All artifacts are in
`outputs/diagnostics/shared-parameter-gradient-precision-20261002/` and its
remote counterpart. Original, corrected, first-public and repaired-public
assets/source snapshots remain distinct. No T7/T30, training, paid job,
factor sweep, FP32 fallback, additional compiler flags or adoption is
authorized by this screen.
