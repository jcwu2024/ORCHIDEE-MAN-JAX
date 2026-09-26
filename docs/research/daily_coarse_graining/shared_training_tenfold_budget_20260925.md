# A tenfold shared-training throughput budget

September26 update: combined source-K/V reuse and expert128 now pass actual
daily andT7 all-parameter/physical admission. Matched daily warm throughput
is1.339x; T7 is1.143x(12.53%less time). T30 admission is running. Earlier
"untested/next" wording below describes the prior hypothesis. These results
confirm only modest execution gains; the10x objective remains unachieved.
Next use savedT7 binaries to profile residual work without new compilation,
while finishingT30/full-update admission. Hardware/precision/architecture
routes must retain their separately stated resource or method costs.

The updated decision is not to promise a sequence of small packing changes
will reach10x. At the current one-V100 FP32 capacity, the optimistic1.21s
dense-only arithmetic estimate below already consumes most of the1.56s
target. This is a feasibility warning, not a proof of an exact lower bound.
T30 remains central: daily/T7 gains alone leave the previous7.688s T30
component, plus optimizer/transport/other work. More GPUs overlapping entire
daily/T7/T30 branches cannot remove that sequential trajectory cost.

The combined daily trace has about35.7% summed matrix time and58.2% fused
kernel time. Even an ideal8x speedup of just those matrix kernels gives only
about1.46x on this category accounting; infinite matrix speed gives~1.56x.
These are illustrative Amdahl calculations, not exact wall-time bounds:
summed device work can overlap and fused ownership remains ambiguous. They
do rule out presenting a nominal Tensor Core8x figure as complete-training10x.

The next exact-schedule experiment must be justified by the savedT7 profile,
not by chunk counts alone. If it identifies material small-matrix/launch
fragmentation, test one larger expert tile against the retained full result.
If not, do not continue a blind tile sweep. A possible later precision route
would use V100 FP16 Tensor Cores only at named neural GEMMs, while keeping
FP32 master weights/Adam/gradient accumulation/norm/softmax and current
physical/conservative/retained/trajectory precision. It needs an explicit
numerical-method decision and a separate declared finite/gradient/learning
and7/30-day drift contract. It cannot quietly relax the current1e-5/1e-6
same-execution gate. No precision change has been implemented or approved.
Width/depth/low-rank/output-head reductions likewise require an architecture
discussion; auxiliary labels still do not justify removing active capacity.

Latest measured evidence: expert128 actual daily micro4 passes all64.884M
parameter-gradient coordinates and physical checks, with1.202x admission /
1.213x saved-binary replay throughput. Device events fall62,617->30,801 while
matrix/fused work remains substantial; see shared_saved_daily_profile_20260925.md.
Common-source backbone reuse is the next bounded real-loss diagnostic, not
an accepted optimization. The10x full-work target below remains unachieved.

Status: decision budget, 2026-09-25. This is not a measured speedup or a
proposal to change the model, loss, data, optimizer schedule, precision, or
parameter/state-gradient boundary. The comparison axis is about 10x *warm
complete-update* throughput for the same daily32 + one T7 + one T30 work.
One-V100/FP32 is the present diagnostic baseline, **not** a user-imposed
permanent hardware or precision constraint. Startup is a separate gate. No
paid job is authorized here.

## Baseline and target

The split model has 64,884,481 trainable parameters, a width-768/eight-block
backbone, width-384/two-block decoder with FFN1536, 161 output fields and
8,550 statically eligible coordinates. Its profiled neural GEMMs are
explicitly FP32 on a 32-GiB V100; the profile does not establish the dtype
of every retained physical operation. The trusted c599236 profile and
direct-loaded two-update replay give:

| Fixed work, one V100 | Warm observed | 10x time target |
| --- | ---: | ---: |
| Complete daily32 + T7 + T30 update | 15.63-16.11 s | 1.56-1.61 s |
| Daily32 (eight micro4 gradients) | 4.269 s fixed-weight replay | 0.427 s |
| One T7 gradient | 1.813 s fixed-weight replay | 0.181 s |
| One T30 gradient | 7.688 s fixed-weight replay | 0.769 s |
| One daily micro4 gradient | 0.5195 s | 0.0520 s |

Component replays and changing-data complete-update wall clocks have different
measurement scopes and are not additive exact identities. At the measured
32/15.63 = 2.05 daily rows per complete-update second, 8,591,565 TRAIN
draws are about 48.6 days of uninterrupted *row-count-equivalent* work;
10x would be about 4.86 days, excluding startup, monitors, IO and scheduler
delay. This is not an epoch or a verified unique-data traversal. Daily64
with unchanged T7/T30 is a different sampling mix: it cannot be reported as a
same-method 2x gain. Its 8.609 s replay is 2.017x daily32, so simple micro4
accumulation does not raise daily sample throughput.

About 62,993 GPU events occur per daily micro4; 55,525 last at most 10 us.
Matrix multiplies take 42.5% and fused kernels 53.1% of *profiled summed
device time*. This indicates fragmentation and small GEMMs, not that all short
events are launch overhead. Roughly 93% of daily source attribution is
ambiguous or in generic shared helpers, so no end-to-end decoder percentage is
measured. The ~1M-instruction optimized graphs and ~9.2-GB recurrent HLO text
are compiler-complexity evidence, not GPU memory or runtime attribution.

## Width-corrected arithmetic sanity check

The 2026-09-23 static cost and packing audits describe the earlier width-768,
FFN3072 model. In particular, their 10.921-billion-MAC field-local dynamic
K/V count is **not** current split-width work. Applying their source-shape
formulas to the same audited 8,550 eligible / 9,046 padded coordinates, with
the current width384, FFN1536, two decoder blocks and existing *per-field*
dynamic reuse, yields this forward-only approximation per sample-day:

| Counted term | Billion MACs | Basis |
| --- | ---: | --- |
| Decoder FFN | 21.342 | `2 blocks * 2 matmuls * 9046 * 384 * 1536` |
| Decoder query/output projections | 5.336 | `2 * 2 * 9046 * 384^2` |
| Per-field distinct dynamic K/V | 2.730 | `2 * 2 * (1380 axis + 3249 local) * 384^2` |
| Shared family/process/carbon K/V | 1.494 | prior width768 count divided by four |
| Attention dots, readout, scalar inputs | 1.265 | source-shape formulas at width384 |
| **Decoder subtotal** | **32.167** | forward MAC estimate, not runtime |
| Backbone, projected broadcast as written | 35.992 | unchanged width768 source-shape formula |
| Backbone, source-K/V sharing candidate | 13.739 | prior source-cache arithmetic ceiling |

Thus the counted total is about 68.16 billion MACs with source projections
repeated per expert or 45.91 billion under source-K/V sharing. The current
source path computes `source_tokens = input_projection(base.source_tokens)`
once, then broadcasts the same 393-token value into seven expert banks
alongside each expert's 140 axis tokens (`shared_transformer.py:629-646`).
Each backbone block calls `_normalized_attention`, which applies its key and
value linears to the entire 7-by-533 bank (`shared_transformer.py:286-341`).
The source code does **not** cache backbone source K/V across experts. The T7
GPU profile's FP32 `M=3731=7*533, N=768, K=768` GEMM shape (112 recorded
events) independently supports projected work at broadcast-bank shape, but
event counts do not prove exact forward/reverse MACs or exclude compiler
rewrites. The expert-specific source mask is applied only after K projection
and head normalization. Thus projecting the common source tokens once per
block, then combining those K/V values with separately projected expert-axis
K/V, preserves attention inputs mathematically; expert query, source/axis
masks and one softmax remain separate. This is an untested execution change
with floating-order, gradient, memory and full-update parity gates, not a
measured 22.25G-MAC runtime saving. GPP's 2,386-token raw-carbon bank is a
distinct *decoder* route already projected once per decoder block; it is not
part of this backbone source-cache opportunity.

The count is *not* a lower bound on the full model: it omits tokenizer,
physical processes, masks, norm, activation, scatter, loss, AD overhead and
rematerialization. Reusing dynamic projections globally
instead of per field could reduce their 2.730-billion term to about 1.260
billion; even an ideal implementation removes only about 2.2% of the
68.16-billion counted total. This is why the old width768 cost percentages
cannot rank current bottlenecks.

For a deliberately generous hardware sanity check, assume 15.7 TFLOP/s
nominal V100 FP32 peak, two FLOPs/MAC, exactly three dense matmul passes for
forward plus two reverse products, no other work, and the optimistic 45.91G
MAC/sample-day. The fixed update has 32 + 7 + 30 = 69 sample-days, giving
`69 * 45.91G * 2 * 3 / 15.7T = 1.21 s` at *peak* arithmetic rate. The
1.56-s tenfold target leaves only ~0.35 s for all omitted computation,
memory traffic and overhead, and demands ~77% of nominal peak even on this
optimistic counted subset. If source K/V is actually repeated, this same
calculation is ~1.80 s, already above target. This is a feasibility warning,
not proof of impossibility: actual GEMM work/AD recompute and usable V100 peak
need direct measurement. A precision change would alter this bound and the
training semantics, so it is not silently included in the same-FP32 target.

## Ranked same-method routes

1. **Measure and improve whole-decoder scheduling at equal work.** Current
   field-local64 execution uses 265 chunks, of which the separate raw-carbon
   GPP field has its own route. Grouping the other 160 fields *within expert*
   gives 71 chunks at128, 38 at256, 22 at512, and 14 at1024 (from audited
   expert row counts and `ceil(rows/min(chunk, rows))`). The tested expert128
   helper passes small scalar/micro4 output and native-gradient parity but
   has no actual-layout V100 timing yet. The chunk-count reductions of ~3.7x,
   ~7.0x, ~12x and ~19x are **not** compute or wall-speed reductions: FFN
   and projection MACs remain, and larger chunks change padding, GEMM shape,
   temporary memory and numerical order. The five-field 256 controls failed
   at least one output gate; that does not prove a different whole-expert256
   schedule fails, but it bars approval by extrapolation. Keep shared
   family/process K/V banks unreplicated, field-local dynamic banks and masks
   intact, all coordinates restored, and full parameter/input/state gradients
   checked. Use native JAX remat only if measured memory requires it; remat
   may slow updates. This is the highest-value *hypothesis*, not a 10x claim.
2. **Co-design backbone/decoder graph staging with compilation.** The decoder
   FFN is now the largest counted decoder term, while the backbone can be
   comparable. Backbone source-K/V sharing has a source-level identity as
   described above, but its 22.25G-MAC arithmetic ceiling must be checked
   against a bounded same-function executable. A staged kernel/loop
   organization may also give XLA larger useful GEMMs and smaller
   trace/lowering units, but is not yet demonstrated on the full graph. Q1
   grouping saves 4.16 ms and five-field
   grouped128 saves 19.26 ms in separate micro4 helper tests; these are not
   additive full-model gains. Avoid a multi-hour cold compile solely for
   either fragment. Require a profile with resolved stage timing or a matched
   full update before attributing the missing order of magnitude.
3. **Parallel execution, reported separately.** The existing serial/dual
   replay must compare equal daily32+T7+T30 work, complete updates and
   card-hours including gradient/parameter transport. Two cards can overlap
   daily and recurrent components; even ideal daily64 overlap leaves the
   9.50-s T7+T30 branch and a 14.76-day row-count equivalent before overhead,
   but daily64 is not a fixed-work comparison. More cards may improve wall
   samples/s without improving per-card efficiency, and T30 has sequential
   state dependence. Do not label a hardware-multiplier or changed objective
   frequency as a single-card same-method 10x result. Extra cards may still
   contribute to the user's practical throughput goal if total GPUs,
   samples/s, card-hours, cost and optimizer meaning are stated. Lower
   precision/Tensor Core execution could widen arithmetic headroom; a
   different update schedule or smaller-capacity decoder could also change
   wall time. Each changes numerical or scientific semantics and needs its
   own explicit comparison and decision, not quiet inclusion in an
   execution-only speed claim.

## Minimum decisive experiment

The prior monolithic macro-reference screens twice exceeded their 20-min
compile bounds, so use the partitioned native-AD reference oracle now being
prepared to check actual-layout expert128 outputs, masked full shapes and
all parameter/input gradients. Record compiled/actual peak memory and cold
trace/lower/backend times. Do **not** use the intentionally modular oracle's
warm time as the production baseline; compare the candidate's warm helper
time to trusted production execution on matched work. If expert128 passes,
measure at most one larger same-expert tile (256 first; 512/1024 only if its
shape, tolerance and memory evidence justify it). Failure at256 is a result,
not a reason to loosen tolerances. The *promotion* gate is one matched
same-checkpoint, same-data daily32+T7+T30 complete update with full
parameter/state-gradient and AdamW parity, warm samples/s, card-hours, and
startup cost. For the current one-V100 FP32 efficiency axis, the tenfold
marker is <=1.56 s/update **without** changing sampling composition. For a
combined multi-GPU or approved precision route, report fixed-work aggregate
speedup and its resource/semantic costs separately; do not infer 10x from
chunk counts. No new paid submission is implied.

## Startup is independent

The cache-hit profile needed 28.79 min to make three graphs ready; a direct
deserialize-only probe loaded binaries in 6.45 min, and the integrated
direct-load two-update replay reached graph-ready in 8.85 min with full
parameter/AdamW parity. This is a real startup improvement for the unchanged
JAX0.4.38/V100 executable and does not reduce the 15.63-s warm update or
first-ever compilation. The one-time CPU prepared input bundle is about
50 MB and took 201.4 s; its intended replay is pending. Count its preparation,
load and filesystem waits end to end before claiming further startup gains.
Follow-up compact replay80665 passed; see shared_split_dual_compact_results_20260925.md.
Graph-ready was748.5s on this two-process comparison (which duplicates recurrent
loads), and complete probe1254.68s. Worker recurrent deserialization was about
162wall/161CPU seconds per executable. It is not a demonstrated faster total
startup than the earlier single-process8.85min replay. Complete warm daily32
was14.851s serial/11.193s dual; daily64 was20.391s/11.336s. These constrain
the parallelism opportunity but do not establish a10x path.
The separate whole-macro reference has exceeded two 20-min compile bounds;
do not keep recompiling it as the comparison oracle. A material graph change
needs an explicit new compile budget, saved references and the prescribed
on-platform runtime preflight.

Sources: `manifests/coarse_graining/shared_transformer_split_width_v1.json`,
`outputs/shared-split-profile-c599236.summary.json`,
`outputs/shared-decoder-cost-audit-20260923.json` and
`outputs/shared-decoder-packing-audit-20260923.json` (last two are **older
width768** layout/formula evidence),
`shared_split_profile_results_20260925.md`,
`shared_split_serialized_updates_20260925.md`, and the current
`shared_transformer.py` / `shared_decoder_macro_packed.py` source. Graph
index generation 2026-09-25T06:04:58Z has no recorded coverage gaps for the
relevant source files but reports changed metadata; source was read directly.
