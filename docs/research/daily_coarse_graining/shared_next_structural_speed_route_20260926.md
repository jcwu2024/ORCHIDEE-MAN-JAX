# Next structural speed route: neural arithmetic, not another tile sweep

Completed evidence now supersedes the preparation uncertainties below:
[`shared_ffn_precision_results_20260926.md`](shared_ffn_precision_results_20260926.md).
Actual-input capture and two arithmetic approaches completed on free V100s.
The native dot-policy route gives1.566x micro4/1.198x micro1 FFN forward+VJP
speed, with finite values and measurable numerical differences; production
precision and full-update speed remain unaccepted. The next boundary is
broader neural execution, not another FFN fixture or immediate training.

Status: bounded component implementation, September 26, 2026. Following the
precision proposal, the user directed continued speed work and explicitly
rejected starting training at the present speed. Proceed with the free-node
component diagnostic below, without adopting a production precision policy,
changing architecture or starting training. No new paid allocation. The
admitted combined FP32 binaries remain the reference. The completed daily64
test is a separate load-balancing and sampling-count experiment, not a
same-workload tenfold result.

Follow-up47778 completed0/passed: dual daily32/64wall9.8594/9.7690s and
64branches daily8.313s/recurrent8.326s. Larger daily count fills the idle
interval and approximately doubles daily rows/s, but leaves both branches'
~8.3sabsolute compute cost. The larger-speed reasoning below still applies.

## Recommendation

Implementation: `scripts/hpc/probe_shared_ffn_precision.py` and free-only
`run_shared_ffn_precision.sh`. Capture63001 completed0 on freeGPU0 in319.87s
after the81.10s runtime preflight. It saved576 FFN calls from four real samples,
both coordinate blocks, in one112MiB payload. Six chunk sizes2/14/24/28/72/128
are present;268 of288 calls per block have128rows. Source is the immutable
c599236 checkout plus a small staged overlay, with no full checkout copy.
Reports are `runtime/outputs/diagnostics/shared-ffn-{capture,fragment}-20260926.json`;
saved inputs are `shared-ffn-payload-20260926.pkl`.

First fragment67557 failed before backend compilation: importing the baseline
FFN for the first time inside JAX tracing initialized a physical lookup table
under a tracer. Eager module initialization fixes this diagnostic entry issue;
eight tests including a fresh-process fragment CLI pass. Capture/preflight need
no repetition. Review also caught a measurement-layout issue: real daily micro4
uses vmap, whereas the initial fragment looped saved sample chunks serially.
Repack the same saved inputs to preserve the four-sample batch axis before the
speed comparison; keep a separate micro1 mode for recurrent-day shapes. This
changes only diagnostic execution, not production sampling or architecture.

Test **neural-only mixed precision** as the next materially
different compute route, starting with the decoder's large FFN workload.
Keep FP32 master parameters, optimizer/gradient accumulation, GELU,
normalization and bounded attention scores/softmax. Preserve the current
precision of physical updates, retained processes and science losses, and
the full checkpointed recurrent AD. Only named neural matrix operands would be candidates
for FP16 conversion and FP32 accumulation/output. This changes numerical
semantics, even with FP32 outputs: it is not an exact-FP32 optimization and
needs an explicit numerical/scientific validation contract before production
adoption. This initial diagnostic reports accuracy and speed; it does not
claim adoption or established Tensor-Core execution before kernel evidence.

This is a way to test substantially different arithmetic throughput, **not a
supported tenfold forecast**. Architecture reduction is not the default next
step. There is no evidence here justifying smaller active heads, fewer outputs,
weaker recurrent gradients, or lower capacity. If precision is not acceptable,
the remaining exact-function work should be scoped as fusion/dataflow research
with an unmeasured ceiling, rather than another promised path to tenfold speed.

## Current evidence and the critical path

The completed combined full-update test41280 passes unchanged daily32+T7+T30:
12.553775s on one V100 and10.270496s on two, including transport/checks/AdamW.
Dual improves wall throughput1.222314x while consuming1.636240x GPU-seconds.
All64,884,481 mixed-gradient/parameter coordinates and full AdamW checks pass.
Startup is still406.2s of graph loading, with no new gradient compilation.

In the second changing dual update, daily is4.326s, recurrent is8.732s,
branches including IPC are10.842s and merge/check/AdamW is0.136s. Doubling
daily rows may occupy otherwise idle daily-GPU time; it cannot make either
daily row evaluation or the recurrent day transition intrinsically faster.
Do not attribute all branch overhead to pure transfer from this observation.

Combined component evidence is modest but real: daily micro4 .501539 to
.374680s; T30 7.681579 to6.730945s. Equal-two-window T7 batching improves
3.155881 to2.241723s (1.407792x) at estimated live5.660GiB, but does not
reduce the dependent-day count or establish one-window full-update speed.

Saved T7 candidate profiling shows299,552 GPU events, with268,893 at most
10us. Those short events account for .820737s,52.24% of summed device time;
they are not all removable launch overhead. Summed categories are56.59%
fused kernels,36.17% explicit FP32 GEMMs and4.86% transfer/set. Every event's
source is unmapped in this saved summary; **fused does not mean retained
physics**, and neither exclusive decoder nor exclusive physics wall time is
known. Instrumented event sums are not additive unprofiled wall-time budgets.

## What the source establishes

The admitted adapter `shared_decoder_expert_execution.py:28-200` builds
canonical field features and masks, projects reusable banks, keeps GPP's
raw-carbon path separate, groups non-GPP coordinates by expert and restores
full output shapes. `shared_decoder_macro_packed.py:25-86,288-371` keeps the
two complete coordinate blocks and readout inside each128-row map body. Every
chunk still performs query/output projections, residual normalization and
the two FFN matrix products. Expert scheduling does not remove that arithmetic.

The width-corrected source budget counts decoder FFN21.342G forward MACs per
sample-day, query/output projection5.336G, total decoder32.167G; the optimistic
source-shared backbone is13.739G. These are estimates using the audited padded
coordinate layout, not measured current kernel totals. They omit activation,
mask/gather/scatter, loss, physical work, AD recomputation and memory traffic.
FFN is about66% of counted decoder MACs and46.5% of that counted combined
neural subset. It is a substantial arithmetic target, but not a measured46.5%
of training wall time.

Exact-function opportunities remain, but are not demonstrated large gains:
`shared_transformer.py:455-535` normalizes projected family/process/carbon
keys inside the attention helper; dynamic keys are normalized after gathering
coordinate instances. A unique normalized-key cache could move row-independent
normalization before reuse/gather, retaining normalized nonzero-bias fallback
keys for inactive rows. Native gradients must still flow through the shared
cache. XLA may already hoist invariant work: source repetition alone does not
prove repeated device reductions. True kernel fusion could avoid intermediate
score/mask/weight materialization, but must retain clipped normalized scores,
one softmax over all segments, the all-masked rule and the segmented value-sum
semantics. Vanilla FlashAttention is not a drop-in replacement for these rules.
Any new backend kernel must establish V100/JAX0.4.38 support before integration.

Do not reopen already weak candidates under new names. Actual block-major
dense packing previously saved only5.30% of a width768 four-field micro4
fragment and increased temporaries38.49%. Concatenating attention segments
into two contractions saved8.36% of one width384 chunk and increased
temporaries73.4%. Those fixtures differ from the full current layout, so they
are not impossibility proofs, but neither warrants another full recurrent
compile. A normalized-bank cache is different from that tested concatenation;
it remains a bounded candidate, not the primary larger-throughput claim.

## Smallest informative experiment, after the precision decision

Use one **decoder-FFN-only forward plus native parameter/input-VJP comparison**
at the admitted width384/FFN1536/two-block shapes. Reuse update0 and existing
artifact742's prepared daily context/micro4 inputs; no Teacher preparation.
Extract FFN input payloads from the unchanged neural forward, preserving the
actual expert128 row counts, padding, dtype and active-mask distribution. Use
all the FFN calls in those two decoder blocks, not a convenient large synthetic
GEMM. Keep the reference and candidate's grouping/chunks identical so precision
is the only treatment. This is a component diagnostic, not scientific training.

Payload extraction itself is not implemented or timed. The saved full-gradient
executable does not expose intermediate FFN activations. First demonstrate a
bounded producer from the existing prepared inputs; a changed intermediate-
return graph may require compilation. If extraction cannot stay within the
preparation budget, stop and revise the screen rather than assuming the saved
binary can supply those activations or hiding a full-model compile in setup.

The narrow implementation hook is `_ffn` as used by
`_run_coordinate_blocks_reusing_projections` in `shared_transformer.py:537-607`,
with the same `_linear`/GELU/residual/norm formulas from
`strong_baseline_architecture.py`. The future probe can reuse the staged
loaders/context and patch scoping in `probe_shared_expert_real_daily.py`, and
the forward/native parameter-and-payload-gradient measurement pattern in
`probe_shared_dense_stages.py`; it must not copy the old width768 fixture,
rebuild Teacher inputs, invent rollout cotangents or change public science loss.
Scope the hook to decoder coordinate blocks: backbone FFNs and retained
physical code must not be silently included in the treatment.

The current `_linear` explicitly casts its input to FP32
(`strong_baseline_architecture.py:886-898`). Casting an input before calling
that unchanged helper would not implement this experiment. An isolated FFN
overlay would need two explicitly typed matrix contractions, with FP16
operands and requested FP32 accumulation/result, followed by the unchanged
FP32 biases and GELU. Do not globally replace `_linear` across physical and
neural callers. Confirm the actual deployed lowering rather than inferring
accumulation precision or Tensor-Core use from a source cast.

Compare unchanged FP32 matmuls with explicitly declared FP16 operand casts,
FP32 accumulation and FP32 output before the unchanged FP32 nonlinear/control
operations. Verify the actual lowering and explicit Tensor-Core kernel evidence;
a cast in source is not proof of Tensor-Core execution. Keep native AD through
casts and both products, including all FFN weights/biases and input gradients.
Record finite values, pointwise errors, relative errors, kernel counts/dtypes,
warm synchronized complete-fragment time, and live/temporary memory. Save the
FP32 result before compiling the candidate. No massive HLO export is needed.

Proposed bounds: CPU payload/shape inspection10min, on-platform runtime check
3min, then one free-GPU fragment process10min total with at most5min per
backend compile. Persist each completed reference/compile/timing result before
the next stage. A timeout or unsupported V100 kernel is a result, not grounds
for an immediate hour-long whole-model compile. This note does not launch it.

Use the existing1e-5/1e-6 execution comparison to report whether exact-route
admission holds; FP16 may fail it. Do not loosen that gate to label mixed
precision exact. A failed pointwise comparison is numerical evidence for the
user's separate precision-method decision. Only after that decision and a
declared error/learning/physical contract could real daily/T7/T30/full-update
checks establish an accepted numerical method. The fragment alone proves
neither recurrent gradient quality nor scientific accuracy.

## Honest gain bounds and the next decision

For this first diagnostic, the report must separate three claims:

- Speed: synchronized forward and forward-plus-native-VJP time for the same
  saved activation calls, shapes and repetition order; include casts and
  FP32 bias/GELU. Record both the absolute time saved and ratio, plus input
  extraction, compilation and profiling time outside the warm measurement.
- Numerics: compare output, every FFN parameter gradient and input gradient;
  retain rtol1e-5/atol1e-6 as the exact-route diagnostic, and separately report
  finite counts, absolute/relative L2 errors and cosine. A synthetic output
  cotangent must be labeled as such: it is not the real daily/recurrent loss
  derivative. Inspect backward contraction dtypes as well as forward dtypes.
- Scope: finite arrays and a fast component do not admit a new numerical
  training method. No optimizer, trained accuracy, physical budget or recurrent
  derivative is established by this fragment. Production stays FP32.

The bounded payload producer may execute unchanged neural-forward chunks
sequentially and return their pre-FFN activations. It must record which
checkpoint and real rows it used; a profile generated at technical update2
must not be relabeled fresh initialization. This producer's diagnostic timing
is not production throughput. Stop and preserve completed progress at its
deadline rather than automatically expanding to a full-model compilation.

If just the counted FFN arithmetic became8x faster, the optimistic neural
MAC-only ratio is approximately `45.906 / (24.564 + 21.342/8) = 1.69x`.
This is not a wall-time bound or prediction. Adding query/output projections
to that hypothetical treatment gives roughly2.03x in the same MAC-only budget.
FFN-only conversion cannot credibly be sold as a tenfold whole-update route.

On the saved T7 device-category accounting,8x acceleration of every explicit
matrix kernel gives only `1 / (0.638279 + 0.361721/8) = 1.46x`; even removing
all such matrix time gives1.57x. Hypothetically accelerating **both** all fused
and all matrix work8x gives5.32x, and optimistically removing all fused work
while accelerating matrices8x gives8.51x. These category illustrations assume
an unrealizable degree of ownership/removability and are not exact wall bounds.
They expose why nominal Tensor-Core throughput alone does not support10x.

If the fragment gives no substantial absolute saving, stop before recurrent
integration. If it wins, quantify its share with a bounded real one-day
neural-forward/native-reverse boundary measurement, then seek the declared
precision-method decision before full integration. Remaining fused/retained
cost must be attributed before choosing a capacity reduction. Wider hardware,
larger batches and architecture changes are separate resource/method decisions;
none is silently folded into this proposal. The goal remains effective complete
training throughput with unchanged scientific outputs and recurrent gradients.

## Evidence scope

Read: `shared_training_tenfold_budget_20260925.md`,
`shared_saved_recurrent_profile_20260926.md`,
`shared_combined_full_update_20260926.md`,
`shared_dense_stage_admission_20260923.md`,
`shared_split_fused_attention_screen_20260925.md`, and
`outputs/shared-recurrent-profile-candidate-T7-20260926.summary.json`.
The completed September26 measurements supersede historical proposed/running
statements in older reports. No new measured speed result is introduced here.

Graph Verify search used project E-ORCHIDEE-MAN-JAX generation
2026-09-26T04:19:57Z with complete relevant search pagination. Cited primary
decoder/backbone source paths had no recorded gaps but changed metadata;
exact source was read. This is a bounded source/profile assessment, not an
exhaustive compiler audit or measured neural-versus-physics attribution.
