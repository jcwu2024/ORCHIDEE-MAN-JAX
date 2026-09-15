# FP64 burden and an explicit FP32 attention candidate

Stage1 is now complete: source-shaped V100 run43e039a exits0 in26.85 seconds,
all5 numerical cases pass, local forward+gradient medians4.81--14.99% lower.
See `attention_fp32_primitive_results_20260915.md` for measured values and limits.
No precision task remains running; actual-model precision validation is next.

User priority: establish which FP64 operations could bottleneck a future RTX5090,
which can be FP32 and how much time is saved. This supersedes immediate larger
training as the next performance task. Do not substitute FP16/TF32 investigation
or nominal device FLOP ratios for this question.

Confirmed first target: `_attention` in `strong_baseline_architecture.py` divides
FP32 query-key scores by `np.sqrt(float(head_width))`. With global x64 enabled,
that NumPy scalar promotes scores, softmax and value aggregation to FP64. The
historical assertion that all active neural matrix computation is FP32 was too
broad: only the parameter/projection claim was established. Local lowering shows
one FP64 value-aggregation matrix product in an otherwise FP32 attention block.
Captured old DGEMM trace metadata independently identifies this same operation;
the old trace is incomplete and cannot establish a whole-update time fraction.

Candidate: `_attention(..., score_dtype=jnp.float32)` explicitly casts the scaling
constant to FP32. Default `None` preserves the historical graph/numerics; ordinary
training is unchanged. No parameters, architecture, physical state, loss, mask,
conservation tolerance or native AD algorithm is changed by the candidate.

Stage1 bounded primitive comparison uses active256/384 widths and8 heads with
five declared representative shapes in `scripts/hpc/probe_attention_fp32.py`.
These are synthetic inputs/initial weights, not real sampled model activation
distributions. Compare the SAME weights/input/mask between modes; include a fully
masked query and masked keys without making any whole case trivial. Time forward
and parameter/input value-and-grad separately with synchronization, warmups and15
alternating paired repetitions. Both use highest matrix precision (no TF32/FP16).
Report lowering FP64 matrix removal, output/loss allclose atol1e-6/rtol1e-5,
all-gradient coordinate/per-leaf tolerances from `gradient_comparison`, finite
gradients and exactly zero masked-input gradients. A numerical failure is reported,
not repaired by changing tolerances. Speed is measured even if negligible/negative.
No model weights are saved. Local3 unit tests and all5 CPU functional cases pass;
CPU timings are not GPU evidence. Free gln01 GPU0,10min ceiling; no paid job.

First free primitive pass completed81.14s, exit0, all5 numerical cases pass.
Its15-repeat timings show early clock/runtime warmup drift, so do not treat the
1--21% per-shape gradient reductions as a final full-model estimate. The next
short pass uses source-fixture-derived forcing45 tokens/width256, axial126x32
groups/width384, expert1x533/width384, maintenance24x419/width768 and carbon
feedback2x2386/width384 (`--source-shaped`). Inputs/weights/masks remain synthetic,
not captured trained activations. Add0.5s alternating workload warmup and51
paired repetitions; no clock settings are modified. Each source-shaped CPU
functional case passes before launch. A singleton query case tests masked keys
but cannot also zero its only query without making the whole workload trivial.

Stage2 captures actual attention call shapes/counts and inventories FP64 sources
in the active full-model graph. Small primitive gains alone do not establish
whole-update speed. Complete traces must have bounded scopes and evidence of
coverage; unknown fused-kernel dtypes stay unknown. Do not turn static opcounts
or captured DGEMM times into a total FP64 percentage. Reuse existing compiled
artifacts wherever the graph is unchanged.

Stage3, after selecting material candidates: bind the explicit policy throughout
the active model with recorded method identity and compare fixed trained
forward/loss/gradients and one-day/T7/T30 physical/science behavior against the
unchanged reference. Only then measure complete-update timing and admit broader
learning. Physical-state FP32 is a separate experiment: budgets, stock differences,
threshold/mask decisions and multiday feedback require evidence, so a global x64
disable is not a valid first implementation. Keep sensitive reductions FP64 until
their significance and precision needs are measured. No formal model acceptance
or 50-year correctness follows from short-window or primitive tests.

The same primitive entry accepts `--backend gpu` for a standard CUDA JAX runtime
on another host, allowing direct RTX5090 comparison later. V100 timings cannot
determine RTX5090 gain; arithmetic intensity, memory traffic, launch overhead and
precision-specific hardware throughput all matter. Record measured values and
unmeasured transfer questions separately.
