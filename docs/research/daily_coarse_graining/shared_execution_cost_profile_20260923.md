# Shared architecture: bounded execution cost investigation

Status: static cost audit complete; real-model GPU trace unavailable, 2026-09-23.
No architecture change or new
paid allocation. The 100-to-500 learning extension remains on hold under
`shared_architecture_efficiency_gate_20260923.md`.

## Questions and evidence boundaries

The immediate objective is to distinguish expensive mathematical work from
inefficient execution of that work. Lower parameter count is not evidence of
lower cost when weights are repeatedly applied across output coordinates.
The retained scientific target is useful production, respiration and biomass
trajectories, not minimum execution time independent of learning quality.

The current shared model has 75,578,497 parameters, an eight-block width-768
backbone and a two-block width-768 coordinate decoder, both FFN3072. Existing
gather/reuse removes permanently masked coordinates and reuses common bank
projections. The deployed CPU admission confirms 8,550 eligible coordinates
out of 39,006; this is not a count of the instructions that survive recurrent
compiler elimination. All potentially active fields and source masks remain.

Source review has already confirmed one native checkpointed rollout per
horizon and retained FP32 neural attention. The previous architecture had a
different decoder and narrower recurrent stock objective. Its speed does not
isolate the effect of architecture, nor imply that old accuracy was sufficient.

## Already measured full-update breakdown

The micro4 matched paid replay gives a full-update median of 21.45656 seconds.
A representative replay has daily32 7.77156, T7 2.60982, T30 11.01064 and
AdamW 0.02153 seconds. This shows recurrent work is now the larger branch.
These are warm fixed-data measurements, not sustained changing-data training.
Evidence: `shared_execution_micro24_results_20260923.md` and local
`outputs/shared-staged-15329445.json`.

Two-card placement should consequently consider daily32+T7 on one device and
T30 on the other: approximately 10.38 versus 11.01 seconds of existing serial
component work. The older daily-versus-(T7+T30) split gives 7.77 versus 13.62.
These maxima are idealized scheduling arithmetic, NOT measured dual throughput.
Transfers, concurrent execution, memory, checks and parameter synchronization
must still be measured before adoption. Both placements preserve all three
gradient contributions and perform a single joint AdamW update.

Reference executables really hit the persistent cache with zero backend
compilations. In the micro4 process their daily/T7/T30 lowering took
211.96/288.00/289.88 seconds and loading took 68.67/127.78/128.29 seconds.
Thus cache hits still incurred about 18.6 minutes of lowering/loading for the
three graphs. The separate new micro4 graph then took 44.37 minutes. Do not
call all startup time CUDA kernel compilation or infer node contention.

## Small free-GPU profile

`scripts/hpc/profile_shared_execution.py` reuses the existing trained update100
bootstrap and selects just one unchanged scalar gradient executable. It loads
the persistent executable only; a cache miss stops before a cold model backend
compile. It makes one warmup, three unprofiled replays and one profiled replay,
and writes one compressed optimized-HLO file plus the profiler's two files.
No optimizer update, new Teacher data, model change or retained gradient is
introduced. Tiny bootstrap primitive dispatches can still compile normally.

The existing `run_shared_compile_cache.sh` launcher supports this diagnostic
via its declared `DIAGNOSTIC_SCRIPT` override; no launcher edit was necessary.
Source model checkout is immutable `ef8c23f`,
`runtime/checkouts/shared-staged-large-v1`. Diagnostic overlay is
`runtime/tools/shared-cost-profile-20260923/profile_shared_execution.py`.

A no-data 64x64 GPU self-test completed in 32.06 seconds and exported real
Volta SGEMM, fusion and copy events. TensorFlow Python tracing is unavailable
but GPU events work. CUPTI warned about per-thread buffers: traced wall time
may be strongly perturbed and must not replace unprofiled throughput evidence.
Four local analyzer tests check exclusion of nested host events, overlapping
streams, ambiguous HLO names, absence of GPU events and batched GEMM dimensions;
Ruff passes.

Real T7 profile started on free gln01 GPU1, wrapper PID18199, at
00:24:54 CST September23, with a 25-minute process limit. It reused artifact742
and source update100. Report/log prefix:
`runtime/outputs/diagnostics/shared-cost-profile-T7-20260923`.
At launch GPU1 was unused. The probe ended with `cache_miss`, exit2 after
470.98 seconds: bootstrap186.38s, lowering282.67s and cache lookup1.92s.
No cold backend model compile or real-model execution followed, so no T7 GPU
trace or new throughput result exists. The cache-only boundary prevented an
unplanned long compile; the exact reason for this miss remains unestablished.
Do not replace earlier proven cache-hit evidence with a general claim that
cache reuse fails, or make another cache investigation the primary workstream.

`scripts/hpc/analyze_shared_execution_trace.py` groups actual GPU events into
matrix multiplication, fused kernels, memory operations and other kernels.
It reports summed device durations separately from their interval union and
span; it never adds nested host events to GPU time. HLO source locations are
hints because fused kernels may contain work from multiple modules. Compiler
FLOP estimates can omit custom calls and are not total measured hardware FLOPs.

## Cost finding and recommended next bounded work

The independent source-shape calculator and accepted-layout comparison are in
`shared_decoder_cost_audit_20260923.md`. It estimates 35.99G forward MACs in
shared input/backbone and 215.80G in the coordinate decoder. Within the decoder,
dynamic axis/local K/V projections account for100.59G and FFNs85.37G. Attention
score/value dot products are only0.73G. These are source-level estimates excluding
tokenizer, physical processes, losses and AD, not measured GPU cost shares.

Only339 distinct axis-token variants and1,798 local-state/sentinel variants
underlie the repeated dynamic projections. The strongest first candidate is
therefore identical-token projection reuse. Alongside source-bank reuse and
ideal packing, the optimistic counted total falls251.79G->128.00G MACs; this
49.2% arithmetic ceiling is NOT a promised speedup and may overlap compiler
optimization. Padding alone accounts for only5.48% of query slots.

Proceed first with an optional unchanged-parameter execution path for repeated
axis/local projections, preserving the current gather_reuse path as a bounded
reference. Then consider packing compatible fields in bounded chunks; do not
materialize all outputs together merely to remove loops. Validate forward,
parameter/state gradients and runtime masks locally before one deployed
same-checkpoint measurement. Reject adoption if equivalence fails, live memory
exceeds the existing envelope, or complete updates do not become faster. Reuse
previous numerical evidence for untouched components. No full training is
needed to decide execution equivalence or speed.

Specific remaining opportunities and boundaries:

1. Reuse any remaining identical axis/local projections within one day and
   block. Establish identity from the actual token construction; retain all
   masks, query-dependent values and gradients. This changes execution only.
2. Pack compatible independent coordinate queries across field boundaries.
   Current per-field chunk loops are a possible source of small kernels and
   large compilation graphs. Preserve output layout, ownership streams,
   coordinate embeddings and the single attention normalization. First test
   equivalence, then measured memory/startup/runtime; packing is not a promise
   that compiler cost or memory will improve.
3. If useful execution changes leave intrinsic repeated decoder work too
   expensive, discuss a uniformly more economical decoder with the user.
   A separate decoder width need not equal backbone width, but narrowing it
   changes capacity and requires fresh bounded learning evidence. Do not use
   historical auxiliary labels to select supposedly unimportant outputs.

These are ordered hypotheses, not implemented optimizations or an approved
new architecture. The next decision must combine static counts and the
profile, state which remedy preserves the mathematical model, and specify
what result would reject it. Scientific acceptance still requires actual
trajectory learning and later longer-horizon evaluation.

## Existing micro8/16 jobs

At00:38:57CST job15329519(micro8) was COMPLETED0:0 after1:17:59. The benchmark
completed normally but classified the candidate `resource_limited`: final
compiled-live estimate43,302,192,072bytes=40.33GiB, temporaries39.76GiB. This
exceeds both the existing24GiB operating gate and the physical32GiB card.
Candidate compilation took3162.65s=52.71min. The micro8 candidate was not
executed, so there is no candidate runtime peak, numerical parity or speedup.
Matched scalar baseline did run at28.6311s and reproduced its saved reference.
Estimated allocation charge is3.59yuan at2.76yuan/hour; no retained updates.
Local report:`outputs/shared-staged-15329519.json`.

The earlier30.79GiB rematerialization warning was an intermediate compiler
estimate, not the final live total. Do not report it as observed allocator use.
At the same collection micro16 job15329520 remained RUNNING with a63.22GiB
intermediate warning only. Preserve its eventual result; do not expand memory
limits or submit replacements from this snapshot. micro4 remains the fastest
admitted measured variant, pending the final16 report and execution changes.
