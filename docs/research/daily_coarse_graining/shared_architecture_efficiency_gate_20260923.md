# Architecture and execution efficiency before larger training

User clarification2026-09-23: formal training has not begun. The completed100
updates are a bounded learning pilot. Before committing to broad data coverage
and long training, establish a scientifically reasonable AND computationally
viable architecture, alongside architecture-independent performance work.
The current shared architecture is not automatically frozen as the final model.
Discuss a concrete architectural revision with the user before building it.

## Evidence to preserve

The shared pilot learned but has not passed scientific accuracy acceptance.
Current micro4 full update21.46s is25.24% faster than its matched scalar path,
yet T7+T30 costs13.62s vs the historical different-model dual branch6.2-6.5s.
New parameter count75.58M vs old109.46M does not measure operations per day.
The shared backbone/coordinate decoder is width768/FFN3072; the old primary
target-oriented coordinate path was384/1536, with different output-category
decoders. Shared weights are reused across many coordinates and days.

Bounded source review confirms native daily-step checkpointing, one recurrent
rollout per horizon and FP32 neural attention remain in the shared route.
The shared loss supervises7 stocks vs the older3, so these are not same-loss
architecture ablations. Existing timings suggest increased per-day work;
they do not quantify the contribution of decoder, loss or execution structure.
Do not diagnose a forgotten AD optimization or attribute all delay to width.

## Two coordinated workstreams

Architecture review: map the repeated operations in input encoding, shared
backbone, output coordinate decoding and retained physics. Inspect actual
tensor sizes and compiler/runtime hotspots, distinguishing static estimates
from measured work and dead-code elimination. Determine whether costly work
supports required long-horizon outputs or reflects a poor representation or
execution choice. Consider only a small number of mechanism-backed revisions
after locating the bottleneck. Attention is not rejected by a slow implementation.
Parameter count alone and historical auxiliary labels do not justify capacity
cuts. Preserve all potentially active outputs and physical/parameter routing.

Execution review: reuse accepted FP32 boundaries and gather/projection reuse;
investigate packing small coordinate operations, device/host synchronization,
graph construction/cache loading and balanced device placement where evidence
supports them. Changes claimed to preserve function need numerical and full
update parity. Precision changes are numerical-method changes, not automatically
function-preserving. Do not rerun completed precision studies by default.

## Admission and next deliverable

First produce a bounded cost breakdown and a small concrete recommendation:
what is expensive, which facts support the explanation, whether the remedy
changes model function, expected opportunity and a test that could disprove it.
Existing micro8/16 probes continue under their approvals; collect their results.
No extra paid allocation or architectural implementation is authorized merely
by this document. Keep the prepared100->500 data plan available, with GPU
continuation behind this review rather than automatically following micro selection.

Architectural suitability cannot be established from shape counts or speed
alone. A proposed candidate must pass physical/gradient admission and bounded
actual-data learning with the fixed production/respiration/biomass readouts.
Assess quality against compute spent, not only equal update counts. Report
startup, sustained changing-data throughput, memory, and cost to reach a
declared useful learning assessment. Larger formal training still requires
the consolidated reproducible method and longer-horizon scientific evidence.
