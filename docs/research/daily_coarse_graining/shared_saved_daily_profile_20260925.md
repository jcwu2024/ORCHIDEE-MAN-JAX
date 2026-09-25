# Saved daily executable profile after expert grouping

FreeGPU1 probe90516 completed0 on2026-09-25. It loaded the original and
admitted expert128 executables at the same fresh update0, artifact742 first
four rows and canonical daily objective. No gradient graph was compiled and
no parameters were updated. The main profile phase took91.8s after startup;
this excludes initial imports/context/checkpoint loading before its first event.

| Metric | Original | Expert128 |
| --- | ---: | ---: |
| Median unprofiled warm time | 0.504113s | 0.415650s |
| Executable deserialization | 44.447s | 11.467s |
| GPU device events | 62,617 | 30,801 |
| Events lasting at most10us | 55,205 | 23,362 |
| Profiled device event span | 0.520641s | 0.423164s |
| Profiled device active union | 0.484538s | 0.405105s |
| Gaps within recorded device span | 0.036103s | 0.018058s |
| Summed matrix-kernel time | 0.205953s | 0.169093s |
| Summed fused-kernel time | 0.257857s | 0.213228s |

Warm throughput is1.213x, consistent with the preceding1.202x admission.
Both physical hard-count sets remain zero and gradients are finite; this
replay does not repeat the full coordinate comparison from admission.

Halving device event count did not halve execution time. Expert128 still
spends approximately42% of summed device duration in identifiable matrix
kernels and53% in fused kernels; most of the recorded device span is active.
This motivates removing structural duplicated arithmetic, including shared
source projections, rather than treating every short kernel as launch overhead
or assuming chunk count predicts wall-time savings. Kernel categories do not
identify exclusive model owners, and fusion is not synonymous with overhead.
There is no complete mixed-update speed result here.

The profiler warned about missing TensorFlow Python hooks and CUPTI per-thread
activity buffers. GPU events were present in both traces. Instrumentation can
perturb timing; use unprofiled warm measurements for throughput and treat
the recorded categories/span/gaps as diagnostic evidence. No HLO source mapping
was exported in this bounded pass, so no exclusive backbone/decoder fraction
is claimed. Serialization reuse is demonstrated for these unchanged binaries
on this JAX0.4.38/V100 runtime; it is separate from first compilation costs.

Reports: `outputs/shared-saved-daily-profile-20260925.json` and
`outputs/shared-saved-daily-profile-{original,expert128}-20260925.summary.json`.
The two local `.trace.json.gz` files total about2.26MB. Remote reports/traces
are under `runtime/outputs/diagnostics/shared-saved-daily-profile-20260925*`.
Existing `analyze_shared_execution_trace.py` summarized device events.
