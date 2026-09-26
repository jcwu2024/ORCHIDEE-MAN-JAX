# Saved T7 execution profile

Status: freeGPU1 launcher31735 completed0/passed in324.39s main time,
while independent T30 admission29812 usesGPU0. No new gradient compilation, HLO export, Teacher
preparation, optimizer update, paid job or precision change.

## Completed measurements

All loss/auxiliary/64,884,481 parameter-gradient comparisons again pass the
unchanged tolerance and physical hard counts are zero. Matched three-repeat
warm medians: original1.807469821s, combined1.574453108s(12.89%less time).
Original/candidate deserialization166.91/98.05s. Startup before the report's
internal timer is additional; concurrent T30 compilation is recorded as host
context, not evidence that it caused any measured slowdown.

| Instrumented device quantity | Original | Combined |
| --- | ---: | ---: |
| GPU event count | 380,309 | 299,552 |
| Events at most10us | 347,733 | 268,893 |
| Event span seconds | 2.135319 | 1.743979 |
| Active interval union seconds | 1.800707 | 1.569692 |
| Gaps within span seconds | .334612 | .174287 |
| Summed fused-kernel seconds | .971370 | .889196 |
| Summed matrix seconds | .690804 | .568350 |
| Summed transfer/set seconds | .082546 | .076367 |

Candidate summed device time is56.59%fused kernels,36.17%matrix kernels and
4.86%transfer/set. Explicit GEMM names identify FP32; they do not establish
all fused-operation dtypes or full-module ownership. Prominent SGEMM names
contain32x32/32x128 kernel tiles; these are not recovered operand dimensions.
Short events are not automatically pure launch overhead. CUPTI activity-buffer
and missing TensorFlow-hook warnings occurred, but both traces contain GPU
events; unprofiled warm medians remain the speed evidence.

This supports evaluating independent-window batching as a throughput
hypothesis, alongside evidence-led scheduling/fusion work. Compare the SAME
two existing trajectories serially versus batched, preserving full native
recurrence and mean gradient. A later production batch-size/mix change is a
separate training decision. No new batching/precision graph has been compiled.
Do not infer10x, exclusive physics costs or a guaranteed batch speedup here.

Local report: `outputs/shared-saved-recurrent-profile-T7-20260926.json`.
Local traces/summaries: `outputs/shared-recurrent-profile-{original,candidate}-T7-20260926`
with `.trace.json.gz` and `.summary.json` suffixes. Remote binaries remain
unchanged and reusable.

Reuse the original and combined executables from the passed T7 admission,
the saved fresh update0 and artifact742/window0 compact case. The profiler
performs three synchronized warm repetitions and one instrumented execution
per variant, checks finite/physical results and compares all parameter
gradients at unchanged1e-5/1e-6. Device-event categories, interval union/gaps
and GEMM names can locate broad costs; overlapping summed kernel times are
not exclusive module attribution. Warm unprofiled timing is speed evidence.

The immutable model checkout remains c599236. The local profiler commit is
e0b6377. Staged files are under
`runtime/transfer/shared-saved-recurrent-profile-20260926`; the daily comparison
helper is explicitly loaded from that overlay rather than assumed present
in the old checkout. Python lint and actual target-shell dry-run pass. GPU1
was empty at launch. The prior no-new-compilation daily profiling path is reused.

Report/log: `runtime/outputs/diagnostics/shared-saved-recurrent-profile-T7-20260926`.
Trace directories are recorded in that report. Original optimized HLO already
exists (~9.2GB text); do not materialize/export it again for this profile.
If source attribution is needed later, reuse existing summaries/assets on CPU.
