# Saved T7 execution profile

Status: freeGPU1 launcher31735 running with10minute ceiling, while independent
T30 admission29812 usesGPU0. No new gradient compilation, HLO export, Teacher
preparation, optimizer update, paid job or precision change.

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
