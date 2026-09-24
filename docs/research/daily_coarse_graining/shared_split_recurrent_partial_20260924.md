# Split-width recurrent admission: timeout and preserved progress

Free GPU launcher18069 exited124 at the240min timeout. It is not a completed
training or numerical rejection. The JSON still says running because timeout
terminated the process during T30 gradient compilation; the launcher log is
the exit-status authority. Completed optimizer updates:0.

Code caed647; unchanged split-width architecture and native checkpointed AD.
Reports/logs: runtime/outputs/diagnostics/shared-split-caed647-recurrent-gpu.*.

Completed evidence:

- Two actual TRAIN artifacts each passed T7/T30 and scalar daily forward hard
  counters. No observed nonfinite prediction, negative stock or budget failure.
- Daily and T7 full parameter gradients were computed, finite and nonzero.
- T7 JVP1.98895239305 versus reverse1.98895210915 passes unchanged tolerance;
  relative difference1.427e-7. T7 compiled live estimate1.836GiB and cumulative
  actual peak3.687GiB at that stage. These are not T30 or full-update memory.
- Daily gradient ready4461.1s; T7 gradient ready9873.7s (T7 compilation5406.94s);
  T7 direction ready10663.8s. T30 gradient compilation began10666.1s and was
  still unfinished when wall time expired. No warm recurrent throughput result.
- Saved743MiB component-gradient artifact includes daily/T7 results and the
  passed T7 directional record. Preserve it and the persistent compile cache.

At10:23CST September24, GPU0/1 were empty and host load0.13. Launched recovery
on free GPU1, launcher PID100157, same caed647 checkout,240min ceiling, using
existing --resume-report. New report/log:
runtime/outputs/diagnostics/shared-split-caed647-recurrent-resume1.{json,log}.
Old report/gradient files are untouched. Completed gradients/directional evidence
are reused; forwards and later updated-weight gradients still run through the
existing recovery path. No model, tolerance or loss change, no paid allocation.

Next: finish T30 gradient/direction and two joint updates, then actual
daily32/micro4+T7/T30 timing via prepared --technical-updates2 launcher. Its
admission path must use the successful resume report. Do not start formal500
learning or claim stable full training before these complete. Startup compilation
is an observed bottleneck; a warm speed claim must report this separately.
