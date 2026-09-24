# Split-width recurrent admission: timeout and preserved progress

## Completed recovery (September24)

Recovery100157 COMPLETED0 in7907.20s (131.79min). Report status passed,
classification shared_daily_T7_T30_two_update_technical_admission_only,
completed_updates2. The earlier timeout remains part of total startup cost.
All component and combined parameter gradients are finite; T30 JVP3.91242063216
versus reverse3.91242105707 passes, relative difference1.086e-7. Earlier T7
directional evidence was reused. Both updates and final replays pass physical
counters. Allocator actual cumulative peak4.6638GiB; T30 compiled gradient
live estimate1.8381GiB. This admission uses one scalar daily sample plusT7/T30,
not the intended32/micro4 full training batch.

T30 gradient compilation finished6055.8s after starting399.0s:5656.8s (~94.28min).
T30 direction compilation769.3s. At update2 the newly compiled T30 executable
was reused (5.7s compile/retrieval/report interval); daily/T7 executables restored
after gradient-only recovery took351/630s in their compile/cache paths. No
backend-cache-hit instrumentation here: do not label those timings proven
persistent cache hits. No steady full-update throughput is established yet.

Joint loss before/after update1:0.66748->0.36509; update2:0.41668->0.48246.
Technical pass does not assert monotonic learning or scientific accuracy. User's
present priority is trainability/throughput/data exposure; retain this mixed
learning evidence without using it to derail the scheduled throughput measure.

Next actual two-update daily32/micro4+T7/T30 timing uses the samecaed647
checkout, fresh initialization and isolated technical checkpoint identity,
and the passed resume1 report. No architecture/AD/loss changes or daily
revalidation. Preserve all completed compiled caches and numerical artifacts.

## Original timeout and recovery launch

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

User follow-up: accept the present cold cost provisionally, finish this route's
speed measurement first, and co-optimize compilation with execution only in a
later necessary revision. Recovery log reaches T30 gradient compilation at399s
after reusing the saved daily/T7 results and T7 directional record. This does
not mean the interrupted T30 compiler itself resumed; that unfinished compile
must restart. There is no measured completion percentage or ETA.
