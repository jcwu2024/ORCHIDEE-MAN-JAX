# Resumable shared scalar32 versus micro2 execution measurement

The user requested prompt continuation after15329407 demonstrated a usable
cross-node daily cache hit. Retain the scientific contract in
`shared_execution_benchmark_v1.json`: shared source100, TRAIN artifact742,
32 daily samples, oneT7/oneT30 window, same objective/native AD/precision/AdamW,
rtol1e-5 and atol1e-6. This changes experiment scheduling and result retention,
not architecture, training method or data. micro2 is the first bounded
candidate, not an assertion that micro4/8 is unsuitable or inferior.

## Reference and candidate stages

1. Reference stage: require existing cached scalar daily/T7/T30 executables.
   Preserve each validated component value/auxiliary/gradient atomically before
   loading the next graph. A missing full-gradient cache exits with an explicit
   result instead of starting an unbudgeted cold compile. Preserve the complete
   serial optimizer proposal and warm timing before any microbatch compilation.
2. Candidate stage: load the matching completed reference, load the existing
   scalar/T7/T30 executables, and time the scalar full update in this same
   allocation. Then compile only micro2, test full daily gradient/scalar/physical
   parity and complete parameter/optimizer/joint-loss parity, and measure three
   warm full updates. T7/T30 actually execute in timed updates; saved gradients
   are evidence/recovery assets, not substitutes in the reported runtime.
3. Compare complete update wall time and measured allocator memory, separately
   from compilation and startup. Keep the24GiB compiled-live ceiling. Only then
   decide whether micro4 is worth its additional compilation and memory cost.

One reference state directory stores a bounded number of consolidated pickle
artifacts, not one file per sample or variable. Identity uses the declared
source/configuration/method/runtime; no bulk file hashing. Existing model
checkpoints are read-only and diagnostic proposals never become training
checkpoints. Retain completed components after timeout/failure and resume them.

## Resources

Reference preparation may use an idle free GPU with40min cap. That run's wall
time is diagnostic; candidate speed comparison retimes both paths in one paid
allocation. The prior0507e1b deployed runtime check already exercised the same
vmapped gradient graph; staged orchestration does not change that AD boundary.
Run relevant local/deployed tiny persistence and selection tests and exercise
actual CPU/free/paid argument expansion before launch. Reuse unchanged expensive
scientific/AD evidence.

Prepared micro2 request (not approved/submitted):gnall, one V10032GiB,
8 allocated CPUs,64GiB host memory,2h scheduler ceiling,110min main limit.
Worst-case additive cost:(2.20+8*0.07)*2=5.52yuan. Choose an idle node explicitly
after rechecking occupancy/load, record CPU affinity and visible GPU processes,
and retain the same model/science protocol. Compilation may consume most of
the initial run; the bound is not a prediction. Stop as soon as the one selected
candidate finishes. No automatic micro4 sweep or additional paid allocation.

Launchers:`run_shared_staged_execution.sh` and `job_shared_staged_execution.sh`.
Workdir/command use the immutable deployed checkout, recorded after deployment.
Request final paid approval only after code, actual-shell checks and the
reference prerequisite are ready. No paid job is launched by this document.

## Deployment and current run

Implementation08e7f48 is deployed at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-staged-v1`.
25 focused tests pass locally and in the deployed GPU Python/JAX environment
on test02 using the CPU backend (46.79s). CPU preparation checks actual
source100 checkpoint/metadata in41.5s. CPU/free GPU/paid entry shell expansions
pass. Existing scientific/runtime evidence is reused.

Free gln01 GPU1 reference wrapper6597 started2026-09-22 22:16:54CST with40min
limit; GPU was idle immediately before launch. Source verification passed and
runtime loading started. Report/log prefix:
`runtime/outputs/diagnostics/shared-staged-08e7f48-free`.
Query from gln01 with`ps -p 6597 -o pid,etime,pcpu,comm`; this wrapper's CPU
percentage is not the Python child's utilization. Confirm completion from the
JSON status and log`run_exit_status`, not process absence alone.

After reference completion, the prepared paid command is
`CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-staged-v1 /rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n03 scripts/hpc/job_shared_staged_execution.sh`
from that checkout, subject to user approval and a fresh idle-node check.
No new paid allocation is approved or submitted. No micro2 speedup is claimed.
