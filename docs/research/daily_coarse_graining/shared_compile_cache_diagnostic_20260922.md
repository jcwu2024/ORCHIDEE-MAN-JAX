# Daily-only compilation/cache diagnosis

Both execution benchmarks timed out before candidate measurement. Diagnose
the unchanged daily reverse graph before repeating microbatch/dual probes.
Use the existing shared-execution0507e1b checkout, artifact742, source100 and
unchanged objective. Keep the diagnostic driver in a small declared runtime
tools directory; no additional full checkout, Teacher materialization or
parameter update is required.

## Stages

`probe_shared_compile_cache.py` reuses the existing probe bootstrap and replaces
only its diagnostic compilation callback. Report load, lowering, cache lookup
and backend compilation separately. `key-only` records the actual JAX key and
entry availability at the compiler boundary, stopping before backend compile
or executable deserialization. `cached-only` permits loading existing code but
rejects a backend compilation; `compile` permits a bounded actual compilation.
These modes establish cache/runtime evidence, not gradient/science acceptance.

The internal compiler hook is scoped to this standalone diagnostic and pinned
to deployed JAX0.4.38. Verify its restoration and cold-cache rejection on a tiny
CPU fixture before the actual graph. Bootstrap may compile small supporting
kernels; the no-backend-compilation promise applies to the selected full daily
gradient graph. The numerical model and production training runner are unchanged.

First use an idle free gln01 GPU for key-only, capped15min. It is not an accepted
throughput benchmark. If needed, follow with cached-only at the same bound;
do not escalate a miss into an unbounded compile on the shared free node.
An approved paid comparison will choose an idle node, record resource/load
facts and measure the same daily-only compile/cache operation. Node selection
does not imply whole-node exclusivity or isolate shared storage.

## Prepared paid envelope (not submitted)

- gnall, one V10032GiB,8 allocated CPUs,64GiB host memory,45min wall limit.
- Main process limit40min; stop immediately when the daily diagnostic finishes.
- Worst-case additive estimate:(2.20+8*0.07)*0.75=2.07yuan.
- Proposed node:ibc13b03n03, IDLE at2026-09-22 21:11; recheck before submission.
- Working directory:`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-execution-0507e1b`.
- Diagnostic tools:`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/tools/shared-compile-cache-20260922`.
- Entry:`job_shared_compile_cache.sh`, with explicit CODE_ROOT/DIAGNOSTIC_ROOT
  and `sbatch --nodelist=ibc13b03n03` from the declared working directory.

The user requested idle paid nodes for future probes and continued preparation.
This new concrete45min/2.07yuan allocation has not yet been submitted or approved
as an estimate. Finish free diagnosis and actual shell/tiny checks before
requesting the final submission decision. Do not repeat heavyweight AD checks
for this logging-only instrumentation of an unchanged gradient graph.

## Implemented preparation and current free run

Sol/high implemented the scoped diagnostic;6 focused unit tests and Ruff pass.
The lead reviewed the backend/cache-hit accounting, added separate lowering
start reporting, and prepared the standalone/free and paid launchers. Bash
syntax and actual GPU/Slurm-entry argument expansion pass on the target shell.
Native test02 invocation initially failed importing the container-built NumPy
because host glibc is older; the CPU launcher now uses the same accepted
Singularity image as the GPU environment. No packages were changed.
Actual JAX0.4.38 CPU self-tests then pass: key-only exits without backend
compilation; cached-only rejects the deliberately absent cache without calling
the backend. They read no Teacher data.

Free GPU1 wrapper108553 started2026-09-22 about21:22, main15min ceiling,
using the existing0507e1b checkout and deployed driver in the tools directory
above. Report/log prefix:`runtime/outputs/diagnostics/shared-compile-key-free-20260922`.
Both GPUs had no compute processes immediately before launch. Read the result
before duplicating it or submitting the prepared paid comparison. It makes no
training updates and cannot enter the full daily-gradient backend compile.
