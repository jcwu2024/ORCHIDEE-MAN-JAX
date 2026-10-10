# Shared staged component timing feasibility

Date: 2026-09-30. This is a bounded feasibility audit of cache-only timing for
the original FP32 field staged executables. No model execution or compilation
was started for this audit.

## Question and decision

Can the five saved real-model stage executables be replayed independently on
artifact 742 row 0, with complete synchronized returns and memory measurements,
without a backend compile?

**Not with the available entry point and evidence. No replay was attempted.**
The saved cache is incomplete for the whole native-AD call: the neural
transpose has saved IR but no executable. The existing native staged entry
point reaches all six compile units as one `value_and_grad` execution and only
returns the complete gradients and diagnostics after the sixth unit. The
instrumentation can refuse a required cache miss before backend entry, but it
does not expose independently callable, identity-verified entry points for the
five internal forward/transpose units. Reconstructing those internal AD calls
as separate functions could alter their native cache identity. Trying them
would therefore not satisfy the cache-only constraint.

## Evidence

The authoritative prior result is
`shared_neural_precision_results_20260927.md`, section “Final bounded outcome
and structural follow-up”; its local raw report is
`outputs/diagnostics/shared-neural-reference-20260927-cache-fixed.json`.
It records five completed compilations followed by a 600-second timeout while
preparing neural transpose ordinal 5. There is no full gradient/result pickle
or complete warm timing. The cache contains the five earlier executables and
their total size is 15,763,738 bytes.

Read-only inspection on `cln01` on 2026-09-30 confirmed that the cache directory
still exists under the declared runtime root and the benchmark checkout is
exactly `c599236bcf57b745e9bafed678298010f4c3b58f`. The five target executable
files and byte sizes are:

| Cached unit | Executable bytes |
| --- | ---: |
| Neural forward (`jit_neural_stage`) | 2,879,028 |
| Physical forward (`jit_transition_stage`) | 9,443,116 |
| Loss forward (`jit_loss_stage`) | 685,726 |
| Loss transpose (`jit_loss_stage`) | 272,695 |
| Physical transpose (`jit_transition_stage`) | 2,483,173 |

The exact call path is in `scripts/hpc/probe_shared_neural_precision.py`:
`staged_daily_functions` creates the three jitted owner stages;
`native_staged_dispatcher` composes them under native `value_and_grad`; and
`run` calls `jax.block_until_ready` on the full gradient result. Its
`stage_compilation_evidence` wrapper has the required-cache-hit guard, but the
warm repeats occur only after that full result returns. This makes the missing
neural transpose a mandatory stop for a safe complete-result timing.

The later precision-variant cache compatibility result
`outputs/diagnostics/shared-union-cache-compatibility-paid-20260929-15479295.json`
is not evidence of a miss for the original FP32 code. It is a different neural
variant: its ordinal-0 canonical computation digest differs from the saved
union reference, and its cache-only guard stopped before backend entry. It
does demonstrate why a cache miss must not be bypassed or labeled as reuse.

## Execution record and limits

- Remote activity was limited to listing the existing cache files and reading
  the immutable checkout HEAD. No files were written remotely.
- No launcher, scheduler submission, GPU process, model call, backend compile,
  or timing sample was produced for this audit.
- Historical per-unit compile/cache-lookup times are not warm execution times.
  No synchronized stage wall time or per-stage compiled memory measurement is
  claimed here.
- Independent component timings, if later obtained from a proven cache-only
  interface, would describe that interface and its returned values. They would
  not establish an exclusive neural/physical split of a full optimizer update;
  stage boundaries, fusion, synchronization, and gradient/diagnostic return
  values affect such measurements.
- Production precision remains FP32. This audit does not change loss,
  sampling, gradients, training defaults, or the immutable checkout.
