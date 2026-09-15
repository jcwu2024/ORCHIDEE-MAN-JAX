# Recurrent compilation: cold work versus cache reuse

The apparent1980.8s versus221.6s compile contrast did not compare two cold
compilations. Do not attribute that ratio to the checked execution change.
The free cache inspection at427ccce completed in229.80s with exit0, no Teacher
reads, no model executions/updates and no new JAX graph compilation. It loaded
two trusted existing executable cache entries directly on the deployed GPU.

| Existing artifact | Backend compile time recorded in cache | Cache deserialization measured now | Optimized HLO instruction lines |
| --- | ---: | ---: | ---: |
| Checked T7, v2 | 1940s | 66.68s | 520357 |
| Original T30 | 1714s | 41.03s | 482644 |

These are different horizons and historical cold timings, not a controlled
cold/cold performance experiment. They establish that original cold compilation
was also expensive and that the221.6s original call reused its existing cache.
The higher-level `.lower().compile()`/runner measurements additionally include
lowering, cache-key work, loading and framework overhead; they must not be
equated with the cached backend time or direct deserialization measurement.

Serialized executable sizes are197.76/207.58MiB respectively. HLO text rendered
in memory is3.79/1.82GiB, with repeated shapes/metadata; this is neither a saved
multi-gigabyte artifact nor device memory demand. Only compact JSON statistics
are written. Optimized programs contain14261/13310 fusion instructions.
Both graphs are large, despite approximately2.435GB compiled live memory.

Source audit explains why tiny returned observations can retain additional
computation: `canonical_rollout.py` checks all161 learned outputs (39006
coordinates), plus budgets/discrete histories, which the scalar-only loss does
not consume. The all-output finite counter is just one integer per day but
keeps its producers live. This is a plausible structural contributor, not proof
of a compiler bottleneck. Neither removal of checks nor a `prevent_cse` change
is justified by the timing contrast. Sol high independently reviewed this
liveness distinction; the main agent verified the cache evidence.

Decision: keep architecture, precision, loss, native checkpointed AD and every
check unchanged. Reuse completed checked T7 compilation, both saved original
T7/T30 gradient references and the original complete-update timing. Finish the
pending checked T30 comparison and full warm-update measurement in one bounded
free GPU run (65min ceiling, no paid job). Only T30 still requires its first
checked compilation; reference gradients are not recomputed. Reference T7:
`mixed-learning-checked-profile-v1.T7.reference.pkl`; T30:
`mixed-learning-checked-profile-v2.T30.reference.pkl`.

The original train route remains the default until equivalence, memory and
actual warm speed are established. Compilation duration alone does not decide
adoption: training repeats the executable thousands of times and later processes
can reuse cache. Conversely, cached compilation does not prove any warm speedup.
No new model accuracy claim follows from this work.

Evidence: `runtime/outputs/diagnostics/recurrent-cache-inspection-v1.json`
(also copied locally under `outputs/`). Inspector:
`scripts/hpc/inspect_recurrent_executable_cache.py`. Follow-up uses the existing
`checked-profile --separate-checked-validation`, `CHECKED_REFERENCE_REPORT`
pointing to v2, and new report/log `mixed-learning-checked-profile-v3.{json,log}`.

Follow-up launch: target-shell dry-run passes, both GPUs idle beforehand;
free gln01 GPU0, Bash15683/Python15724, immutable427ccce in
`runtime/checkouts/mixed-learning-checked-reuse-v1`,65min ceiling. No paid job.
