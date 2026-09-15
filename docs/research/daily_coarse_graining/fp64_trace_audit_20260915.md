# FP64 trace audit

## Scope and status

This is a bounded audit of the completed mixed-learning profile trace. It does
not change a model, precision policy, runner, data, or training result, and it
does not claim a V100-to-RTX-5090 speedup. Its purpose is to state what the
existing trace can establish about FP64 execution and define a minimal next
measurement that can attribute the attention precision candidate safely.

The profile is the free `gln01` V100 run reported in
`mixed_learning_performance_profile_20260914.md`, from immutable checkout
`6f26537`. The retained source artifacts are:

- local copied Chrome trace:
  `outputs/mixed-learning-profile-v1.trace.json.gz` (11,628,610 bytes);
- deployed trace and XPlane:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-profile-v1.trace/plugins/profile/2026_09_14_22_13_15/`;
- deployed runner log:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-profile-v1.log`.

**Trace completeness status: not complete for whole-update GPU attribution.**
The trace status was `completed`, but that only means export succeeded. It does
not mean every GPU event required for the 16.883-second replay was retained.

## Observed evidence

The runner records the profile trace stage at 997.2 seconds and completed at
1048.8 seconds. During collection, its log contains:

```text
W .../xla/backends/profiler/gpu/cupti_tracer.cc:1193]
Fail to use per-thread activity buffer, cupti trace overhead may be big.
CUPTI ERROR CODE:1
```

The copied trace contains 61,189 GPU duration events. Its first-to-last GPU
event span is 3.825394801 seconds and the union of their durations is
2.549987394 seconds. In contrast, its enclosing host `replay` event is
16.883227063 seconds. Consequently neither the union nor a sum of GPU event
durations represents whole-update GPU utilization.

The captured event groups are:

| Captured group | Event count | Captured duration (s) | Interpretation |
| --- | ---: | ---: | --- |
| SGEMM | 10,480 | 1.769231867 | FP32 BLAS calls captured during a partial interval |
| DGEMM | 100 | 0.008701708 | FP64 BLAS calls captured during a partial interval |
| Other kernels | 50,609 | 0.772053819 | Includes fusions; name alone does not disclose arithmetic dtype |

The 100 DGEMM events are named `volta_dgemm_*`, so they establish FP64 for
those captured BLAS calls. Their trace arguments map them to `hlo_module =
jit_batch`, `hlo_op = custom-call.*`, and an attention value-aggregation
`dot_general` under the JVP path:

```text
jit(batch)/jit(main)/jvp(vmap(...hqk,...hkd->...hqd))/dot_general
```

That is useful source attribution for a captured path, not a global FP64 time
share. In particular, 0.008701708 seconds must not be divided by either the
2.549987394-second captured union or the 16.883227063-second replay to report
an FP64 fraction. The `other` group cannot be classified from kernel names:
fused softmax, conversions, reductions, and elementwise work can be mixed or
have dtype hidden by fusion.

The DGEMM records contain no tensor dimensions. Their only argument keys are
`context_id`, `correlation_id`, `hlo_module`, `hlo_op`, `name`, occupancy
fields, `scope_range_id`, and `tf_op`; no key or value contains a shape. Volta
tile names such as `volta_dgemm_64x64_nn` are kernel implementation choices,
not proof of the logical query, key, value, or batch shape. The old trace
therefore cannot recover a real primitive shape for a new benchmark.

## Deployed profiler interface

The deployed canonical GPU environment is JAX and jaxlib `0.4.38` at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu`. Its installed
`jax/_src/profiler.py` defines only:

```python
jax.profiler.start_trace(log_dir, create_perfetto_link=False,
                         create_perfetto_trace=False)
jax.profiler.trace(log_dir, create_perfetto_link=False,
                   create_perfetto_trace=False)
```

It also supports `TraceAnnotation` and `StepTraceAnnotation` for source
labels. The lower-level `ProfileOptions` stub exposes host/Python tracing,
HLO proto, timestamps, duration, dataset operations, and repository path. The
public 0.4.38 `trace`/`start_trace` functions do not accept such an options
object, and none of these supported settings controls CUPTI activity-buffer
allocation. `create_perfetto_trace=True` only writes another trace file; it
cannot restore missing device coverage.

No unverified environment variable or JAX option should be added to enlarge
CUPTI buffers. The installed API gives no supported knob for that repair.

## Attention candidate and required primitive measurement

The active strong-baseline attention implementation historically evaluates
`np.sqrt(float(head_width))` as a NumPy float64 scalar. With x64 enabled, the
subsequent division promotes FP32 score output; its softmax and the attention
weight/value aggregation can therefore be FP64 despite FP32 learned
projections. The existing captured DGEMM source path is consistent with that
mechanism. The precision-study `score_dtype` hook in
`research/daily_coarse_graining/daily_operator/strong_baseline_architecture.py`
is the correct narrow comparison point. Default behavior must remain the
historical behavior until an explicit candidate passes its own gates.

Two other implementations contain the same raw NumPy-scale pattern:
`daily_operator/neural_architecture.py` and
`daily_operator/successor_architecture.py`. They are not automatically in
scope for a strong-baseline result; identify the deployed caller before
generalizing a primitive result.

Use a fresh real-shape primitive probe on the deployed V100 before any full
panel benchmark. Obtain `(batch, query length, key/value length, model width,
heads)` from the actual frozen inputs at the primitive call boundary, then
construct no synthetic substitute dimensions. The recommended six cases are:

| Case | Mode | Exact shape source |
| --- | --- | --- |
| 1 | Daily micro8 forward | first fixed daily microbatch at the attention call |
| 2 | Daily micro8 JVP or backward-relevant primitive | same microbatch and tangent path |
| 3 | T7 forward | fixed T7 profile window at the call |
| 4 | T7 JVP or backward-relevant primitive | same T7 window and tangent path |
| 5 | T30 forward | fixed T30 profile window at the call |
| 6 | T30 JVP or backward-relevant primitive | same T30 window and tangent path |

For every case, run the default score path and the explicitly FP32 score path
with identical FP32 projections, inputs, mask, and JAX x64 setting. Compile
and synchronize outside the timed samples; then take repeated synchronized
samples using `block_until_ready`. Record medians and ranges, output/gradient
differences under the candidate's declared numerical tolerance, and the exact
dimensions. This isolates the causal precision change without treating a
changed full model as a timing experiment.

Lower each primitive before timing. The default lowering must show the
attention score-to-softmax-to-value path in f64, while the FP32 candidate must
show that this primitive contains no f64 attention score, softmax, conversion,
or value-aggregation operation. This HLO assertion is necessary because kernel
names cannot classify fused operations. Timing is then an independent outcome,
not a type inference.

## Later trace protocol

Only run a trace after the primitive result warrants wider attribution. Use
one boundary-synchronized component per trace, never a complete mixed update:
the attention primitive first, then, if useful, separate daily, T7, and T30
loss/gradient calls. Wrap each scope in a stable
`jax.profiler.TraceAnnotation` or `StepTraceAnnotation`, run one warmup before
the trace, and block before leaving the annotated scope. Preserve the runner
wall duration, trace directory, JAX/jaxlib version, XLA/JAX environment, and
the collector log with each artifact.

A component trace is usable for scoped source attribution only when all of the
following hold:

1. the collector log has no CUPTI activity-buffer warning or profiler error;
2. the trace contains the expected named scope and device events inside its
   time interval;
3. the device event span is plausible against the independently synchronized
   component wall time; and
4. repeated short traces have stable event counts and grouped durations under
   a tolerance declared before collection.

These checks can reject another clearly incomplete capture. They cannot prove
absolute absence of CUPTI event loss because JAX 0.4.38 exposes no loss counter
or buffer configuration. If the warning recurs or these checks fail, retain
wall-clock component timing and HLO dtype evidence, mark the trace unusable,
and do not estimate a whole-update FP64 or GPU-utilization share from it.

The V100-to-RTX-5090 comparison should use the same admitted primitive cases,
software-level dtype checks, warmup, synchronization, and reporting fields on
both devices. A new hardware result may compare measured median wall times; it
must not extrapolate from this partial V100 trace.
