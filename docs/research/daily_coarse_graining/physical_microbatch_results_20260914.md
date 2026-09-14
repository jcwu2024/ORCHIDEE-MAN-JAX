# Physical microbatch resource results

Both microbatch8 and16 have now executed on the same free gln01 GPU0/V10032GB,
using frozen code76e756e, the same64 TRAIN rows, fresh seed20260911, physical-v2
objective, full109464375 parameters and AdamW. One warmup update was discarded
before three timed updates. No model was retained. All GPU processes exited;
there is no current paid or free GPU job from this probe.

## Actual measurements

| Same batch64 | Micro8 | Micro16 |
| --- | ---: | ---: |
| Median complete-update time excluding full-tree host finite checks, s | 6.094960 | 5.546922 |
| Median checked-total update time, s | 8.444197 | 7.818685 |
| Samples/s | 10.500479 | 11.537931 |
| Actual benchmark peak live GPU allocation, GiB | 16.369336 | 26.690036 |
| Compiled estimated live gradient bytes | 15798461424 | 26884789288 |
| JAX pool fraction | 0.85 | 0.90 |
| Complete timed updates | 3 | 3 |
| Finite parameters/gradient/optimizer and physical hard gates | pass | pass |

Micro16 increases throughput9.880% (update time decreases8.992%) at an extra
10.321GiB peak. Measurements include hot reader/transfer, synchronized gradient
accumulation, physical-count checking and optimizer execution; the separate
full-tree host-finiteness checks add about2s/update. These are three warm-cache
updates, not a sustained global-data measurement or comparison against micro4.

The first micro16 attempt was stopped before execution because its compiled
estimate25.038GiB exceeded the declared24GiB guard. It did not OOM. Micro8's
measured non-executable overhead and the32GiB device justified a bounded
actual-runtime completion: compiled cap26GiB, pool fraction0.90,35min cap.
Only these resource envelope values changed, under
`manifests/coarse_graining/physical_microbatch_runtime_envelope_v1.json`.
The original resource-limited report is preserved. Micro8 was not repeated.
Micro16 reused the unchanged graph/cache and completed278.4s, with100.5s spent
tracing/loading its compiled executable versus978.7s in the original compile.

## Numerical comparison and its limits

CPU comparison confirms model, rows, seed, loss, optimizer and execution commit
identical, allowing only the two explicitly recorded memory-envelope differences.
Complete gradient comparison passes: relative L2 error1.8072461e-7, maximum
absolute error9.1552734e-5, zero failing gradient coordinates. Initial and all
three timed losses agree at the original tolerance. Physical gates are zero.

The strict first-update displacement gate remains **rejected**: relative L2
error1.6966010e-5 is below its1e-3 limit, but4 of109464375 coordinate differences
exceed the original absolute-plus-relative floor. Each difference is exactly
1.1920928955078125e-7, float32 epsilon at unit scale. No tolerance was loosened
and this report does not relabel that machine gate as passed.

| Parameter path/index | Delta micro8 | Delta micro16 |
| --- | ---: | ---: |
| axial_blocks/0/attention_norm_scale[170] | 9.8943710e-6 | 9.7751617e-6 |
| expert_blocks/5/2/ffn_norm_scale[316] | 3.6954880e-6 | 3.5762787e-6 |
| graph_norm_scale[0,3,61] | 8.4638596e-6 | 8.3446503e-6 |
| summary_input/kernel[3,231] | -7.5101852e-6 | -7.3909760e-6 |

These are single-precision-rounding-scale differences with close aggregate
gradients/loss trajectories, not evidence of a new gradient-algorithm failure.
The initial CPU comparison also exposed an unsupported None placeholder in
inactive axis embeddings. The comparator now treats None as JAX's empty tree
node, preserving structure checks; its nine focused tests pass. Only CPU
comparison was repeated; saved GPU gradients and updates were reused.

## Decision and next work

Use batch64/micro8 as the next combined one-day+T7 admission configuration.
Micro8 completed its physical/full-update checks and retains substantially
more memory headroom. Micro16 is demonstrably runnable and faster for this
one-day workload, but its modest speed gain does not justify making26.69GiB
peak the default before measuring mixed/multiday execution. The automated
comparator makes no speed recommendation because its displacement gate fails;
the micro8 choice is an explicit resource judgment, not a claimed micro16
equivalence acceptance. Do not promote batching choice to science acceptance.

Return to shared native-checkpointed combined-loss optimizer admission, then
T30 and expanded learning budget. Do not reopen historical streamed-adjoint
work or indefinitely pursue bitwise parity for this bounded resource probe.

Local evidence under outputs/:
physical-microbatch-8.json, physical-microbatch-16.json (pre-execution stop),
physical-microbatch-16-runtime.json, physical-microbatch-comparison.json.
Corresponding remote reports and diagnostic first_update.pkl containers remain
under runtime/outputs/diagnostics. No Teacher regeneration, paid job or weights
promoted for the final reproducible training method.
