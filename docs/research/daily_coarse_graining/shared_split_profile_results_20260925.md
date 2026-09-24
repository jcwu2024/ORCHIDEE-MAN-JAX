# Split profile: reuse works, throughput remains inadequate

Status: completed performance diagnosis, 2026-09-25. No architecture, loss,
Teacher dataset or scientific acceptance threshold changed. Long learning
remains on hold. This report supersedes the running profile31906 snapshots.

## Completed runs and provenance

Free gln01 GPU1 launcher31906, checkout c599236 at
`runtime/checkouts/shared-split-width-v1`, exited0 after3473.46s (57.89min).
All daily/T7/T30 persistent-cache lookups hit; zero backend compiles occurred.
The unchanged two changing-data technical optimizer updates passed. These are
engineering checks, not a new scientific learning/rollout acceptance.

Remote artifacts are under `runtime/outputs/diagnostics/`:

- `shared-split-profile-c599236.json`, `.updates.json`, `.log`, `.assets/`;
- `shared-split-profile-c599236.summary.json`, offline test02 analysis;
- `shared-split-serialized-load-20260925.json`, `.log`, free GPU0 launcher49083.

Compact JSON results are copied under local `outputs/`. The asset directory
has10files occupying about4.4GB; retain the executable/trace evidence. Do not
repeat the large HLO export for ordinary training or logging changes.

## Measured execution

Fixed-weight, preloaded-host-data measurements use one warmup and three timed
replays. They exclude dataset preparation and do not count as learned samples.
Daily64 concatenates real32-row artifacts742/275; date overlap is not audited.

| Component | Median seconds |
| --- | ---: |
| Daily micro4 value/gradient |0.51953|
| Daily32, accumulated micro4 |4.26904|
| Daily64, accumulated micro4 |8.60855|
| One T7 loss/gradient |1.81306|
| One T30 loss/gradient |7.68763|

The second changing-data complete update was16.1053s: daily32 4.59187s,
T7 2.01645s,T30 8.11008s, optimizer0.02838s. Internal total14.7734s leaves
about1.33s outside that subtotal; attribution remains incomplete. Actual peak
allocator use11.962GiB is distinct from the25.39GiB reserved pool.

Daily64 takes2.017times daily32: unchanged micro4 accumulation does not improve
sample throughput. At7.43-7.50 daily rows/s, even daily-only arithmetic is about
13.4days per8,591,565 TRAIN row-count equivalent. Ideal two-card daily64 versus
T7+T30 overlap gives max(8.61,9.50)=9.50s, or14.76days per TRAIN row-count
equivalent, before communication/IO/other overhead. This is not measured dual
performance or a true epoch with verified unique coverage.

T7 and T30 take0.2590s and0.2563s per simulated day. This evidence is consistent
with roughly linear horizon work, not a quadratic-time rollout explosion.

## Startup and direct executable loading

Despite cache hits, complete daily/T7/T30 startup stages took345.45s,
571.97s and593.56s; all graphs were ready at1727.4s (28.79min, including
bootstrap). Saving executables and exporting huge HLO added about1527s
(25.45min) in this diagnostic. That export cost is not normal update time or
a new backend compile; it explains much of the57.89min total.

The separate deserialize-only probe49083 exited0. All three trusted saved
executables loaded and stayed resident in387.15s (6.45min):

| Component | File read seconds | Native deserialize/load seconds |
| --- | ---: | ---: |
| Daily |30.67|48.83|
| T7 |0.47|153.46|
| T30 |0.38|153.08|

This establishes an executable-loading route that skips model tracing and
lowering on the current JAX0.4.38/V100 environment. It does not yet establish
end-to-end training startup savings: model/data preparation is excluded,
and this probe did not execute the loaded functions. No numerical parity,
cross-device portability or universal cache compatibility is claimed.

## What the traces establish

One daily micro4 gradient generated62,993 GPU events,55,525 of them at most
10microseconds. Fused kernels account for53.09% and matrix multiplies42.51%
of summed device event time. T7 generated380,309 events,346,906 at most
10microseconds; fused kernels53.74%, matrix multiplies38.47%. This supports
investigating fragmented execution and small matrix work. It does not show
that all small operations can be eliminated or merged at equal memory cost.

Explicitly named daily GEMMs are FP32. This does not certify every operation
as FP32. Most source attribution is unresolved or points to generic shared
linear/FFN helpers: about93% of daily and90% of T7 device time. Do not turn
the small explicitly mapped retained-physics bucket into a claim that physical
processes cost almost nothing, or assign the whole unresolved bucket to one
decoder. CUPTI/optional profiler warnings occurred; GPU traces are present,
but use uninstrumented replay times above for throughput.

Optimized HLO contains907,503 nested instruction occurrences for daily,
1,022,653 forT7 and1,023,686 forT30. T7/T30 textual IR is about9.20GB each;
the executable files are about499/500MB, not9GB of GPU memory. Metadata
suffixes account for at most about1.5% of recurrent text, and constant
instruction lines about3.56MB. Very long tuple/type lists are present (up to
1.36MB per line). Removing source names or large constant values alone cannot
explain away this size. Exact tuple-byte share and compiler-pass time are not
measured. Similar T7/T30 graph sizes do not support30days of Python unrolling.

The analyzer now streams binary HLO lines, retains only traced symbol mappings,
and checks long tuple assignments. All three files parsed with zero recorded
unparsed assignments; six unit tests cover streaming, long tuples, source
ambiguity and saved-asset handling. Nested IR counts are not GPU kernel counts.

## Next bounded work

1. Connect saved executables to the existing technical-update runner at its
   compile factory, preserving argument/output trees and the current public
   losses. Run the same two changing-data updates and compare saved reference
   loss/gradient/update evidence. No new whole-model compile is needed for
   this reuse check. Measure total startup including preparation; preserve
   normal compile mode for genuinely changed graphs.
2. Use a bounded actual-helper experiment to reduce fragmented decoder
   execution and staged graph size while keeping all active coordinates,
   shared weights, masks and gradients. Check both gradients and runtime/
   memory before any whole-model compile. Do not repeat the rejected width768
   outer-map or dense-packing variants unchanged. A new micro8 graph or device
   layout needs its own explicit startup budget; it is not a free cache replay.
3. Use the implemented microbatch-to-dual bridge only after real-model device
   compatibility is checked. Compare equal64-row work, complete update time
   and card-hours. Two cards alone cannot resolve the measured full-pool scale.

No new paid task or500-update training was submitted. Both free GPU probes
and the offline CPU analysis are complete.
