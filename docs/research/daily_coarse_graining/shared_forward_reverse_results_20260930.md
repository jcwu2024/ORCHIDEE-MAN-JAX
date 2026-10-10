# Original-Mix Forward and Marginal AD Diagnosis

Date: 2026-09-30. Fixed update-0 weights, saved artifact 742, one free
`gln01` V100, immutable `c599236` checkout. This is execution research, not
training or scientific acceptance.

## Matched Component Measurement

The probe validated the original profile, compact real-input bundle, prepared
context and all three current admission reports. It loaded the same admitted
candidate value-and-gradient executables used by the successful original-mix
full proposal. It lowered a forward-only graph for each unchanged public
objective under the admitted expert128 decoder and backbone source-reuse
patches. The patch files in the daily/recurrent overlays have matching SHA256s.
No model-gradient graph was compiled. The complete returned gradient of each
saved evaluator was synchronized; the new forward result was compared to that
evaluator's loss and entire auxiliary tree at `rtol=1e-5, atol=1e-6` before
timing. All three comparisons passed with zero failed coordinates and all
hard physical counts zero. Three interleaved warm pairs followed per component.

| Component | Forward-only median | Combined median | Incremental AD premium | Premium / combined | Lower / backend compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daily micro4 | 0.079142 s | 0.378674 s | 0.299532 s | 79.10% | 69.638 / 182.185 s |
| T7 | 0.275584 s | 1.588739 s | 1.313155 s | 82.65% | 34.796 / 168.108 s |
| T30 | 1.196103 s | 6.714903 s | 5.518800 s | 82.19% | 33.838 / 168.962 s |

The difference is **incremental native AD cost**, not a timed pure reverse
pass. The forward-only and combined executables are separately compiled and
can use different fusion, memory, and scheduling. In T7/T30, the native
`jax.checkpoint` transition may recompute forward operations during reverse AD.
The combined median includes forward, physics, loss and reverse AD. The
forward-only median also includes both neural and retained physical execution;
neither measurement isolates their shares.

## Original-Mix Arithmetic

For eight daily micro4 calls plus one T7 and one T30, the sum of these
separately measured component medians is **2.104821 s forward-only** and
**11.333034 s combined**, a **9.228212 s marginal AD premium** (18.57% /
81.43% of that summed evaluator estimate). This is arithmetic across separate
processes, not an observed decomposition of one update. The prior synchronized
tree proposal measured **11.483 s** in ten evaluator calls and **0.331 s**
outside them, versus **11.814 s** total. Its evaluator sum is 1.3% greater
than the new summed component medians. The normal opt-in tree proposal median
was **11.547 s**; synchronization changes its timing.

For scale only, combining the new 11.333034 s evaluator estimate with the
earlier 0.331 s outside time gives an illustrative 11.664034 s baseline.
Holding those components fixed, halving the marginal AD premium would give
1.65x total speedup; making it four times faster gives 2.46x. Eliminating it
entirely gives an optimistic 4.79x ceiling. Halving forward-only work gives
1.10x; eliminating it gives an optimistic 1.22x ceiling. These cross-run
Amdahl calculations are conditional illustrations, not observed speedups,
hardware bounds, or claims that any scientific work can be removed. A 10x
unchanged-work update cannot come from eliminating either measured partition
alone under this illustrative composition.

## Opportunity and Limits

The shared native AD path is the larger measured opportunity across all three
components. The next function-preserving experiments should inspect native
checkpoint placement and repeated forward work during T7/T30 reverse AD,
then dense and fused operations in the shared parameter-gradient path. Preserve
complete parameter and state gradients, the original loss, row counts and
physical masks, and measure memory against the 24 GiB gate. Kernel fusion or
batching could reduce the many short operations, but prior traces do not show
that all short-kernel time is removable launch overhead. Forward-path reuse
can also speed reverse AD or rematerialized forward work; the 1.22x ceiling
above applies only when **forward-only time** changes and AD cost stays fixed.
The completed FP32 union micro4 comparison saved only 1.337% warm wall at
higher memory, so that specific reuse path is not evidence for a large gain.
Retained physical execution could be material inside either partition; its
fraction is **unknown**. A neural/physical split needs a new source-aligned
measurement with numerical and gradient parity, not kernel-name attribution.

The three successful process intervals (launch to final report mtime, 1 s
timestamp resolution) total about 916 s, or **0.2544 one-V100 wall-exposure
card-hours**. The separate deployed-runtime preflight passed in 82.896 s
(0.0230 card-hours). Two initial launches failed before lowering and are not
included in that subtotal; neither changed scientific assets. Paid cost is
CNY 0, no Slurm job or optimizer update. Each successful process used one GPU;
their approximate launch-to-report intervals were 272 s daily, 307 s T7 and
337 s T30. These include startup, deserialization, lowering, compilation,
parity checks and timed calls; startup alone was not isolated.
Process-cumulative device peaks were 11.066 GiB daily, 3.750 GiB T7 and
3.757 GiB T30, below the declared 24 GiB limit. These are not isolated
forward-only versus combined peaks. Startup and compilation are excluded from
the warm timings. The GPU was empty after all probes.

Evidence: [daily](../../../outputs/diagnostics/shared-forward-reverse-20260930-daily.json),
[T7](../../../outputs/diagnostics/shared-forward-reverse-20260930-T7.json),
[T30](../../../outputs/diagnostics/shared-forward-reverse-20260930-T30.json),
[runtime preflight](../../../outputs/diagnostics/shared-forward-reverse-20260930-preflight.json),
[probe](../../../scripts/hpc/probe_shared_forward_reverse.py), and
[launcher](../../../scripts/hpc/run_shared_forward_reverse.sh). The synthetic
runtime preflight also remains under `runtime/outputs/diagnostics/` on
Explore1000. Local `test_shared_forward_reverse.py` passed (2 tests),
Python syntax and target Bash syntax/actual `--dry-run` expansion passed.
