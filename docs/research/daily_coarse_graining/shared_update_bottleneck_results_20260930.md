# Original-Mix Complete-Update Bottleneck Diagnosis

Date: 2026-09-30. This is a fixed-weight, discarded-proposal execution
diagnostic, not training throughput or scientific acceptance.

## Workload and result

The saved field model executables and inputs ran on one free `gln01` V100:
daily32 in eight microbatches of four, one T7 and one T30, followed by the
unchanged mixed gradient and full AdamW proposal. Three interleaved warm pairs
used default per-leaf and opt-in whole-tree daily slicing. No new model-gradient
graph compiled and no proposal was retained. Complete-gradient, proposed-
parameter, full AdamW, auxiliary, scalar and physical gates passed; all 39
hard physical counts per proposal were zero.

| Measurement | Seconds | Interpretation |
| --- | ---: | --- |
| Normal full proposal, default leaf slicing median | 12.606 | Current default execution |
| Normal full proposal, opt-in tree slicing median | 11.547 | Same data and model; faster daily slicing |
| Separately synchronized tree proposal | 11.814 | Diagnostic synchronization changes execution; 2.31% slower than normal tree median |
| Ten synchronized model evaluator calls | 11.483 | 97.20% of that diagnostic proposal |
| Outside those evaluator calls | 0.331 | 2.80% of that diagnostic proposal |

The ten calls are eight daily micro4 evaluations plus T7 and T30. Each timed
call synchronizes its input before timing and its full returned value, auxiliary
mapping and gradient afterward. Its time includes model forward, loss, native
reverse AD, retained physical processes, and execution/dispatch inside the
evaluator. The outside portion includes daily slicing and accumulation, mixed
gradient work, physical/finite gates, AdamW and host scheduling. It is not an
independent measurement of pure Python overhead or of neural versus physical
time. The 97.20% share applies to the separately synchronized, opt-in tree
diagnostic; it is not the exact composition of default leaf-slicing training.
The default path was 1.059 seconds slower than the tree path in this run.

In the normal, untraced tree proposals, disjoint runner timers assign median
wall fractions of about 27.6% to daily32, 13.7% to T7 and 58.3% to T30.
These are workload priorities, not per-day cost comparisons. The optimizer is
about 0.17%; branch timers do not account for every host cost.

## Conditional speed limits

These Amdahl examples hold all other measured work fixed and assume the named
time can disappear or shrink without side effects. They are arithmetic
illustrations, not hardware bounds or achievable speed forecasts.

| Hypothetical change | Full-proposal speedup |
| --- | ---: |
| Remove all 0.331 s outside synchronized evaluator calls | 1.029x |
| Remove the entire measured daily32 branch | 1.38x |
| Remove the entire measured T7 branch | 1.16x |
| Remove the entire measured T30 branch | 2.40x |
| Make all evaluator calls 2x / 4x / 8x faster | 1.95x / 3.69x / 6.69x |

With the synchronized tree fraction fixed at 97.20%, a 10x full-proposal
speedup would require evaluator calls to become about 13.5x faster while the
outside time stays fixed. Removing any single branch would also remove its
scientific work, so those branch rows only show how much time is at stake.

## GPU trace and limits

One separate instrumented tree proposal took 12.677 seconds, 1.130 seconds
longer than the normal tree median. CUPTI warned that the per-thread activity
buffer could not be used and trace overhead could be large. The 685,739
recorded GPU events span only 46.54% of the full proposal window from first to
last event, and 32.40% carry HLO-op metadata. Their interval union is 5.139
seconds. The remaining wall time is **not** a measured CPU-wait fraction or
removable overhead: buffering, synchronization, host work, contention and
profiler effects are unresolved. Kernel labels do not distinguish neural from
physical work, or forward from reverse AD. The profiler's 9.79% wall
perturbation and partial metadata preclude a reliable GPU-occupancy ceiling.

The first free-node attempt failed before any full proposal when a diagnostic
annotation requested the wrong optimizer key. The wrapper was corrected,
covered by a regression test, and rerun in a separate path. The successful
retry passed 17 focused local tests, `py_compile`, shell syntax/argument
expansion, the target CPU asset check and `git diff --check` before execution.
Both free-node processes ended. Paid cost: CNY 0; GPU count: one V100; no
Slurm allocation or paid card-hours. The successful run's saved warm proposals
used 72.536 V100 card-seconds (0.02015 card-hours), which is not the whole
diagnostic process time.

Evidence: [trace analysis](../../../outputs/diagnostics/shared-update-bottleneck-20260930-retry1-analysis.json),
[preparation and launch record](../../../outputs/diagnostics/shared-update-bottleneck-preparation-20260930.json),
[probe](../../../scripts/hpc/probe_shared_update_tree_slice.py) and
[trace analyzer](../../../scripts/hpc/analyze_shared_update_trace.py). The
complete proposal JSON, log and raw trace remain under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-update-bottleneck-20260930-retry1*`;
the deployed overlay is
`runtime/transfer/shared-update-bottleneck-20260930-retry1`. The immutable
model checkout remains `runtime/checkouts/shared-split-width-v1` at `c599236`.

Next, investigate evaluator-internal costs shared by daily, T7 and T30, and
recurrent-specific scheduling/checkpoint costs, while preserving row counts,
loss mix, full parameter/state gradients and physical gates. No architecture,
precision, sampling, training or default-slicing change follows from this
diagnosis.

Subsequent same-input, candidate-matched forward-only comparisons are in the
[forward/AD result](shared_forward_reverse_results_20260930.md). They estimate
the incremental cost of obtaining native gradients; the partial GPU trace in
this report still does not identify pure reverse, neural or physical kernels.
