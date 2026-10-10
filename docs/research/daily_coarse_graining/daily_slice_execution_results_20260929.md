# Daily slice execution comparison

Paid diagnostic job **15469919** completed with exit code `0:0` on `ibc13b03n04` in **3:25**. The comparison isolates daily batch slicing with the admitted field executable; it does not measure a complete optimizer update, T7/T30, training, or the 10x target.

## Workload and gates

- Artifact742 supplies the 32-row workload; artifacts742 and275 are concatenated in their saved order for64 rows. The field executable is `shared-combined-real-daily-20260925.candidate.executable.pkl`. Its four-row original field replay from job15461046 verified the exact original rows0-3 before timing.
- The baseline uses the existing per-leaf Python pytree slice. The candidate uses one cached JIT pytree dynamic slicer per input shape, with dynamic row start and the same microbatch size4.
- Both routes preserve the 590-leaf daily pytree, scalar loss, auxiliary, every one of 64,884,481 parameter-gradient coordinates, and unchanged per-microbatch finite and physical gates. All six interleaved comparisons pass `rtol=1e-5`, `atol=1e-6`; all11 hard counts are zero in each mode and batch.
- No model-gradient graph was compiled. The daily gradient executable and original-four reference were reused. Estimated evaluator live memory was11.34GB; observed active-buffer peaks were13.00GB at32 rows and13.55GB at64 rows, below the unchanged24GiB gate.

## Matched timing

Each workload had explicit leaf and tree warm-ups followed by three synchronized interleaved pairs. Reported medians cover warm execution only.

| Daily rows | Leaf median | Tree median | Leaf/tree ratio | Slicer compile, separate |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 5.010s | 3.158s | 1.587x | 4.118s |
| 64 | 10.055s | 6.315s | 1.592x | 3.686s |

This shows about1.59x lower warm daily-branch time for both batch sizes. The 3.69-4.12 second shape-specific slicer compile is separate, so these data do not show a cold first-call or full-update speedup. Startup and saved-reference loading took43.80s; total probe elapsed time was200.02s. The deployment used **0.05694 GPU card-hours** (`205/3600`) and estimated CNY0.16 at the allocation rates, within the approved CNY0.46 worst-case envelope.

The completed probe retained each workload's warm-up parity summary, timings, medians and hard counts. It overwrote pair-level error summaries on each `matched_pair` event, so the top-level maximum absolute difference `2.2352e-8` is only the last recorded daily64 pair3 result, not a maximum over all six pairs. The saved four-row truth-guard maximum was `4.4703e-8`. Every pair's tolerance assertion passed during execution. Future probe output now stores each pair's order, mode timings and parity-error summary in `runs`.

## Evidence and source

- [Compact measured summary](../../../outputs/diagnostics/shared-daily-slice-execution-paid-15469919.summary.json), [raw run JSON](../../../outputs/diagnostics/shared-daily-slice-execution-paid-15469919.json), and [scheduler log](../../../outputs/diagnostics/shared-daily-slice-15469919.log).
- Full scalar/auxiliary/gradient results remain in `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-daily-slice-execution.full.pkl` (about1.3GiB); they were not copied locally.
- [Accumulator and opt-in slicer](../../../scripts/hpc/shared_daily_microbatch.py), [daily probe](../../../scripts/hpc/probe_shared_daily_slice_execution.py), [CPU asset check](../../../scripts/hpc/probe_shared_daily_slice_assets_cpu.py), and [preparation evidence](../../../outputs/diagnostics/shared-daily-slice-execution-preparation-20260929.json).
- The CPU gate checked all590 leaves across8 slices of32 rows and16 slices of64 rows. The saved artifact742/275 bundle, original-four field replay identity and row hashes, serialized field-executable input treedef, and all64,884,481 field-gradient coordinates passed validation on `test01`.
- The launcher reused the completed job15461046 deployed-runtime preflight. At launch, node load was `0.00/0.01/0.05`, CPU affinity was `0-3,56-59`, and visible GPU0 reported0MiB and0% utilization with no listed compute PID.

This is a bounded daily execution result. It does not establish a cause for the historical branch timing gap, a full-update improvement, recurrent speed, scientific learning, or the 10x research goal.
