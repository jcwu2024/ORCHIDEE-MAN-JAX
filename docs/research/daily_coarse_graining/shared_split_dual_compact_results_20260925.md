# Compact saved-executable serial/dual results

Status: completed diagnostic, September 25, 2026. Free gln01 launcher80665
exited0. Immutable model c599236 and compact replay from the original fresh
update0 checkpoint; no formal training or scientific acceptance.

## Complete-update measurements

Each row compares identical inputs, weights and objectives between serial and
dual execution. Three interleaved, synchronized proposals per mode include
parameter/gradient transport, physical checks, gradient merge and AdamW.

| Daily rows + one T7 + one T30 | One V100 | Two V100s | Speedup | GPU-hour ratio | Estimated charge ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 14.851 s | 11.193 s | 1.327x | 1.507 | 1.354 |
| 64 | 20.391 s | 11.336 s | 1.799x | 1.112 | 0.999 |

Daily throughput is2.155/2.859 rows/s for serial/dual32 and3.139/5.646 for
serial/dual64. Charge estimates assume8allocated CPU cores per run,
0.07yuan/core-hour and2.20yuan/GPU-hour from deployment-explore1000.md;
the free experiment incurred no paid allocation. Startup is excluded from
these warm ratios. These are three-repeat measurements on one node, not
confidence intervals or production continuous-training measurements.

Dual64 versus serial32 gives about2.62x daily rows/s, but changes the daily
sampling count relative to unchanged T7/T30. It is not a same-method2.62x
speedup. Both32 and64 remain far from a tenfold throughput gain. Dual32
reduces wall time24.6% while increasing estimated equal-work charge35.4%.
Dual64 balances these two branches better, with nearly equal estimated
charge to serial64. Neither row establishes learning quality at that batch.

## Correctness and startup

Two real changing-data updates (artifacts742/275) pass full64,884,481-coordinate
gradient and parameter comparisons and129,768,963-coordinate AdamW comparisons.
Every checked coordinate passes unchanged rtol1e-5/atol1e-6. Largest gradient
absolute difference is5.96e-8. Final parameters/full AdamW also match the
previous trusted replay checkpoint; technical loss/scalar records pass.
Fixed-weight32/64 proposals pass the same full-tree comparisons. This is
bounded execution equivalence, not long-run scientific or recovery acceptance.

The GPU launcher began16:13:54CST; measured probe completed1254.68s after its
internal timer, with exit0 confirmed16:37. The one-time free CPU preparation
previously took201.40s and produced one~50MiB input bundle. Parent loaded
daily/T7/T30 and worker loaded T7/T30 with no model-gradient trace/compile.
Graphs were ready748.5s after the internal timer (which excludes some imports).
Worker T7 deserialization took161.63wall/160.87CPU seconds and T30 took
161.60wall/161.55CPU seconds: this phase is real CPU work, not just IO wait.
This diagnostic deliberately loads recurrent executables in both processes
to compare serial with dual; its startup is not a production dual minimum.

## Decision

Keep dual execution as an available measured route, not the main10x solution.
Next evaluate same-function common-source backbone K/V reuse and whole-expert
decoder scheduling. Preserve network capacity, all active coordinates, masks,
losses and native AD. Avoid another full-model compilation solely for tiny
helper gains. Use a partitioned native-AD reference for decoder correctness;
its summed timing is not the original monolithic performance baseline.
Hardware/precision changes remain separately declared routes with resource
cost and numerical/learning validation. No new paid job is authorized here.

Evidence: remote runtime/outputs/diagnostics/shared-split-dual-compact-20260925
(.json/.log/.updates.json and diagnostic checkpoints), local
outputs/shared-split-dual-compact-20260925.json. Original measured reports
are preserved; no Teacher data was regenerated.
