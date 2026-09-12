# Broader one-day data: four-sample batching/resource probe

Freeze before execution. No optimizer or training updates. Use the current
fresh-start v3 model/loss/scales and seed20260911, all109,464,375 parameters.
Read the first four rows of the accepted catalogue's first32-row training
batch (not the excluded monitor):

| Artifact,row | Point | Year | Teacher row |
| --- | --- | --- | --- |
| 1400,14 | 279.0-095.0 | 1994 | 71 |
| 1066,10 | 249.0-113.0 | 2000 | 163 |
| 1678,31 | 009.0-087.0 | 1984 | 207 |
| 2216,2 | 119.0-107.0 | 1993 | 74 |

Materialize these exact rows using the bounded one-day cache. Compare one
vmapped four-sample mean-loss gradient with the arithmetic mean of the four
single-sample gradients at identical parameters. Preserve the v3 per-sample
loss reduction; do not silently replace it with another batch-global bias
formula. Check finite losses/gradients and every existing physical hard count.
Use per-coordinate and per-leaf gradient comparison atol1e-6,rtol1e-4,
and report global norms. Record cold/hot
evaluation times and actual active/pool memory separately. These shared-node
timings are planning evidence, not formal throughput acceptance.

Compile memory is inspected before execution; use18GiB for this probe so other
live arrays fit the current20.63GiB allocator pool. CPU abstract shape checks
precede one free GPU for at most45minutes. This measures batching/IO/resource
feasibility on broader inputs and does not replace FD validation for a future
training protocol. No v6 regeneration, v5/validation/test reads or paid job.

After resource results, freeze a larger persistent run's batch, sampling,
update count, monitoring/stopping and budget. Do not claim the63,707 available
training days have already been consumed or that this resource probe trained
a model.

## Execution record

The catalogue-selection/batch-gradient/cache/sampling tests pass10 cases.
Real CPU abstract single/batch4 gradients pass at a49643f in662.2s on test03;
report `runtime/outputs/diagnostics/one-day-batch4-shapes-v1.json`.
The free GPU0 numerical/resource probe started2026-09-12 13:04 CST at4410c41,
launcher PID108717/Python108758,45-minute cap. Report and log are
`runtime/outputs/diagnostics/one-day-batch4-v1.json` and `.log`.
The probe completed in2144.28s and passed: every physical hard count is zero,
all109,464,375 parameter coordinates agree at the frozen tolerances, and L2
gradient difference is6.61e-7. Actual peak active JAX memory is7.85GiB versus
20.63GiB allocator pool. Hot batch4 evaluation takes0.400s; the four single
evaluations total0.930s including a cold first call. Batch4/single gradient
compilations take957.90/953.13s. This admits microbatch4 for a complete-loop
technical smoke; no optimizer or training has run in this resource probe.
