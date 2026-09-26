# Equal-two-window recurrent batching diagnostic

Status: local implementation preparation, free resources only. This tests
execution efficiency for a fixed two-window workload; it does not change
the production sampling mix, optimizer batch size, architecture or precision.

## Hypothesis and fixed work

The passed combined T7 profile still has299,552GPU events,268,893shorter
than or equal to10us. Independent trajectory batching may amortize these
operations and improve device utilization. Short events are not necessarily
pure launch overhead, and this profile does not guarantee a batching gain.

Use existing artifact742/window0 and artifact275/window0 T7 cases from the
compact input bundle, the same fresh update0 and the same admitted combined
backbone/decoder execution. The reference executes BOTH cases serially and
averages their gradients/losses; the candidate processes those SAME two cases
using native vmap, differentiating the mean scalar objective outside vmap.
Both cases retain their own consecutive daily scan, initial state, masks,
parameters/static context and full native checkpointed differentiation.
Do not parallelize successive dates of one trajectory or detach feedback.

Average loss/flux/stock auxiliary terms. Sum physical hard counters and
declared_mask_change; no averaging away a failed trajectory. Reuse the
existing daily microbatch aggregation policy, which already covers these
paths. No custom reverse implementation or independent observation-cotangent
construction. No Teacher preparation or additional dataset files.

## Admission and resources

First validate actual case tree/shape/dtype compatibility and context/adapter
loading on free CPU. Local small native-scan tests compare parameter and
initial-state gradients for distinct cases with masks. The actual GPU test
checks loss/auxiliary/all parameter-gradient coordinates at unchanged
rtol1e-5/atol1e-6, exact zero physical hard counters, finite results and
compiler estimated live memory<=24GiB. Actual independent initial-state
cotangent parity is not implied by parameter-only saved executable evidence.

Save the serial two-case aggregate before compiling. Save candidate executable
and full result before numerical comparisons; reuse the completed reference
after interruption. Warm serial timing includes both executions and device
gradient aggregation, with the aggregation helper prewarmed. Candidate timing
includes the single two-case loss/gradient call. Report windows/s, warm wall,
compile/lower/load times and memory separately; do not compare one window's
latency to two windows' throughput as if work were unchanged.

One freeGPU1,60minute main ceiling after a3minute runtime check. No paid
allocation. T7 is the first screen; do not extrapolate memory linearly or
launch T30/micro4/large sweeps before this result. A larger production batch
needs a declared sampling/update strategy and learning validation. A gain
here is not yet a complete daily32+T7+T30 update speedup or the10x goal.
