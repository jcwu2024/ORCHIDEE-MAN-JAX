# Real-model dual-device update: numerical and timing acceptance

Job15230086 COMPLETED0:0 in28m45s onibc13b03n01,2026-09-16 15:54:50–16:23:35
server time. Immutableb2e1562;2V10032/8allocatedCPUs/64GiB, no requeue or retry.
Python elapsed1653.9832s. Estimated additive charge1725/3600*4.96=2.38yuan,
below the approved7.44yuan cap; this is an estimate, not a billing receipt.

The real-model complete update passes existing numerical/physical admission
and is1.7426x faster on two devices in this bounded fixed-data benchmark.
Warm update time falls42.61%; estimated charge for equal update count increases
3.13% assuming8 CPUs for either run and additive CPU/GPU billing. This supports
integration into continuous training. It does not yet certify rotating-data
throughput, recovery, scientific learning, or long-term model acceptance.

## Evidence and unchanged method

- Local report: `outputs/mixed-learning-dual-paid-15230086.json`.
- Remote report/log: runtime/outputs/diagnostics/
  `mixed-learning-dual-paid-15230086.{json,log}`.
- Complete serial reference retained in matching `.serial-reference.pkl`.
- Protocol: `dual_device_paid_validation_20260916.md`.
- All109,464,375 parameters; source checkpoint update200 from
  `runtime/outputs/training/mixed-learning-v1/report.update-000200.checkpoint.pkl`.
  This is an execution diagnostic asset, not initialization of a new final model.
- Same weights/data, daily64/micro8 +.5T7+.5T30, unchanged loss/AD/precision/AdamW.
  GPU0 daily accumulation;GPU1 T7 thenT30; merge/clip/update once onGPU0.
  Reads, checks and updated-weight transfer toGPU1 are inside complete timing.
- All optimizer proposals were discarded: persistent_parameter_updates=0;
  rotating_data_throughput_validated=false. No new learning result.

## Numerical result

| Compared tree | Error L2 / reference L2 | Coordinates outside tolerance | Result |
| --- | ---: | ---: | --- |
| Full mixed gradient |8.4364e-10|0|pass|
| Updated parameters |7.1235e-11|0|pass|
| AdamW state |4.6922e-13|0|pass|

Physical counters agree and all required hard counts are zero. Loss, norms and
clip scalar comparison passed; initial maximum scalar relative error1.2206e-10.
All three warmed repetitions also pass scalar/physical parity, maximum relative
scalar error3.7309e-10. Small floating-point differences exist; this is numerical
acceptance at unchanged tolerances, not a claim of bitwise identity.

## Time, balance and cost

Three interleaved warmed repetitions, same process/node and workload:

| Mode | Individual seconds | Median seconds |
| --- | --- | ---: |
| Single-GPU mixed |12.9282,12.5849,12.5710|12.5849|
| Dual-GPU mixed |7.1734,7.2221,7.3187|7.2221|
| Single-GPU daily-only |6.5291,6.5365,6.6039|6.5365|

Speedup1.742562; dual/single elapsed ratio0.573868. GPU-hour ratio1.147735,
so GPU-only cost increases14.77%. With unchanged8 allocated CPUs, rates are
2.76yuan/h single and4.96yuan/h dual; equal-update total charge ratio1.031298.
This3.13% estimate is near equal cost and depends on billing and sustained
throughput; three fixed-data repetitions do not establish an exact long-run bill.

Dual daily branch6.761–6.888s; recurrent branch6.163–6.528s. Work is reasonably
balanced: neither GPU spends most of the step waiting for the other. Merge,
checks and optimizer take0.332–0.364s; weight sync0.078–0.086s. Parallel work
also runs a little slower per branch than the serial reference, so ideal2x
speedup is not achieved. No new optimization prerequisite is proposed.

For intuition only,1000 warmed fixed-data mixed updates extrapolate from3.496h
to2.006h and9.65yuan to9.95yuan. Excludes startup, rotating-data effects,
monitors/checkpoints and queueing. Do not quote this as a complete training budget.

Raw compile/lower/load call times: dailyGPU0=104.04s,T7GPU0=255.40s,
T30GPU0=270.47s,T7GPU1=268.85s,T30GPU1=261.70s. Startup and first execution
dominated this short diagnostic's28m45s allocation; these are not warm step
times and must still be budgeted for future jobs.

Allocator-reported peak live device memory:GPU0=15.662GiB,GPU1=2.646GiB.
Both reserved pools were26.978GiB. These are different quantities; GPU1's
low memory peak does not imply a low compute load. Process host peak31.775GiB.

## Next gate

The repeated helper/device-cache failure is closed for this real-model
same-update path. Do not rerun the unchanged profiler or historical AD audits.
Keep the common helper-cache scope and Slurm device allocation handling.

Next integrate the admitted two-device execution into the common training
runner, retaining one authoritative parameter/AdamW state onGPU0 and explicit
peer synchronization. A bounded integration must use distinct successive
batches/windows, actually retain at least two updates, and compare against
serial execution from the identical state. Check that the second update uses
the first update's synchronized weights; exercise checkpoint save/load and
continued update, phase/recurrent cursor and physical checks. Report measured
rotating-data time/memory and recovery parity. Preserve both learning pilot
checkpoints; no training schedule/architecture/loss change is implied.

Any subsequent paid integration or learning job needs its own declared command,
resources/time/cost approval. No new job was submitted while collecting this
result. No active project computation remains after15230086 completed.
