# Split training efficiency: profile and microbatch dual bridge

Status: profile31906 and direct-load49083 completed successfully. Results and
current next steps: `shared_split_profile_results_20260925.md`. Running/next
measurement text below records the original launch, not current task state.
The user prioritizes compiler startup, sample throughput and full-TRAIN scale
before long training. The prior500-update launch is on hold. Preserve the
admitted64,884,481-parameter architecture, physical constraints, objective,
native checkpointed AD and independent formal initialization.

## Completed work

`probe_shared_split_profile.py` invokes the unchanged technical learning path
and requires actual persistent-cache hits for daily micro4,T7,T30. The scoped
compiler observer stops before a backend compile on any miss. It saves each
loaded executable with input/output trees and compressed optimized HLO, then
runs the same two changing-data technical updates. Three fixed-weight timed
replays plus one warmup measure micro4,daily32,daily64,T7,T30. The64-row batch
concatenates the two real32-row artifacts742/275, without padding; overlap of
dates between artifacts is not audited and benchmark replays are not learning.
Data are already loaded on the host, so these component results exclude
dataset IO. Trace one micro4 reverse and oneT7 reverse, not all30days.

This diagnostic changes no model or gradient calculation. Seven local tests
pass; Ruff and diff checks pass. A tiny actual JAX0.4.38 container test ontest02
serializes/reloads a native reverse executable with exactly equal results.
The target-shell launcher dry-run passes. Unchanged physical/AD preflight
evidence is reused. Saved executable files are trusted internal artifacts,
not portable model checkpoints or proof of device-placement compatibility.

`probe_shared_dual.make_microbatch_daily_branch` now connects the existing
exact daily mean accumulator to `SharedDualExecutor`. It supports declared
row and microbatch counts without another division of the daily gradient;
placement, per-microbatch physical/finite checks and final gradient checks
remain. Legacy/default runner behavior is unchanged. Local shared-dual,
microbatch-bridge and daily-accumulator tests pass13/13. A new two-host-device
test compares64rows/micro4, both recurrent components, full gradients,
AdamW/parameters and peer synchronization across two changing updates, plus
nonfinite/hard-count rejection. The same integration passes in22.01s using
the actual JAX0.4.38 GPU Python container with two CPU devices ontest02.
This is deployed-runtime plumbing evidence, not real-model dual-GPU admission.
The small source overlay is runtime/tools/shared-split-dual-microbatch-20260924;
the running model checkout was not changed for this test.

Independent Sol/high source audit and offline HLO/trace summarizer are in
`shared_split_compile_audit_20260924.md` and
`scripts/dev/summarize_shared_split_profile.py` (four local tests pass).
Luna/high throughput accounting and tests are in
`shared_split_throughput_scale_20260924.md` and
`scripts/dev/estimate_shared_training_scale.py` (four local tests pass).
The lead reviewed outputs, corrected arithmetic/attribution wording and
validated integration. No new scientific conclusion follows from these tests.

## Active free profile

- Launcher PID31906 on free gln01 GPU1, started22:50CST September24.
- Code c599236, runtime/checkouts/shared-split-width-v1; keep it immutable.
- At launch both GPUs0MiB/0%; host load0.00/0.01/0.14.
- Maximum90min, command runtime/transfer/launch_shared_split_profile.sh.
- Report/log runtime/outputs/diagnostics/shared-split-profile-c599236.{json,log}.
- Separate original-run report .updates.json; execution assets .assets/.
- At23:02CST daily hit complete, whole stage345.4s; T7 startup ongoing.
- No paid allocation, no model capacity or loss change.

Check wrapper with `ps -p 31906 -o pid,etime,pcpu,comm`; completion authority
is the final report and run_exit_status in the log, not PID absence alone.

## Next decision

Collect all cache hits, repeated component timings and memory; summarize HLO
and kernel/source categories offline. A large ambiguous source bucket must
remain ambiguous. Instrumented trace duration is not the warm speed metric.
Use actual daily32/64 scaling to plan a same64-sample serial/dual comparison;
the microbatch bridge alone does not admit that experiment. Budget any new
device-bound or micro8 graph separately because current single-device caches
do not prove compatibility. Do not repeat cold compilation merely for profiling.

Choose the next execution optimization from measured dominant work. Existing
width768 toy outer-map and dense-packing results are preserved; do not retry
those identical candidates without new evidence. A training-cadence/staged
schedule changes the optimization method and needs an explicit reproducible
comparison, not an equivalent-execution speedup claim. Scale every proposal
against8,591,565 TRAIN rows and card-hours, not only milliseconds per step.
