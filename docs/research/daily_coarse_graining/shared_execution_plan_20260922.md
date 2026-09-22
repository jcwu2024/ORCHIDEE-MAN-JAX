# Shared training execution optimization before coverage expansion

The user approved performance optimization before more training on2026-09-22.
Current full shared update is29.90s median wall time on one V100. Components
are14.91s for32 sequential daily rows,2.62s T7 and11.04s T30. Prior architecture
used daily64/micro8 and dual-device partitioning (~12-13s serial,7.2-7.6s dual
update timer). Older long-run total cycles were slower than that update timer.
These optimizations have not been admitted for the consolidated shared model.

## First bounded comparison

Freeze `shared_execution_benchmark_v1.json` before execution. Use trained
shared update100 only as a diagnostic source, artifact742's same32 daily rows
and first T7/T30 window, unchanged losses/normalization/precision/native AD
and AdamW. Test original scalar sequential accumulation against micro2 and
micro4. Compute `grad(mean(vmap(single_day_loss)))`; never materialize a
microbatch-sized tree of per-example parameter gradients. Reduce hard counts
by sum and equal-size microgradients by the complete sample mean. Clip/update
once after adding the unchanged T7/T30 gradients.

Preserve the original daily gradient and complete optimizer proposal before
another large compilation. Require finite values, matching physical counts,
scalar/daily full-gradient/full-parameter/full-AdamW parity with declared
rtol1e-5/atol1e-6. Report relativeL2 and failed coordinates; do not relax
tolerances to produce a pass. One warmup and three complete joint replays at
the same source checkpoint, no retained learning updates. Report compile time,
component/update timing and allocator peak separately. Identical preloaded
data in both paths means these timings do not establish end-to-end data IO
throughput. Compile memory ceiling24GiB; skip resource-limited variants without
running them. Two larger sizes are sufficient for the first decision; no
automatic micro8/16 sweep.

Free idle gln01 GPU,180min main ceiling plus a5min tiny deployed runtime check
required for the new vmapped gradient execution. Exercise actual launcher args
and tiny helper tests in deployed JAX before the real-model run. No paid job,
Teacher regeneration, architecture reduction or new science loss. Do not rerun
the prior expensive JVP suite for unchanged recurrent objectives.

## Following step

After measured microbatch parity/memory/throughput, select the fastest admitted
daily execution (or retain scalar if none improves). Then measure the unchanged
joint update with daily work on one GPU and T7/T30 on another, joining gradients
before one AdamW update and synchronizing the next parameters. Check changing
data and subsequent updates, not only identical-weight replays; report GPU-hour
cost as well as wall time. Reuse the existing dual-device executor when its
callback boundary fits, without importing its old network/objective. This
second comparison has not been launched or claimed accepted.

Resume larger-coverage learning only after selecting the measured execution
policy. The100-update model's science conclusions remain unchanged.

## Local implementation checks

The new helper and existing runner tests pass17 tests. They compare masked
scalar/vmapped gradients, additive physical counts, clipped AdamW and two
successive changing-data joint updates between original32-row execution and
the full-batch adapter. Ruff passes. The benchmark delegates joint combination
and the single AdamW step to the same `execute_learning_update` implementation;
its candidate daily callback returns the whole32-row mean exactly once.
These small tests do not establish real-model memory or performance.

## Deployed checks and launch

Immutable0507e1b deployed to`runtime/checkouts/shared-execution-0507e1b`.
Actual CPU/GPU target-shell expansion and shell syntax pass. Free test02
loaded the869MiB source100 checkpoint and verified its method/cursor in38.8s,
exit0. The actual GPU Python environment (CPU backend for tiny fixtures)
passes10 microbatch tests in16.19s, including two successive changing-data
joint updates. An initial combined remote test invocation returned exit1 with
no captured diagnostics; it is not counted as a pass. The logged focused rerun
passes; disabling pytest's cache emits a harmless cache_dir config warning.

Both free GPUs were idle immediately before launching GPU1 wrapper38944.
It runs the separate5min-bounded synthetic GPU runtime check followed by the
180min-bounded real-model benchmark. Report/log prefix:
`runtime/outputs/diagnostics/shared-execution-0507e1b-gpu`.
The reference snapshot will be a single `.reference.pkl` under that prefix;
no candidate proposals become training checkpoints. No paid job.
The synthetic GPU check PASSED in34.87s on deployed JAX0.4.38. Main benchmark
has verified the checkpoint and entered actual runtime loading.

Next collect full-gradient/optimizer parity, compiled memory and complete
update timings. A faster admitted microbatch is only fixed-data execution
evidence; changing-data dual placement and sustained learning remain subsequent
steps. Do not duplicate this benchmark while38944 is running.
