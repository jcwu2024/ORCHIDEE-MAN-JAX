# Shared execution benchmarks: compilation timeouts

Collected2026-09-22 at20:55 cluster time. Neither benchmark established a
microbatch or dual-device speedup. Both launchers exited124 from their own
bounded `timeout`, without a logged OOM or numerical exception. The partial
JSON files still say `running` because termination did not reach the Python
completion writer; the shell logs and scheduler are the terminal authority.

| Run | Terminal evidence | Last completed work | Missing result |
| --- | --- | --- | --- |
| Free38944,0507e1b | `run_exit_status=124`, process absent | daily/T7/T30 scalar reference compilation; entered reference measurement at10746.8s | reference snapshot, micro2/4, parity and timing |
| Paid15328832,8cce49e | Slurm FAILED124:0,02:51:39 | daily and T7 on GPU0; began T30 at9259.0s | completed serial reference, GPU1 recurrent compilation, dual parity/timing |

The paid allocation used2 GPUs/8 CPUs. At the approved additive rate4.96yuan/h,
elapsed-time estimated charge is14.19yuan, not an invoice. Limit was14.88yuan.
Synthetic runtime checks passed, but they are not real-model acceptance.
No retained training update was made and no candidate timing was obtained.

## Compilation evidence

| Graph | Previous100-update runner compilation, seconds | Free benchmark, seconds |
| --- | ---: | ---: |
| daily |306.682|2382.566|
| T7 |428.184|4035.262|
| T30 |435.657|4083.979|
| Total |1170.523|10501.807|

All five reported memory-size fields for each free reference executable match
the previous learning report exactly. Estimated compiled live memory remains
6.67GiB daily and3.28GiB recurrent; these are compiler estimates, not measured
process peaks. This supports investigating compilation/cache execution rather
than attributing this exit to a larger network or GPU-memory failure. It does
not prove identical HLO or successful persistent-cache reuse.

The free probe calls the same `compile_gradient_evaluators` helper as learning
and inherits its launcher/cache directory. The dual probe explicitly adds the
existing device-specific cache namespace through `device_helper_cache`.
Consequently the same directory alone is not evidence that both probes reuse
the original serialized executables. The cause of the free probe's approximately
9x aggregate compile-time increase is still under investigation.

Independent Sol/high source review found no changes between4408daf and0507e1b
to the evaluator helper, shared runtime, process head or recurrent objective;
artifact742 also starts the learning schedule. Dual compilation adds a nonempty
custom cache-key suffix absent from learning, so it cannot directly hit the
original unscoped learning entries. This is intentional device-placement
isolation, not a reason to remove the scope without placement evidence.
Deployed JAX normally strips debug metadata from the key; different checkout
paths alone are not an established explanation for the free slowdown.

The user also requested choosing an idle paid node before future compilation
probes. A later same-evening scheduler snapshot showed the previous paid node
ibc13b02n03 at82/112 allocated CPUs and CPULoad81.59, whileibc13b03n03 was
IDLE with0 allocated CPUs and CPULoad0.01; several other nodes were IDLE.
This is a reason to record and control host load in the next probe, not proof
of load during15328832. Its own accounting records8CPUs/2GPUs. The bounded
node/time accounting query returned only that job, so other-job overlap or
same-GPU sharing during its lifetime has not been established.

## Next action

Do not repeat either complete paid/free benchmark unchanged merely with a
longer timeout. First inspect compiler-cache behavior and the differing entry
boundaries using a bounded diagnostic, then preserve individual completed
stages before later compilations. The smallest next diagnostic is daily-only
lowering without calling `.compile()`, comparing actual cache keys and entry
availability for4408daf and0507e1b under identical device/environment settings.
This separates key drift from absent/unreadable entries without paying for
another full cold compile. Reuse unchanged scientific/AD evidence;
the shared architecture, objective and source100 learning result remain valid.
No renewed paid allocation is submitted in this collection. Further runtime
archival is paused at the user's request.

Raw partial reports/logs are preserved remotely under
`runtime/outputs/diagnostics/shared-execution-0507e1b-gpu.*` and
`runtime/outputs/diagnostics/shared-dual-15328832.*`, with local copies under
`outputs/` using the same names. Do not relabel these partial reports as passed.
