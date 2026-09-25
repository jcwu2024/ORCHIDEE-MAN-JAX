# Bounded saved-executable comparison on an idle paid node

Status:15386166 FAILED124 after28min6s; completed14:38:11CST. No optimizer
update or full dual comparison completed. Initial RUNNING notes below are historical.

## Outcome and next correction

The internal28minute timeout ended the job while loading T30. Slurm recorded
FAILED124:0, not a numerical rejection or a backend gradient-compilation error.
The final JSON still says running/load_saved_gradient; scheduler/log termination
supersedes that stale snapshot. At project additive rates, elapsed usage costs
approximately2.32yuan, within the approved2.48yuan maximum; this is an estimate,
not an invoice. No dual throughput claim is supported.

| Phase | Measured wall time |
| --- | ---: |
| Configuration stage to bootstrap stage | 152.0s |
| Runtime/head/objective preparation to fresh initialization stage | 1218.6s |
| Initialization/checks/update0 checkpoint | 10.5s |
| Prepare first gradient inputs | 52.8s |
| Read daily executable pickle | 0.498s |
| Deserialize/load daily executable | 43.956s |
| Read T7 executable pickle | 0.439s |
| Deserialize/load T7 executable | 158.821s |

The first four rows are differences between coarse progress messages, not
exclusive profiled functions. No substage attribution within1218.6s is available.
In particular, do not label all startup time as filesystem IO or compilation.
Successful short preflight reads did not predict the complete startup duration.

A subsequent free gln01 CPU-only JAX startup check timed out at60s:
import48.22s, total wall61.46s versus user1.14s/system1.38s; device discovery
had not returned. A55s stack-instrumented follow-up completed in33.52s,
user1.19s/system0.98s: import26.43s, CPU devices ready29.85s. Its15s traceback
was in importlib.get_data while importing JAX sharding/stage modules from
the shared environment. This establishes module-read waiting on that free
node at that time, not the cause of every paid-node deserialize second.

Source inspection also found avoidable replay work: the diagnostic calls the
full learning bootstrap, which reloads the coherent typed-sidecar index,
statistics, canonical ordinary/restart examples, retained templates and head,
even though saved gradient executables need only parameter/data trees.
Prepare those exact two-artifact argument trees once on free CPU and replay
from a consolidated bundle plus the declared fresh update0 checkpoint.
Preserve source/method identity and all serial/dual gradient, parameter and
AdamW comparisons. This preparation route now passes10focused local tests
and CPU/GPU target-shell dry-runs; actual preparation98568 started15:48:44
on free test02. It has not yet demonstrated GPU speed. Do not repeat the paid
job unchanged. This approval was consumed by15386166, not a standing approval.

The loader now separates read and deserialization progress and CPU/wall time;
worker diagnostics remain on stderr to preserve the private pickle protocol.
This reporting change does not alter model computation.

## Submission evidence

User explicitly approved this allocation after the prepared proposal.
Immediately before submission, ibc13b03n04 was IDLE with CPUAlloc0,
CPULoad0.01 and no allocated GPU. The exact command below returned15386166.
Slurm start2026-09-25T14:10:05CST, deadline14:40:05,2GPUs/8CPUs/64GiB.
Actual startup reports CUDA0,1, CPU affinity0-3,56-59 and host load0.00.
The three short reads passed; the runner entered configuration loading.
No complete update or throughput result is claimed by this startup snapshot.

Log: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dual-replay-15386166.log`.
Result: same directory, `shared-dual-replay-15386166.json`.
Inspect with `/rmprog/slurm/v22.05.7/bin/squeue -j 15386166` and, after exit,
`/rmprog/slurm/v22.05.7/bin/sacct -j 15386166 --format=JobID,State,Elapsed,ExitCode`.

The resumed free probe104560 started13:24CST through cln01. At13:47 it
remained in bootstrap, with20seconds cumulative Python CPU across23minutes,
waiting in Lustre calls. The same Python library file was readable from cln01
in0.11seconds. This does not establish the storage fault's cause, but supports
trying a different compute node rather than recompiling the model.

At13:55 the free attempt was confirmed exited124 at its30minute bound,
still in bootstrap with no optimizer update or full-gradient binary load.
Python exited and no GPU process remained. Do not treat the last JSON's
running stage as current; the launcher log records the timeout.

## Exact proposed allocation

- Partition: gnall,1node,2V100,8allocated CPUs,64GiB,30minutes maximum.
- Preferred node: ibc13b03n04. Scheduler snapshot13:44CST: IDLE,0allocated
  CPUs/GPUs,CPULoad0.01. Recheck before submission; do not request exclusivity.
- Working directory:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`
  (unchanged c599236).
- Exact command:
  `/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n04 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/job_shared_serialized_dual.sh`
- Maximum charge using current project rates and additive CPU/GPU billing:
  CPU8*0.5*0.07=0.28yuan; GPU2*0.5*2.20=2.20yuan; total2.48yuan.

`job_shared_serialized_dual.sh` maps the actual two Slurm CUDA selectors to
the persistent parent/worker; target-shell dry-runs cover0,1 and2,5 allocations.
It reads64bytes each from the existing runner, Python stdlib and saved profile,
with15second bounds per file. A failed read exits75 before model setup. This
is an early rejection for clear IO stalls, not proof all later reads are fast.
The outer Slurm bound is30minutes and inner model probe28minutes, with30second
termination grace. The known-free-only launcher has a separate paid staging
filename, leaving the active free launch script untouched.

The job loads existing trusted daily/T7/T30 binaries, compares two changing
updates and matched32/64 complete proposals, including IPC and AdamW. It does
not recompile model gradients, change architecture/loss, regenerate Teacher
data or retain a formal trained model. Exact comparison protocol remains in
`shared_split_serialized_dual_preparation_20260925.md`. Reports use the Slurm
job ID. Success is measured parity plus actual complete-update throughput;
no speedup is promised from the node change alone.
