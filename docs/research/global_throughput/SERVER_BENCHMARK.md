# Explore1000 Server Benchmark Contract

Status: CPU1 exploratory B16 job15591356 completed in1176s using1 CPU and0 GPUs.
Its full timing record is publishable as exploratory performance evidence;
cross-backend scientific consistency is not accepted and accepted CPU stage count
is0. Initial reader rejection was repaired and the actual saved-snapshot internal
reader validation passed; this does not make the CPU stage scientifically accepted
or authorize CPU4. Current work is the bounded B256 signature diagnostic and
preparation for the single-GPU B256/B512 continuation below;
the completed CPU assets remain available. The earlier bounded paid matrix and
GPU job15579869 remain historical evidence.
See the [CPU1 terminal result](../../../outputs/research/global_throughput/cpu-performance-results-15591356/CURRENT_RESULT.md),
[terminal measurements](../../../outputs/research/global_throughput/cpu-performance-results-15591356/terminal-measurements.json),
[cost scenarios](../../../outputs/research/global_throughput/cpu-performance-results-15591356/server-cost-results.json),
[initial reader attempt](../../../outputs/research/global_throughput/cpu-performance-results-15591356/cpu-internal-reader-admission.json),
[reader repair report](../../../outputs/research/global_throughput/cpu-reader-repair-20261006/REPAIR_REPORT.md),
[exact reader record diff](../../../outputs/research/global_throughput/cpu-reader-repair-20261006/record-differences.json),
[actual repaired-reader validation](../../../outputs/research/global_throughput/cpu-reader-repair-20261006/actual-reader-validation.json),
[historical stage evidence](../../../outputs/research/global_throughput/server-cost-results/EVIDENCE_REPORT.md),
[consolidated verification](../../../outputs/research/global_throughput/server-cost-results/verification.json),
and the [historical paid-round attempt and cost ledger](../../../outputs/research/global_throughput/server-cost-preparation/PAID_EXECUTION_LEDGER.json).
The independent CPU1 terminal attempt is recorded in the [new CPU1 ledger](../../../outputs/research/global_throughput/cpu-performance-preparation-20261006/PAID_EXECUTION_LEDGER.json).
This document records the prior server matrix and the bounded B256 interface
diagnostic. Its lower-only check completed in123.377s under a180s cap: caller,
public tree and internal AllArgsInfo each reported536 inputs, of which417 were
kept. No complete model compile or run occurred. CPU1 assets remain available;
CPU work is not a GPU prerequisite.

## CPU1 Terminal Measurement

Job15591356 completed B16 resident scalar:16 fixed real points, two consecutive
seven-day blocks,48 half-hour steps/day (224 point-days),1 CPU,0 GPUs, whole-job
elapsed1176s. It saved complete outputs and five synchronized core and prepared-
wrapper samples. Core median/range was31.458503002s/[30.855936289,31.737742664];
wrapper was31.760353759s/[30.915467992,32.648889609]. Accepted GPU B16 measured
4.149942968s/[4.142994318,4.152027067] core and4.364678632s/[4.359835308,
4.370462861] wrapper. GPU/CPU throughput ratios are7.580466345 core and
7.276676345 wrapper. These ratios describe exploratory performance only, not
scientific consistency or final cost superiority.

| Same-workload B16,224 point-days | Allocation | Core median [range] | Wrapper median [range] | Warm cost, primary core / wrapper |
| --- | --- | ---: | ---: | ---: |
| CPU1 resident scalar | 1 CPU,0 GPU | 31.458503002s [30.855936289,31.737742664] | 31.760353759s [30.915467992,32.648889609] | CNY0.00043692365281 / 0.00044111602443 |
| Accepted GPU B16 | 4 auxiliary CPUs,1 GPU | 4.149942968s [4.142994318,4.152027067] | 4.364678632s [4.359835308,4.370462861] | CNY0.00276662864536 / 0.00290978575498 |

The main price scenario is CNY0.05 per allocated CPU-hour and CNY2.20 per GPU-hour;
the conservative CPU rate is CNY0.07. Adding four auxiliary CPU charges makes one GPU
allocation CNY2.40/hour at the main rate (48 CPU units) or CNY2.48/hour at the
conservative rate. For this B16 warm workload, GPU/CPU warm cost ratios are6.332064
core and6.596418 wrapper: CPU1 is cheaper for the measured sample while GPU is faster
by7.580466345 core and7.276676345 wrapper throughput. This is a B16 warm-workload
price scenario, not a full-run, annual, or 669-point/50-year winner; ratio44 is not a
measured 44-CPU result.

There were17 fixed gates:15 passed,2 finite tolerance-only gates failed at
299.0-115.0 (bound ratio1.2284882227) and285.0-087.0 (1.7088155265). Native
dtypes and `atol=1e-8`, `rtol=1e-10` were unchanged; frozen repeat was identical.
Collector SHA/contract/sample checks admit the complete exploratory measurement,
while accepted CPU stage count remains0. The initial reader rejection came only from
type mismatch at `record.execution_origin.environment.loadavg`: checkpoint tuple
`(0.14,0.06,0.06)` versus JSON list with equal values. The patched reader permits
tuple/list equivalence only at that telemetry field, with length/finite/nonnegative
and exact per-element type/value checks; other record and science admission remains
strict. Actual validation against the unchanged CPU1 B16 snapshot passed
`_validate_cpu_internal`, including schema/nonfinite/repeat/timing checks, 760 leaves,
16 lanes, two complete blocks, and five core plus five wrapper samples. This grants
`cpu_internal_comparison` reader admission only; it does not clear either original
finite GPU parity failure or make CPU accepted. No model lower, compile, CPU4 runner,
or scientific execution occurred. See the reader-repair links above.

Whole-parent-job cost scenarios are CNY0.016333333 primary and0.022866667
conservative, counting job15591356 once over1176s. They include startup, compile,
gates, repeats, and I/O; they are not billed costs or per-point-day charges.
Keep warm work and setup separate: lower56.317166498s, compile688.378443662s,
first call32.158282455s, checkpoint serialization0.005034860s and atomic write
0.025802774s. No complete stage startup/validation accounting exists for a K-reuse
estimate. Full values and provenance are in the linked terminal artifacts above.

Launcher defaults remain `VALIDATION_MODE=strict`, `CPU_REFERENCE_ROLE=accepted`,
sizes `16 64`, CPU1 methods `scalar batch`, and CPU4 methods `pool batch`. The completed
CPU1 run selected `exploratory`, `cpu-internal`, size `16`, and `scalar` only. Exploratory
timing may retain and time complete outputs with finite tolerance-only CPU/GPU
differences, while reporting the numerical gate as failed. Point/state identity,
schema, shape/dtype, discrete values, nonfinite policy, and complete-output presence
remain hard stops. CPU internal snapshots are comparison references only. The initial
reader rejection was caused solely by `record.execution_origin.environment.loadavg`
being a tuple in pickle and an equal-valued list in JSON. The staged reader allows that
single metadata representation difference only, with strict value/type checks; actual
reader admission and all subsequent CPU-internal checks passed on the unchanged B16
snapshot. It remains `cpu_internal_comparison`, not accepted science. The earlier
conditional45-minute/4-CPU scenario or command is not ready for submission; CPU4 was
not run or approved. See the reader-repair artifacts linked at the top.

The CPU scripts request `ibc11b02n04`. A 2026-10-06 06:09:54Z snapshot recorded
that node IDLE, CPUAlloc 0/56, load 0.49, and FreeMem 171657 MiB; this is not a
reservation and occupancy must be checked again. The launcher default `CODE_DIR` is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu-performance-v1`;
the GPU launcher retains its separate `global-throughput-v1` default. CPU code commit
`4bc82a0747c2fd156430956e41ba426271a88b20` was deployed by verified bundle/fast-forward
and is clean. Local toy/protocol tests passed 49 cases; actual target Bash expansion
passed six normal and nine rejection cases. Canonical test02 CLI and four-worker
stdlib checks passed without model/JAX execution. The preparation review predates the
completed CPU1 measurement above. See the
[CPU run review](../../../outputs/research/global_throughput/cpu-performance-preparation-20261006/PAID_RUN_REVIEW.md)
for the original command, output/log roots, scheduler snapshot, and unverified
planning scenarios. Its CPU1 CNY0.025/0.035 figures were based on a30-minute ceiling,
not actual parent-job cost; actual terminal scenarios are CNY0.016333333/0.022866667.
The conditional CPU4 CNY0.15/0.21 scenario is not executable approval.

## Current Work Order: Repair and B256/B512 Preparation

Job15600405 is terminal FAILED on `ibc13b03n04` after2,135 parent seconds, with one
Tesla V100 and four allocated Slurm CPUs; observed affinity used two physical cores
through SMT. B128 fixed512 is complete and independently validated. B256 reached
compile-complete at stage elapsed2,058.173263s (lower63.179012s,
compile752.614130s), but its first `run_collection` call failed at
`executable(*first)` with `TypeError: Computation compiled for 673 inputs but called
with 536`. B256 has no completed first call, checkpoint or five-plus-five warm
samples, so no throughput is published. This is an AOT invocation-signature error,
not a numerical gate failure, OOM or timeout. The historical job did not diagnose
the cause, and B512 did not run. The bounded canonical JAX0.4.38 lower-only check
did not reproduce the natural failure: the caller, public tree and internal
AllArgsInfo agreed at536 inputs with417 kept. No full model compile or run occurred.
Sequential synthetic B128/B256/B512 signature checks passed; unrelated fault
injection of a673-input lazy-binding mismatch recreated the old exception, while
the `Compiled.call` bypass produced identical outputs. The defensive call-adapter
repair is not evidence that the historical cause was reproduced or fixed.
Frozen physics, dtype and tolerances are unchanged. See the
[repair report](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/REPAIR_REPORT.md),
[current result](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/CURRENT_RESULT.md)
and [paid-run preparation](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/PAID_RUN_REVIEW.md).
The latest canonical parent row records
`FAILED`, elapsed2135s, exit`1:0`. Earlier RUNNING snapshots remain historical.

The user approved one v2 submission, job15643327, submitted from `gln01` at2026-10-08 14:26:10 Asia/Shanghai. The requested n03 had all eight GPUs allocated, so after notifying the user the still-pending job was retargeted to n02 with `scontrol`, without requeue or restart. The retarget snapshot at2026-10-08T06:28:47Z recorded CPUAlloc12/112, load3.19, freeMem323835MiB and AllocTRES `cpu=12,gres/gpu=1`. The job started14:30:20 and ended14:30:21 Asia/Shanghai, `FAILED` exit127:0: its copied wrapper incorrectly looked for the child launcher under `/ram/var/spool/slurm/d/job15643327/`. It failed before GPU/model initialization; no stage, output, gate or sample exists. Parent accounting is one row: 1 wall second, 4 CPU-seconds and 1 GPU card-second. Estimated one-second allocation cost is CNY0.000666667 primary / CNY0.000688889 conservative; billing/rounding are unverified. See [submission receipt](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/submission-receipt.txt), [initial pending snapshot](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/submission-status.txt), [retarget-time node snapshot](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/submission-node-snapshot.txt), [retarget receipt](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/node-retarget-receipt.txt), [terminal parent accounting](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/job-15643327-parent-final.txt), and [startup failure](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/job-15643327-startup-failure.txt).

The v2 approval covered only that one attempt and is consumed. No automatic retry was issued or scheduled. Corrected v3 wrapper/deployment is verified at commit `12b495b9425de3ad1dc16efa3a5a0e69d7e89222` in the independent checkout `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3`; its run root `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512` is unused. The exact v3 command is [APPROVAL_COMMAND_V3.sh](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/APPROVAL_COMMAND_V3.sh). The v2 run root was not populated with benchmark outputs.

The workload contains 512 lanes from 64 unique FULL64 points repeated eight times,
covering two consecutive seven-day blocks. It measures 7,168 lane-days; 896
unique-point-days are listed separately and are not the effective throughput
denominator.

The B128 reader was rechecked against original SHA/bytes, 760×512 leaves, 515
gates, 5+5 samples and provenance. JSON tuple/list equivalence applies only to
`loadavg`; science-input and schema mismatches remain rejected. The old B128
lazy-call and new B256/B512 Python-bound dispatch differ, so actual collection
times and costs can be compared but rate ratios cannot be attributed to batch size
alone. The natural old failure remains unexplained. Final module-level canonical
JAX0.4.38 validation passed against the actual immutable checkout at
`9d19f0760b93e1186eaf796db4903ddfd16a806f`; the diagnostic records module paths,
hashes and commit. Final Bash syntax and target-shell argument expansion also
passed, producing candidate `[256,512]`; dry-run imported no JAX and initialized no
model/GPU. See [final module diagnostic](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/final-module-canonical-diagnostic.jsonl),
[launcher verification](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/final-launcher-verification.txt)
and [deployment verification](../../../outputs/research/global_throughput/gpu-signature-repair-20261008/deployment-verification.json).
The checkout is a clean independent clone with no hardlinks or alternates; old
checkout `a51efcd` is unchanged. No GPU B256/B512 collection has been validated.

| B128 fixed512 capacity stage | Median [range] | Throughput / warm cost scenario |
| --- | ---: | ---: |
| Synchronized core, five collections | 29.617320834s [29.605399007,29.623860434] | 242.020541 lane-days/s; CNY0.019744881 primary /0.020403043 conservative |
| Prepared wrapper, five collections | 33.838030240s [33.780365391,33.889177628] | 211.832661 lane-days/s; CNY0.022558687 primary /0.023310643 conservative |

The complete stage passed 515 gates and 512 independent saved-output/source-B4
comparisons across 760 native output leaves per lane and both blocks. All 512 output
lanes were retained. Strict numerical and saved-reference checks passed; checkpoint
SHA256 is `e91db1391ccb5f525f9ec29f1cba24235c4ce9cd4153b9297fe2ed014d0c9cb5`.
This is `exploratory_capacity_performance` with `scientific_acceptance=false`;
repeated lanes do not represent 512 distinct real grids, and B128 alone neither
establishes a sweet spot nor demonstrates global scientific benefit.

Separate B128 timing: lower92.979298s, compile744.949671s, first complete call
29.778435s, host stack3.176700s, device transfer1.047304s, checkpoint serialization
0.321704s and atomic write0.251202s. Actual compiler analysis recorded argument,
output and temporary buffers of 31,118,364/27,529,184/582,289,872 bytes. The selected
V100 snapshot immediately before compilation had 33,721,155,584 of 34,359,738,368
bytes free; job-start Host MemAvailable was 352,786,501,632 bytes; cumulative process
peak RSS through B128 was 14,736,539,648 bytes. GPU memory is a boundary snapshot,
not a measured VRAM peak; compiler analysis is not actual peak allocation. B256 and
B512 require their own memory evidence. Warm costs above cover only one synchronized
512-lane collection using one GPU and four allocated CPUs; they exclude setup,
compilation and checkpoint IO. Whole-job cost will use one terminal parent `sacct`
row and count job15600405 once.

The parent allocation used 0.59305556 GPU card-hours and 2.37222222 allocated CPU
count-hours. At the declared GPU CNY2.20/card-hour and CPU CNY0.05/core-hour rates,
the estimated whole-job primary cost is CNY1.423333333; using CPU CNY0.07 gives
CNY1.470777778. Terminal collector cross-check accepted these as allocation
estimates, not a verified bill. Queue time and child steps are not added; the parent
is counted once.
The report's `observed_window_wall_seconds=2121.916518688202` is a separate window
measurement. `elapsed_seconds=2060.684793` is stale and is not terminal billing.

Evidence: [current result](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/CURRENT_RESULT.md),
[terminal validation](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-validation.json),
[terminal transfer verification](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-transfer-verification.json),
[B256 failure diagnostics](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-failure-diagnostics.json),
[collector verification](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/collector-verification.json),
[terminal raw nine-field `sacct`](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-sacct-raw9.txt),
[terminal collector seven-field `sacct`](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-sacct-collector7.txt),
[terminal evidence capture](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-evidence/capture.json),
[preserved B128 checkpoint](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/job-15600405-terminal-evidence/capacity__stage_capacity_B128_fixed512_v1.pkl),
and [current ledger](../../../outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/preparation-ledger.json).

Deployed code remains immutable at `a51efcdabf7ff2e9f97a463a6f72e9f6a2cf478e`.
Heartbeat `gpu-15600405` was confirmed `PAUSED`; its old-job terminal follow-up is
complete. The deleted old Explore1000 heartbeat is not restored, and no new
automation is planned. The repair diagnostics are free and bounded. Immutable
deployment and private-API validation passed. The v2 paid continuation was approved
and attempted once; job15643327 failed during wrapper startup before model/GPU
initialization, as detailed above. The corrected v3 continuation is not approved
or submitted: proposed resources are `gnall`, 1 node, 1 task, 4 Slurm CPUs,
1 Tesla V100 32GiB and 90 minutes, with scenario estimates CNY3.60 primary and
CNY3.72 conservative. The immutable checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3`
at commit `12b495b9425de3ad1dc16efa3a5a0e69d7e89222`; its run root is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512`
and remains unused. The 2026-10-08T06:50:43Z snapshot found no idle eligible node;
candidate `ibc13b03n02` was mixed (12/112 CPUs allocated, load2.85, 323601MiB free,
one GPU allocated), not a reservation. Refresh scheduler state before any approved
submission. Obtain fresh explicit approval for resources, wall time, working
directory, exact command and worst-case cost before submission. Reuse the sealed
B128 evidence; run B256 then B512 only, retaining the 48-step/two-block native-output
and 5+5 synchronized-sample contracts. The 1,800-second B512 runtime floor plus
buffer/slack is an admission guard, not a throughput acceptance criterion. Existing
GPU B16/B64 results are
separate evidence; 1-to-2-card work comes later. CPU reader commit `7d20a17` is
accepted and CPU-internal reader admission passed, but the two CPU/GPU tolerance
failures remain and CPU4 was not run. CPU expansion remains paused and is not a
prerequisite for the GPU screen.

## Completed Paid Matrix: Fixed Workload

Reuse the admitted Linux FULL64 inputs at
`runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`.
Run only the first 16 and all 64 distinct landpoints. Every point starts from
the same frozen Day1 state and runs the same two seven-day blocks, with 48
half-hour steps per day, native dtypes, complete state outputs, and the frozen
forcing and parameters. Do not synthesize duplicate points to fill a batch.

The scalar CPU result for each selected size is the CPU parity reference. The
four-worker and batched CPU modes must consume the identical point IDs and
inputs. GPU B64 must first match the preserved complete FULL64 B4 numerical
references under the existing exact science, dtype, shape, mask, and
non-finite policy. Candidate outputs, gates, and at least five synchronized
core and wrapper samples must be atomically saved before an optional new B4
resident baseline is compiled. The prepared GPU launcher does not request
that optional stage by default. Once the B64 candidate is sealed, measure B16;
pass sizes in explicit order `64 16` so the B16 screen cannot precede the B64
candidate gate.

The requested matrix is:

| Launcher | Slurm resources | Work | Maximum wall time |
| --- | --- | --- | --- |
| `slurm_global_server_cpu_1core.sh` | `cnall`, 1 node, 1 task, 1 CPU | Resident sequential scalar then JAX batch with CPU budget 1; B16 and B64 | 1 hour |
| `slurm_global_server_cpu_4core.sh` | `cnall`, 1 node, 1 task, 4 CPUs | Four persistent one-core workers, then JAX batch with CPU budget 4; B16 and B64 | 1 hour |
| `slurm_global_server_gpu_b64.sh` | `gnall`, 1 node, 1 task, 4 auxiliary CPUs, 1 GPU | Candidate-first B64 validation, then B16 measurement; five synchronized core/wrapper repeats per measured size | 1 hour |

Each Python run must prove it is inside the matching Slurm allocation. CPU
launchers use `srun --cpu-bind=cores`; the benchmark checks its inherited
affinity contains exactly the requested number of distinct physical cores
before importing JAX. The CPU node exposes 56 physical cores with one hardware
thread per core. The GPU node exposes 112 Slurm CPUs over 56 physical cores
with two-way SMT. Do not add `--hint=nomultithread` to the GPU request: preserve
the requested four Slurm CPUs, then record `AllocTRES`, logical affinity,
physical-core mapping, and SMT siblings. The pool controller shares worker
zero's assigned core; there is no fifth CPU or nested full-node process pool.

The old 10 GiB (CPU1) and 40-60 GiB (CPU4) compiler figures were planning estimates,
not measured peaks. A later single-point CPU diagnostic reached 20.181 GiB RSS on a
different workload, so do not present the 10 GiB estimate as current evidence. CPU1
scalar and batch stages require 24 GiB host `MemAvailable`; CPU4 pool requires 96 GiB
and CPU4 batch requires 24 GiB. These are launcher admission guards, not reservations
or measured peak RSS. Do not
request `--mem`: CPU-node `RealMemory=1` is a broken scheduler value. Record actual
RSS and host memory at runtime.

The CPU matrix loads the complete B4 numeric reference bundle for parity; it
does not compile another B4 executable. CPU1 still requires one resident
scalar compile and B16/B64 batch stages. CPU4 requires four resident scalar
worker compiles plus B16/B64 batch stages. The one-hour limit is therefore a
bounded ceiling with unknown completion probability, not a completion-time
estimate. If stages remain incomplete, keep their atomic outputs and report
the missing stages without requesting extra time.

The current CPU1/CPU4 scripts request `ibc11b02n04`; the 2026-10-06 06:09:54Z
snapshot recorded it IDLE, CPUAlloc 0/56, load 0.49, FreeMem 171657 MiB. Recheck
occupancy before any submission; that snapshot is not a reservation. The old paid GPU
launcher requested `ibc13b03n03`, which was mixed-use in its bounded scheduler
snapshot, not idle. Node choices are point-in-time observations, not reservations.
The CPU1/CPU4 hard limits are now 30/45 minutes. Preserve complete atomic stages;
report missing stages without extending the limits or retrying without new review.

Use the accepted project environments at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu` and
`.../.venvs/orcjax_gpu`, both on JAX/JAXLIB 0.4.38. The GPU launcher uses the
accepted Singularity runtime and dynamically resolves host `libcuda.so.1` and
`libnvidia-ml.so.1`. It must preserve Slurm's allocated `CUDA_VISIBLE_DEVICES`
for execution. Resolve host `nvidia-smi` identity from the single global ID in
`SLURM_JOB_GPUS`, and record both that ID and `CUDA_VISIBLE_DEVICES`; Slurm can
remap CUDA ordinals inside cgroups or steps. Do not infer host identity from
the visible ordinal or force physical GPU 0. Select a GPU node only from a
fresh occupancy check within the approved allocation. A node observed free
earlier is not a reservation.

The launchers preserve `ORCHJAX_SLURM_ALLOC_TRES` for benchmark provenance.
They prefer the exact `AllocTRES=` field from `scontrol show job -o`; when
that field is absent they use the exact allocated `TRES=` field. They never
substitute `ReqTRES=`. `launch-context.txt` records the selected field source
as `slurm_alloc_tres_source`.

The present bounded project inventory found no current-platform-matched
`orchidee_ol`, `orchidee.e`, or `xios_server.exe` plus build configuration.
Therefore this matrix can establish CPU/GPU throughput and cost against the
same-workload JAX modes, but it cannot report a current Fortran speedup. The
historical estimate of two minutes per point-year remains context only until a
matching Fortran executable, configuration, and frozen workload are measured.

## Current CPU Launcher Inputs

Set these variables before submission when the default paths do not match the
immutable Explore1000 checkout:

- `CODE_DIR`: clean, transferred code checkout containing the CPU benchmark entry
  points. The CPU1/CPU4 default is
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu-performance-v1`.
  The paid GPU launcher keeps its separate historical
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-v1` default.
- `PREPARED_INPUT`: the frozen FULL64 pickle above.
- `GPU_REFERENCE_DIR`: the accepted complete FULL64 B4 numerical-reference
  tree used by the CPU gates. The current default is
  `runtime/outputs/research/global_throughput/gpu-round4/model`.
- `CPU_REFERENCE_DIR`: for the four-core CPU launcher, the `scalar` output
  directory produced by the one-core job, explicitly marked with its `accepted` or
  `cpu-internal` role. `cpu-internal` requires `exploratory` mode and is not accepted
  scientific truth.
- `VALIDATION_MODE`, `CPU_REFERENCE_ROLE`, `BENCHMARK_SIZES`, and `CPU1_METHODS` /
  `CPU4_METHODS`: defaults are strict/accepted, sizes `16 64`, methods `scalar batch`
  for CPU1 and `pool batch` for CPU4. The reviewed CPU1 command overrides these with
  exploratory/cpu-internal, size16, scalar only.
- `RUN_ROOT`: a new per-job directory under
  `runtime/outputs/research/global_throughput/server-benchmark/`.
- `CONFIG`: the declared `configs/orchidee_man_250919.yaml` file in `CODE_DIR`.

The default job output roots are
`server-benchmark/cpu1-${SLURM_JOB_ID}`,
`server-benchmark/cpu4-${SLURM_JOB_ID}`, and
`server-benchmark/gpu-b64-${SLURM_JOB_ID}`. Their mode directories are
`cpu1-*/scalar` and `cpu1-*/batch1`, `cpu4-*/pool4` and `cpu4-*/batch4`, and
`gpu-b64-*/candidate-first`, respectively. Launcher logs are written to
`runtime/logs/global_server_cpu1_<jobid>.txt`,
`runtime/logs/global_server_cpu4_<jobid>.txt`, and
`runtime/logs/global_server_gpu_b64_<jobid>.txt`. Set `CPU_REFERENCE_DIR` for
the CPU4 launcher to the completed CPU1 scalar directory:
`.../server-benchmark/cpu1-${CPU1_JOB_ID}/scalar`.

After the jobs finish, summarize only the mode directories that exist, using
the actual Slurm job IDs. For the complete matrix, run from `CODE_DIR` and
replace the shell variables below with the recorded job IDs:

```bash
SERVER_BENCHMARK_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark
CPU_PYTHON=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python
CPU1_ROOT=${CPU1_RUN_ROOT:-$SERVER_BENCHMARK_ROOT/cpu1-${CPU1_JOB_ID}}
CPU4_ROOT=${CPU4_RUN_ROOT:-$SERVER_BENCHMARK_ROOT/cpu4-${CPU4_JOB_ID}}
GPU_ROOT=${GPU_RUN_ROOT:-$SERVER_BENCHMARK_ROOT/gpu-b64-${GPU_JOB_ID}}
"$CPU_PYTHON" scripts/dev/summarize_global_server_cost.py \
  --cpu-dir "$CPU1_ROOT/scalar" \
  --cpu-dir "$CPU1_ROOT/batch1" \
  --cpu-dir "$CPU4_ROOT/pool4" \
  --cpu-dir "$CPU4_ROOT/batch4" \
  --gpu-dir "$GPU_ROOT/candidate-first" \
  --job-accounting "$SERVER_BENCHMARK_ROOT/job-accounting-cpu1-${CPU1_JOB_ID}.json" \
  --job-accounting "$SERVER_BENCHMARK_ROOT/job-accounting-cpu4-${CPU4_JOB_ID}.json" \
  --job-accounting "$SERVER_BENCHMARK_ROOT/job-accounting-gpu-${GPU_JOB_ID}.json" \
  --reuse-counts 1 10 100 \
  --output "$SERVER_BENCHMARK_ROOT/cost-summary-${CPU1_JOB_ID}-${CPU4_JOB_ID}-${GPU_JOB_ID}.json"
```

The summarizer creates a new output file and withholds incomplete or
unverified stages. Omit unavailable mode directories rather than pointing it
at a partial directory; include a directory only when it contains that
mode's `stage_manifest.json` and `report.json`. The accounting files are
separate observed Slurm job records. Include a record for each terminal job
that produced one or more complete stages, even when the job ended by timeout
or failure. For CPU1, list both mode directories from the same job when both
were created; if the job ended before batch started, list the scalar directory
alone. The whole-job charge is associated with each listed stage but is not
additive across stages.

Create the external accounting JSON from the actual terminal Slurm record
(`sacct` plus the job's `launch-context.txt`). This CPU1 example shows the
required fields and both mode directories; its job values are illustrative
and must be replaced with that run's observed values:

```json
{
  "directories": [
    "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/cpu1-123456/scalar",
    "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/cpu1-123456/batch1"
  ],
  "status": "completed",
  "wall_seconds": 1842.7,
  "allocated_CPU_count": 1,
  "GPU_count": 0,
  "provenance": {
    "source": "sacct job record and launch-context.txt",
    "job_id": "123456",
    "partition": "cnall",
    "node": "ibc11b02n25",
    "AllocTRES": "cpu=1,mem=...,billing=1",
    "observed_wall_seconds": 1842.7
  }
}
```

Record the actual terminal state, such as `completed`, `timeout`, `failed`, or
`cancelled`, and copy observed wall seconds, allocated CPU/GPU counts, node,
partition, and `AllocTRES`. CPU4 lists its existing `pool4` and `batch4`
directories; GPU lists its `candidate-first` directory. The summarizer accepts
terminal statuses including `completed`, `timeout`, `timed_out`, `failed`,
`cancelled`, `canceled`, `over_budget`, and `stopped`. Whole-job accounting
supplies aggregate job cost only. The summarizer reports K-reuse estimates only
when the accounting record separately includes observed
`stage_startup_and_validation_seconds` for the exact stage names and that stage
report includes measured serialization and atomic-write times. Without both,
K-reuse remains unavailable; job wall time alone is not an estimate of
per-stage amortization overhead.

## Historical Paid Approval-Time Output Paths

Use a new named parent under the permitted server-benchmark output root when
preparing fixed approval commands. The short commit component prevents reuse
of a previous run's output paths:

```bash
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-v1
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
SERVER_BENCHMARK_ROOT=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark
APPROVAL_ROOT=$SERVER_BENCHMARK_ROOT/global-throughput-v1-REPLACE_WITH_SHORT_COMMIT
cd "$PROJECT"

RUN_ROOT=$APPROVAL_ROOT/cpu1
"$SLURM_BIN/sbatch" --export=ALL,CODE_DIR="$CODE_DIR",RUN_ROOT="$RUN_ROOT" \
  "$CODE_DIR/scripts/hpc/slurm_global_server_cpu_1core.sh"

# After CPU1 scalar stages complete, point CPU4 at that immutable scalar result.
RUN_ROOT=$APPROVAL_ROOT/cpu4
CPU_REFERENCE_DIR=$APPROVAL_ROOT/cpu1/scalar
"$SLURM_BIN/sbatch" --export=ALL,CODE_DIR="$CODE_DIR",RUN_ROOT="$RUN_ROOT",CPU_REFERENCE_DIR="$CPU_REFERENCE_DIR" \
  "$CODE_DIR/scripts/hpc/slurm_global_server_cpu_4core.sh"

RUN_ROOT=$APPROVAL_ROOT/gpu-b64
"$SLURM_BIN/sbatch" --export=ALL,CODE_DIR="$CODE_DIR",RUN_ROOT="$RUN_ROOT",MEASURE_B4_BASELINE=0 \
  "$CODE_DIR/scripts/hpc/slurm_global_server_gpu_b64.sh"
```

Replace `REPLACE_WITH_SHORT_COMMIT` with the transferred clean checkout's commit prefix
and choose an unused parent before seeking approval. These commands define
paths for review; they do not authorize submission.
When these explicit roots are used, set `CPU1_RUN_ROOT=$APPROVAL_ROOT/cpu1`,
`CPU4_RUN_ROOT=$APPROVAL_ROOT/cpu4`, and
`GPU_RUN_ROOT=$APPROVAL_ROOT/gpu-b64` before running the summarizer above.

The GPU launcher uses `FULL_REFERENCE_DIR` for the preserved complete FULL64
B4 reference group. Its candidate-first mode does not read the old GPU
round-three anchor reference directory. The current default is the same
`gpu-round4/model` reference group. All sources must remain immutable and
must not overlap a new output directory. The launcher rejects an already
existing job output path.

The CPU one-core job writes scalar results first and then uses those files for
the one-core batch parity gate. The four-core job consumes that passed scalar
reference directory for both the worker-pool and four-core batch gates. Run
the CPU jobs sequentially so `CPU_REFERENCE_DIR` names a completed, immutable
result. Each size and mode writes to a separate child of `RUN_ROOT`.

## Recorded Measurements

### Completed GPU Job 15579869

The job completed on `ibc13b03n03` from 2026-10-06 00:21:41 UTC to 00:56:51 UTC,
using one V100 and four allocated Slurm CPUs. The recorded affinity was logical
CPUs 28, 29, 84, and 85, mapping to two physical cores. Both accepted stages
passed the fixed `atol=1e-8`, `rtol=1e-10` gates with native dtypes and verified
complete checkpoints. Each timing metric has five fully synchronized samples.

| Stage | Point-days | Core median [range] | Core throughput | Wrapper median [range] |
| --- | ---: | ---: | ---: | ---: |
| B64 | 896 | 5.632416 s [5.627946, 5.635270] | 159.079 point-days/s | 6.222333 s [6.189429, 6.311674] |
| B16 | 224 | 4.149943 s [4.142994, 4.152027] | 53.976645 point-days/s | 4.364679 s [4.359835, 4.370463] |

Lowering, compilation, first call, stack/transfer, and checkpoint serialization/write
measurements are retained separately in [terminal-measurements.json](../../../outputs/research/global_throughput/server-cost-results/terminal-measurements.json);
they are not included in hot-core throughput. No `K`-reuse estimate is available because observed
stage setup/validation accounting is absent. CPU1 still failed the fixed B16 gate at
point 7 (299.0-115.0). That paid B16/B64 scalar reference remains incomplete. A
separately authorized free correctness diagnostic has now completed one full scalar
Days 2-8 block for only this point on one CPU core and zero GPUs. The 380-leaf fixed
comparison has one failure: `[0][0][7]`, the final-state
`diffuco_previous_step_state.rveget[0,13]` value (whole-canopy stomatal resistance,
s/m; `float64`, shape `(1,14)`). CPU was 592.8152487724788 versus GPU B4 reference
592.8152488575903: signed delta `-8.511153737345012e-8`, absolute delta
`8.511153737345012e-8`, bound `6.928152488575903e-8`, ratio `1.228488222708634`.
The gate-wide max absolute difference `8.940696716308594e-8` is aggregate, not the
target leaf error. There are nonzero differences in 165 leaves; the other 379 pass,
including discrete and sentinel checks. The old paid CPU1 candidate was not saved;
the value above belongs only to this diagnostic. See [current result](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/CURRENT_RESULT.md),
[target-leaf comparison](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/remote-evidence/shared-final/rveget_comparison.json),
[all-leaf summary](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/remote-evidence/shared-final/all_leaf_comparisons.json),
[collection verification](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/COLLECTION_VERIFICATION.json),
and [IR analysis](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/IR_ANALYSIS.md).

The CPU run is a one-point, one-block correctness result without warm repeats; it is
not a CPU throughput or CPU/GPU speed/cost comparison. Total window wall was
1636.648 s (1653 s conservative accounting), using both allowed full compilations
and one complete block call. The first compilation was lost to a duplicate
post-compile artifact-entry guard before any model call; the corrected final call
completed without time or RSS termination. The target-owned closure was not compiled
because the 2/2 budget was exhausted, so closure binding cannot be distinguished
from CPU scalar/GPU B4 arithmetic, backend, or compiler contributions. The exact old
StableHLO identity remains unmet: its raw
IR was not saved. Two preserved IRs are textually isomorphic after internal-order
permutations; set-derived field ordering is a plausible mechanism, not an established
cause. Compiler arithmetic/fusion and NumPy mismatch remain hypotheses. The paid
execution ledger and heartbeat stay closed/paused; this diagnostic's completion does
not mark the broader throughput research paused. CPU4 remains withheld. No Fortran,
annual, 669-point, 50-year, or 10x result is established.

Using unverified billing rates of CNY 0.05/0.07 per allocated CPU core-hour and
CNY 2.20 per GPU card-hour as additive scenarios, the GPU whole-job estimate is
CNY 1.406666667 primary and CNY 1.453555556 conservative. Across all terminal
attempts, allocated totals are 9,316 CPU-count-seconds and 2,115 GPU-card-seconds,
giving CNY 1.421888889 primary or CNY 1.473644444 conservative under those scenarios;
these are not verified actual charges. Each job is counted once; no step costs are
added again. The round remains within the original CNY 2.83 conservative approved
ceiling. The [evidence report](../../../outputs/research/global_throughput/server-cost-results/EVIDENCE_REPORT.md)
and [paid execution ledger](../../../outputs/research/global_throughput/server-cost-preparation/PAID_EXECUTION_LEDGER.json)
are the current records.

The GPU launcher records pre-query job, host, affinity, global GPU ID and
unchanged `CUDA_VISIBLE_DEVICES` before resolving physical identity. The
read-only `global_gpu_identity.py` helper matches a numeric Slurm global ID
to the exact NVIDIA `/proc` device minor, then verifies UUID, device minor
and PCI address against accessible `nvidia-smi -q -x` metadata. A UUID Slurm
ID requires the same driver/NVML agreement. It rejects unknown or ambiguous
mappings and never substitutes the first visible GPU or uses visible index 0
as a physical identity. Numeric ID-to-device-minor agreement is an explicit
admission constraint; a differently ordered scheduler mapping needs separate
evidence before acceptance. The launcher saves `gpu-identity.json`, names
each startup step and reports failing commands with exit codes. It queries
inventory and visible processes only for the verified UUID and retains query
stdout, stderr and exit codes; these optional diagnostic failures do not
invalidate the mandatory physical-identity check or stop startup. The helper
does not independently resolve a CUDA visible selector to a UUID when NVML
exposes multiple devices. A single allocated Slurm GPU cgroup and unchanged
CUDA mask remain execution conditions checked by the allocated GPU entry.

Job 15579759 exited 6 before Python in 2 seconds with an empty scheduler log and
no launch context. Existing evidence localizes this to identity/driver setup
before context logging; its actual global GPU ID and failing command were not
retained, so a cgroup index mismatch remains a hypothesis. gln01 metadata
checks confirm XML UUID/minor/PCI agreement on that free node, without proving
the cause or paid-node runtime acceptance. Repairs deploy to a new immutable
checkout while the CPU1 checkout is active.

For every mode and size, retain the complete report, point identities,
numerical gates, cold compilation/lowering time, hot core time, wrapper time,
input preparation and stack/transfer time, synchronized sample values, output
and checkpoint I/O, peak RSS, and process CPU time. Record the commit and clean
checkout state, Python/JAX/JAXLIB/NumPy versions, actual node, Slurm job ID,
partition, `AllocTRES`, requested and observed CPU affinity, physical-core and
SMT mapping, host load, available memory, and GPU UUID/model/driver plus
visible GPU processes for GPU work. Keep job wall time separate from measured
compute time.

Report throughput as `real point-days / synchronized seconds`, with 14
point-days per landpoint. Compare scalar and batched modes only at the same
point count and total point-days. Report the median and full range of at least
five warm samples; keep cold start/compile and wrapper costs separate. Any
projection beyond the measured 16- and 64-point windows must be labeled as an
estimate, with its assumed reuse count and compilation amortization shown.

## Historical Paid Cost Bound

The primary comparison follows the user's current GPU-to-CPU rate ratio of 44
CPU core-hours per GPU-hour. At CNY 2.20/GPU-hour, that implies CNY 0.05 per
CPU core-hour; the implied CPU rate is not a new billing confirmation. The
deployment guide separately lists CNY 0.07 per allocated CPU core-hour, so
use that higher CPU rate for the conservative additive upper estimate until
the billing rule is confirmed. Charge requested Slurm CPUs, not measured CPU
utilization. Do not request exclusive nodes or extra GPUs.

| Job | User-ratio CPU estimate | Conservative CPU estimate | GPU estimate | Conservative total |
| --- | ---: | ---: | ---: | ---: |
| 1-core CPU, 1 h | CNY 0.05 | CNY 0.07 | CNY 0.00 | CNY 0.07 |
| 4-core CPU, 1 h | CNY 0.20 | CNY 0.28 | CNY 0.00 | CNY 0.28 |
| 4-core + 1-GPU, 1 h | CNY 0.20 | CNY 0.28 | CNY 2.20 | CNY 2.48 |

The full three-job matrix has a conservative maximum of CNY 2.83, assuming
the CPU and GPU charges are additive. The GPU job is 1 allocated GPU-hour and
4 auxiliary CPU-core-hours; at the user-ratio rate its combined cost is
equivalent to 48 CPU-core-hours. Under the user-ratio rate the three jobs total
CNY 2.65. These are requested-time ceilings, not expected spend. A paid
submission still requires explicit approval of the selected jobs, exact
resources, finite wall time, working directory, command, and worst-case cost
after fresh scheduler and occupancy inspection.

## Submission Boundary

These scripts are preparation artifacts, not authorization to submit. Before
submission, verify the transferred checkout and all input/reference paths,
create `runtime/logs` and the output parent under the project runtime tree,
change to the project root, inspect current
`sinfo` and candidate-node CPU/GPU occupancy, review the final `sbatch`
expansion, and obtain explicit approval for each paid resource envelope.
Never run a performance benchmark on a login node or shared free test node.
