# Explore1000 Server Benchmark Contract

Status: user-approved bounded execution. CPU1 job 15579758 failed its fixed
numerical parity gate after852 seconds, before a complete scalar B16/B64
reference was sealed. CPU4 is withheld and CPU1 is not automatically replayed.
GPU startup replacement15579869 is pending with a58-minute limit. All startup
failures are counted against the original resource envelope. The authoritative
attempt and cost record is
`outputs/research/global_throughput/server-cost-preparation/PAID_EXECUTION_LEDGER.json`.
This document defines the server matrix for resident scalar, persistent CPU
workers, JAX CPU batching, and the first complete GPU batch.

## Fixed Workload

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

The CPU compiler peak estimate is about 10 GiB for the one-core run and
40-60 GiB for four persistent workers plus the controller. These are planning
estimates, not measured server peaks. Do not request `--mem`: CPU-node
`RealMemory=1` is a broken scheduler value. Record measured peak RSS and
available host memory instead.

The CPU matrix loads the complete B4 numeric reference bundle for parity; it
does not compile another B4 executable. CPU1 still requires one resident
scalar compile and B16/B64 batch stages. CPU4 requires four resident scalar
worker compiles plus B16/B64 batch stages. The one-hour limit is therefore a
bounded ceiling with unknown completion probability, not a completion-time
estimate. If stages remain incomplete, keep their atomic outputs and report
the missing stages without requesting extra time.

Both CPU launchers currently constrain the request to `ibc11b02n25`. The GPU
launcher currently constrains it to `ibc13b03n03`, the lowest-load observed
node with an available card in the bounded scheduler screen. That GPU node was
mixed-use, not idle. These are reviewable node choices from a point-in-time
snapshot, not reservations or claims of current availability; recheck before
submission and update the node constraint if occupancy changes. Each one-hour
wall time is a hard diagnostic ceiling, not a prediction that every compile,
gate, and timing stage will finish. Preserve completed atomic stages and report
missing stages as incomplete; do not extend the wall time without a new
resource approval.

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

## Launcher Inputs

Set these variables before submission when the default paths do not match the
immutable Explore1000 checkout:

- `CODE_DIR`: clean, transferred code checkout containing this branch's
  benchmark entry points. The default in all three launchers is
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-v1`.
- `PREPARED_INPUT`: the frozen FULL64 pickle above.
- `GPU_REFERENCE_DIR`: the accepted complete FULL64 B4 numerical-reference
  tree used by the CPU gates. The current default is
  `runtime/outputs/research/global_throughput/gpu-round4/model`.
- `CPU_REFERENCE_DIR`: for the four-core CPU launcher, the `scalar` output
  directory produced by the one-core job.
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

## Approval-Time Output Paths

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

## Cost Bound

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
