# CPU48 / B64 Preparation Protocol

This directory records local preparation for one exploratory CPU throughput
measurement. It does not authorize an allocation or submission and contains no
physical-model result.

The proposed execution is one `cnall` node, one task, 48 allocated CPUs, no GPU,
and a 90-minute parent wall limit. The coordinator runs one scalar FULL64
reference first, capped at 45 minutes, then starts one single-process native
CPU B64 `vmap` only when at least 30 minutes remain. Both phases use the same
64 unique landpoints, two seven-day blocks, 48 half-hourly steps per day,
full outputs, fixed `atol=1e-8` and `rtol=1e-10`, and five synchronized core
and wrapper samples. The scalar phase is an internal comparison; the candidate
must also be recomputed against the pinned GPU reference before any performance
summary is emitted.

The batch thread policy is Eigen multithreading enabled, `OMP_NUM_THREADS=48`,
and MKL/OpenBLAS/NumExpr set to one. The execution entry checks a single native
CPU JAX device, x64 mode, actual 48-CPU parent `AllocTRES`, physical-core
selection, per-phase memory observations, thread affinity, and process CPU
samples. Runtime telemetry describes the observed run; it does not establish
48 effective threads or scientific acceptance.

After execution, `collect_global_cpu48_b64.py` verifies sealed scalar and batch
checkpoints, recomputes the 64 GPU-reference gates and full CPU-internal
cross-mode comparison from saved arrays, validates the complete output schema
and runtime telemetry, and reports coordinator-scope costs separately from
whole-parent cost. Whole-parent cost remains pending until terminal `sacct` or
raw9 accounting is supplied. Performance findings remain exploratory and do
not establish CPU scientific acceptance, annual/Fortran speedup, or a 10x gain.

Terminal accounting is supplied with `--parent-accounting`. Capture the actual
parent row after completion with:

```bash
"$SLURM_BIN/sacct" -n -P -j "$JOB_ID" --format=JobIDRaw,State,Elapsed,AllocTRES
```

Select the row whose `JobIDRaw` exactly equals the parent `$JOB_ID`; do not use a
`.batch` or `.extern` step row. Preserve that exact four-field pipe-delimited
line, including its final newline, in `raw_record` and include its SHA-256 in
`raw_record_sha256`. The JSON input also carries `source` (`sacct` or `raw9`),
`scope` (`whole_slurm_parent_job`), `job_id`, `terminal_status`, `Elapsed`,
`AllocTRES`, parsed `wall_seconds`, `allocated_CPU_count` (48), `GPU_count` (0),
and `directories` containing the exact scalar and batch run directories. The
collector binds the row's job ID to `orchestration.json`, requires `Elapsed` to
match `wall_seconds`, rejects duplicate or non-48 CPU TRES and any positive GPU
TRES, and checks the preserved row and file hashes. For example:

```json
{
  "source": "sacct",
  "scope": "whole_slurm_parent_job",
  "job_id": "123456",
  "terminal_status": "COMPLETED",
  "Elapsed": "00:23:41",
  "wall_seconds": 1421,
  "AllocTRES": "cpu=48,mem=192G",
  "allocated_CPU_count": 48,
  "GPU_count": 0,
  "directories": ["/path/to/run/scalar", "/path/to/run/batch"],
  "raw_record": "123456|COMPLETED|00:23:41|cpu=48,mem=192G\n",
  "raw_record_sha256": "<sha256 of the exact UTF-8 raw_record>"
}
```

Before any future submission, recheck node occupancy and host CPU load, review
the exact execution commit and transfer prerequisites, provide a new output
root, inspect final scheduler argument expansion, and obtain the required
resource, wall-time, command, directory, and worst-case-cost approval.

## Local Validation Boundary

The preparation harness uses synthetic metadata and arrays only. The focused
local tests cover launcher/coordinator argument expansion, 48-CPU admission,
the scalar-to-batch timing window, runner resource telemetry, parent billing,
and internal reference identity checks. Reference-reader tests exercise JSON
tuple/list normalization for `loadavg` and reject tampering in scientific
identity, gates, timing, provenance, and runtime fields. A synthetic sealed
scalar/batch/FULL64 fixture exercises the successful collector CLI path,
including all 760 output leaves, 64-point identity, five timing and CPU samples
per metric, and rejection of changed CPU-internal arrays and corrupted
accounting. No model compilation or execution is part of these checks.
