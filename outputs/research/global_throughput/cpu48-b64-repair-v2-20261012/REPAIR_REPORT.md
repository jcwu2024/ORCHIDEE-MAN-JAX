# CPU48/B64 v2 Launcher Repair

The v2 repair makes the launcher preserve the raw `scontrol show job -o`
stdout, stderr, and exit code for the active allocation. The coordinator checks
that record against the Slurm job environment, requires one RUNNING `cnall`
job with one node/task and 48 CPUs, and records which TRES field supplied the
validated allocation. When `AllocTRES` and `TRES` are both present, any shared
resource key with conflicting values is rejected. `TRES` is accepted as a
fallback only when the record is for a RUNNING job. The launcher uses the v2
checkout and output namespace.

The preserved failure was job `15782454`. A later read-only query of that
purged job produced empty `scontrol` stdout, stderr
`slurm_load_jobs error: Invalid job id specified`, and exit status 1. This is
not the output of the original v1 startup command: its `scontrol`
stdout/stderr/exit status were not retained. Slurm no longer retains the job,
so its original `scontrol` record cannot be recovered. The prior execution
report says the launcher saw no `AllocTRES`; that does not establish why the
field was unavailable. The failed v1 record and checkout remain unchanged.

Read-only control snapshots show no job in the captured current-user queue and
one unrelated RUNNING `cnall` allocation, job `15795486`, with 200 CPUs on 40
nodes. Its saved `scontrol` record has `TRES=cpu=200,node=40,billing=200`;
`sacct` reports matching `AllocTRES` and a different `ReqTRES`. This confirms
only the captured command formats and field distinction. It is not evidence
about job `15782454`, an idle-node reservation, or the future CPU48 allocation.
The snapshots are point-in-time observations, not a resource reservation.

Local validation passed: 22 focused tests, including Bash-to-coordinator-to-
runner dry-run expansion; `bash -n` passed; and `git diff --check` exited 0.
The reviewed immutable pin was deployed at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v2`
on `ibtest01` through the restricted `test01` route. The detached checkout is
clean at `b1b5cf0835da298bddb388e61d754feeb6c14cf2`, has no object alternates,
and resolves `jax_orchidee` from that checkout. Target Bash is GNU 4.2.46 and
the canonical CPU Python is 3.11.15. The bundle SHA-256 is
`42de12b0262eb721666ce7d5da9fb493c832eaa20a1cea0124c03a3a2162e2f9`; the
server mirror ref resolves to the same execution commit. Full deployment
identity is in [deployment-receipt.json](deployment-receipt.json).

The target-side non-dry launcher check used fixture-backed `scontrol` and
synthetic `SLURM_*` variables on `ibtest01`; it did not create or inspect a real
allocation. Six unique cases were exercised across seven launcher invocations
because valid TRES fallback was repeated once. Both valid cases selected
`TRES=cpu=48,node=1,billing=48`, reported the expected one-node/one-task/48-CPU
coordinator allocation, and reached the controlled `srun` stub. The stub
returned 97 intentionally, so the launcher returned 1. Command failure,
missing fields, duplicate fields, CPU mismatch, and conflicting TRES each
returned launcher status 2 and did not invoke the `srun` stub. All raw
launcher captures and saved `scontrol` stdout/stderr/exit-code triplets are
preserved under `target-validation/`; coordinator JSON and stub logs are under
`target-validation/coordinator-runs/`. No model, JAX, or Slurm job ran.

The target check proves the deployed launcher reaches and enforces the
allocation-validation branch for the saved formats. Real allocation validation
was subsequently verified by the unique paid job recorded below. That
submission retained `cnall`, one node/task, 48 CPUs, zero GPUs and a 5400-second
limit. The one-run grant is consumed; no retry or second `sbatch` call is allowed.

Reviewed repair files in research commit
`72a983fc3d7352e5bec0cc7266df87568e60cb84`:

- `scripts/dev/prepare_global_cpu48_execution.py`
- `scripts/hpc/global_cpu48_b64_coordinator.py`
- `scripts/hpc/slurm_global_server_cpu_48core.sh`
- `tests/unit/test_global_cpu48_b64_preparation.py`

The preparation helper's clean-source guard remains active. The immutable
execution package is based on `4f6b2fc906c453b9030acff489637a8f83605dca` and
contains only the four reviewed harness/helper/test paths listed above; its
certificate confirms all 104 `jax_orchidee` files are byte-identical to that
base.

The unique authorized paid submission was job `15805735`. Its real allocation
passed validation on `ibc12b08n35` (`cnall`, one node/task, 48 CPUs, zero GPUs;
`TRES=cpu=48,node=1,billing=48`), then ended `FAILED`, exit `1:0`, elapsed
`00:01:49`. The scalar step reached AOT lowering but failed in
`benchmark_global_batch.py`: `_aval_signature` accessed `.shape` on a Python
`float` during `describe_aot_signature`, before `.compile()`. The saved trace
does not identify the offending argument leaf. No stage, scientific gate,
warm sample or batch step completed. Exact terminal accounting and hashed
captures are indexed in the [v2 execution result](../cpu48-b64-execution-v2-20261012/execution-result.json).
The authorization is consumed; this failure does not permit a retry.
