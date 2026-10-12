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
The test fixture's `scontrol` record is synthetic and explicitly does not
represent a new allocation. Dry-run coverage confirms argument expansion and
that the runner dry-runs do not execute the model or import JAX. These local
checks do not validate the deployed filesystem, Slurm environment, or actual
launcher startup on Explore1000.

Before the one authorized paid submission, deploy the reviewed immutable pin
and exercise the non-dry launcher on a restricted test node through saved raw
record capture and the real coordinator `--validate-allocation` branch. Use a
controlled stub after validation to prevent `srun` and model execution. Preserve
that target-side result. The target exercise and paid submission have not yet
occurred; the paid submission must remain exactly one `sbatch` invocation.

Candidate repair files for the review-stage commit:

- `scripts/dev/prepare_global_cpu48_execution.py`
- `scripts/hpc/global_cpu48_b64_coordinator.py`
- `scripts/hpc/slurm_global_server_cpu_48core.sh`
- `tests/unit/test_global_cpu48_b64_preparation.py`

The preparation helper's clean-source guard remains active. Package v2 only
after these candidate changes are reviewed and committed, then use the
committed source SHA and the declared historical base.
