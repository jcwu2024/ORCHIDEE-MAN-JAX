# CPU48 v4 child environment preparation

This is unpaid, nonmodel preparation. It does not admit a physical 48-core
runtime, approve a new paid job, validate the scientific outputs, or measure
candidate throughput.

## Defect and installed behavior

The immutable v3 source exported parent `OMP_NUM_THREADS=1`. Its coordinator
declared `--export=ALL,OMP_NUM_THREADS=48,...` but launched `srun` without a
Popen environment override. The environment dictionary in its report was an
intended policy, not an observed child environment. The actual v3 child OMP
value was not logged; the preserved traceback proves only that the pre-JAX
binding guard saw a value other than absent/`48`.

The installed Slurm 22.05.7 binary was separately exercised on free
`cntest/ibtest02`, one task and one CPU. An inherited diagnostic value won
over an explicit duplicate in `--export=ALL,key=value`. The exact command,
affinity mask `0x1`, output, installed documentation search limitations and
receipt hash are in [slurm-export-precedence.json](slurm-export-precedence.json).
That reproduction does not recover the historical v3 OMP value.

## Implementation

Production dispatch now normalizes the `srun` parent environment and executes
a shared stdlib child boundary after `srun`. The boundary removes inherited
OMP/MKL/OpenBLAS/NumExpr, XLA/JAX and TF thread configuration before applying
the exact declared policy. Both dry-run and production use this boundary.
Receipts distinguish inherited transport values from the actual applied child
whitelist, affinity, requested thread policy and JAX-not-imported state.

The runner checks that actual environment and the existing real Slurm and
48-distinct-physical-core admission before importing JAX/model code. Success
and failure produce a pre-JAX receipt. A correct environment does not bypass
allocation, physical-core, native-device or memory gates.

The v4 launcher is explicitly B64-only with a 3600-second parent window and
no scalar regeneration. It requires an external scalar admission record and
its exact byte hash. The coordinator validates that seal before dispatch; the
runner validates it again before JAX and the reference reader checks the
candidate/source identities. Invalid admission stops execution. The existing
legacy combined route remains available without silently becoming v4.

Mandatory candidate runtime comparisons remain in place. V4 additionally
seals actual permutation, replacement and frozen-reset arrays in one bounded
sidecar, including candidate identity and explicit index mappings. Stage
provenance binds its schema, path, byte count and SHA256 so the collector can
independently inspect the saved arrays.

## Local verification

Command from the fixed global worktree:

```text
conda run --no-capture-output -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_global_cpu_child_environment.py tests/unit/test_global_cpu48_runtime.py tests/unit/test_global_cpu48_b64_preparation.py tests/unit/test_global_server_benchmark.py tests/unit/test_global_server_cpu_launchers.py
```

[local-tests.stdout.txt](local-tests.stdout.txt) records 108 passed in 13.73s;
the exit status is 0 and stderr is empty. The one warning is the existing
unknown `cache_dir` setting when pytest's cache provider is disabled. Python
lint passed with exact owned paths recorded in the check invocation.

The new regressions use real subprocesses, including production
`run_process`/`coordinate` dispatch through an executing transport that injects
conflicting policy after Popen. They inspect the actual runner receipt, test
wrong-environment and no-allocation failures before JAX, test external-seal
failure without scalar regeneration, preserve mocked 48-core/SMT hard-gate
coverage, and verify the saved auxiliary array sidecar with synthetic
arithmetic only. Allocation observations in coordinator unit tests are
explicit unit mocks; they are not target admission evidence.

The standalone local process diagnostic is saved in
[local-processes/environment-diagnostic.json](local-processes/environment-diagnostic.json).
It imported no JAX and executed no model. Windows has no Linux affinity
evidence in this diagnostic; target checks must provide that separately.

## Target evidence

The restricted checks completed on free `cntest/ibtest03` with one task,
one available CPU and the canonical `orcjax_cpu` Python. The actual process
diagnostic is in
[target-processes/diagnostic/environment-diagnostic.json](target-processes/diagnostic/environment-diagnostic.json).
Scalar and batch children normalized an injected stale transport policy to
their requested thread budgets of 1 and 48, respectively; both reported no
unexpected policy keys and actual affinity `[0]`. The wrong-policy child
failed the environment guard; the correctly normalized child without a
paid physical allocation failed allocation admission. All four checks
reported JAX not imported and no model execution. The diagnostic exit status
is 0; its stderr is empty. A policy budget of 48 on one CPU demonstrates
normalization only, not effective parallelism or physical 48-core admission.

Launcher `bash -n`, launcher `--dry-run`, the emitted coordinator
`--dry-run`, and their one-CPU target transport all returned status 0; the
status and output files are in [target-shell](target-shell/). The reviewed
[coordinator.stdout.json](target-shell/coordinator.stdout.json) contains
only a B64 batch command, a 3600-second parent window, the fixed 1800-second
minimum candidate window, the post-transport child boundary, external scalar
and admission arguments, and `--export=ALL` without duplicate policy keys.
The runner's actual dry-run receipt again records affinity `[0]` and no
JAX/model activity.

These target-shell checks use explicitly labeled placeholder execution-pin
and admission hashes for CLI expansion only. The observer verified matching
local/remote bytes for the six staged sources and reported that the planned
checkout and dry-run output root remain absent. Three generated Python cache
files were removed from staging after the checks. These observations and
the original source/output hashes are preserved in
[target-check-receipt.json](target-check-receipt.json). This snapshot precedes
the final immutable-pin CLI checks below; no production deployment,
scientific acceptance, throughput result or paid submission is implied.
After this snapshot, the runner's informational dry-run billing label was
clarified to `full current parent allocation; actual Slurm admission required`.
Ruff passed on that changed file. The snapshot retains the original runner
bytes and label; the final checks below use the frozen final source.

## Final immutable-pin nonmodel checks

The final checks use raw blobs from
`codex/global-cpu48-b64-v4` at
`60dccbddde10be4295aab7e7fe55fa7864d3c3f3`. The compact record is
[CPU48_B64_V4_EXACT_PIN_CHECK_RECEIPT.json](exact-pin/CPU48_B64_V4_EXACT_PIN_CHECK_RECEIPT.json).
The primary free `cntest` job is `15852487`, one task and one allocated CPU,
with affinity `[0]`. Both the proposal and pinned launcher passed `bash -n`;
the actual launcher, transformed coordinator and runner dry-run succeeded,
as did the frozen saved-reference admission validator. All check exit files
are 0. The runner reports no JAX import or model execution. The raw allocation
log has an erroneous `host=1` line; retained `scontrol` NodeList/BatchHost
identify `ibtest03`. The receipt preserves this driver limitation.

The accepted launch invocation and combined completion output are in
[invocation-job-15852487.txt](exact-pin/invocation-job-15852487.txt).
The tool records exit 0 and 5.055 seconds, including allocation/transport.
The actual check body is retained in
[driver-stdin.sh](exact-pin/checks/job-15852487/driver-stdin.sh), the terminal
scheduler record in
[scheduler-accounting-job-15852487.txt](exact-pin/scheduler-accounting-job-15852487.txt),
and all 18 original log hashes in
[sha256.txt](exact-pin/checks/job-15852487/sha256.txt). The scheduler records
the parent allocation as completed with exit `0:0`, one CPU and four seconds;
its compute step took one second. All 18 local log files match the retained
hash list. These are check durations, not model throughput measurements.
The six-file
[source-transfer-sha256.json](exact-pin/source-transfer-sha256.json) records
the local raw-blob hashes and the earlier observed transfer comparison,
including the unchanged resources dependency outside the changed-harness
pin map. The same map includes the pin/proposal/admission artifact digests.
Their earlier remote matches were observed, but the original raw remote hash
output was not retained; packaging performed no independent remote recheck.
The frozen validator independently checked the staged admission checksum
against the exact declared seal.

Review the actual emitted and transformed arguments in
[coordinator-argv.json](exact-pin/checks/job-15852487/coordinator-argv.json),
the final expansion and pre-JAX child receipt in
[coordinator-dry-run.stdout.txt](exact-pin/checks/job-15852487/coordinator-dry-run.stdout.txt),
and the seal result in
[scalar-admission-validation.stdout.txt](exact-pin/checks/job-15852487/scalar-admission-validation.stdout.txt).
The B64-only command retains the 3600-second proposed parent window and fixed
1800-second minimum candidate window. The observed `SLURM_CPUS_PER_TASK=1`
was unset only for the launcher dry-run's planning guard; it was never set to
48. Actual one-CPU scheduler evidence is retained separately.

The launcher uses the isolated staging copy of the real admission bytes,
SHA256 `f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d`,
instead of the proposal's undeployed persistent admission destination. To
execute its emitted command without creating the future checkout, only the
coordinator script and `--entry` locations were subsequently replaced with
staged paths; the declared future code/config/result paths remain in the
arguments. This is scoped exact-pin CLI evidence, not literal execution of
every proposed path. The validator checked the real old scalar files against
the supplied sealed pin's declared candidate identity with `job_id=null`.
It did not compute `candidate_identity` from the absent future checkout or
repeat the independent numerical audit.

The prior unused dispatch `15852458`, driver failure `15852476` after its two
successful syntax checks, and successful extra-override run `15852482` remain
separate from the primary evidence. Their retained transcript excerpts and
capture limits are in
[failed-dispatch-tool-transcripts.txt](exact-pin/failed-dispatch-tool-transcripts.txt):
`15852458` has combined PTY tool output only, and `15852476` has no separately
saved stdin driver. The future checkout/result directory remain absent, and the
persistent admission destination was not written by these checks. No
production deployment, physical CPU48 admission, model compilation,
throughput measurement, GPU work, `sbatch` submission or supervisor contact
occurred. Actual future clean-checkout and paid-allocation admission remain
runtime requirements after authorized deployment.
