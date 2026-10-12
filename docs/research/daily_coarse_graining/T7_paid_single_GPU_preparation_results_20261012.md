# Single allocated-GPU paid preparation and proposal

Current execution status: the [terminal result](T7_paid_single_GPU_execution_results_20261012.md) records the later independent human approval, sole job15812634 FAILED2:0/6s and closed/consumed scope. No retry is authorized. The preparation-stage snapshot below remains historical; its original false scope is preserved.

2026-10-12. Ready for new live review; not execution approval. The new
[contract](T7_paid_single_GPU_execution_contract_20261012.md)
`T7_affine_allocated_single_gpu_v1` is explicit in scope and manifest. It requires
actual computation on the single allocated GPU; OS denial of opening other
GPU devices is independently recorded as `not_verified`. The old strict helper
default remains available. Old58a/f4ae contracts, deployments and consumed
grants remain unchanged; free b676 remains paused/approved/unconsumed.

## Frozen candidate and actual CPU evidence

Immutable code: `c8a44c3e5ba203f57da5f6b44aae47587af31d31`. Independent full-bundle clone at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v1` has the exact HEAD, clean status and no shared-object alternates.
New output directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v1`.
Scope `scope.v1.json` is proposed/false, SHA256
`59bd008466c9d01aedadd53e3342add952a204caa449ad7754a3045542bae4b5`;
source/deployment manifest SHA256 `539e6da0e0fa0cdb2811f00e1f37f60c017cdc90a264078a1fa61d282ddfa2a6`.
No old approval automatically migrates to this scope. Final documentation
versioning is separate from this immutable code commit.

Actual `ibtest01` CPU admission completed at `2026-10-12T12:30:27.066453+00:00`
using canonical `.venvs/orcjax_cpu/bin/python`. Real Bash Slurm `--dry-run`
expanded the exact submission and seven-path downstream command; executing
that downstream paid Python `--dry-run` exited0 in
0.907704s. Declared paid/project imports resolve to the
immutable checkout. Source/input/scope identities stayed unchanged, false
authorization and12measurements/91arrays/4small compiles were admitted, and
attempt/result/ledger/preflight/timeout/startup-identity files remain absent.
JAX was not imported by admission. No GPU command, driver initialization,
container/PJRT/preflight, affine science or scheduler submission occurred.
The new actual-affine identity gate therefore remains target-unexecuted.

Local private CPU fixtures:22 contract cases,28 affected cases,37 shell cases,
1 shared no-hook result/release regression:88 passes. These cover broad/control
wildcards with correct UUID/driver/PJRT, strict-default rejection, changed
allocation/kernel/UUID and multiGPU failures, explicit mandatory container
fields, actual affine rejection before upload/compile and result identity
admission/evidence retention. Initial missing-new-contract/old optional-hook
fixture failures are retained; correction changed fixtures, not hard gates.
Private system-TEMP fixtures removed; existing pytest cache_dir warning remains.
No scientific numerical probe was rerun.

## Concrete one-job request

Resources: fixed `ibc13b03n04`, gnall,1node/1task/4actual allocated CPUs/1V100,
20minutes, nonexclusive/no-requeue,24GiB peak gate. Startup admission and the
unchanged original affine12/91 comparison share this one allocation: preflight
240s, probe780s, protected release180s, total1200s. Actual allocation and cost
clock remain mandatory, followed by CSV/XML/kernel/job-cgroup binding,
container CVD/contract, independent driver count1/UUID, PJRT oneGPU/local0 in
both preflight and actual affine process, strict numerical/result and release
gates. Startup failure grants no retry or second paid request.

Additive worst case: CPU `4×(1/3)×0.07=0.0933333333` plus
GPU `1×(1/3)×2.20=0.7333333333`, totalCNY0.8266666667;
upward quoteCNY0.83, requested approval ceilingCNY1.00. Actual bill may differ.
Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v1`.
Submission command, only after a new explicit approval:

```bash
sbatch --partition gnall --nodes 1 --ntasks 1 --cpus-per-task 4 --gres gpu:1 --time 00:20:00 --no-requeue --nodelist ibc13b03n04 --chdir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v1 --output /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v1/slurm-%j.stdout.txt --error /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v1/slurm-%j.stderr.txt /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v1/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh
```

The command above is the exact tested expansion with explicit resource,
time, node, cwd and output/error flags. Review inherited `SBATCH_*` settings
immediately before any approved submission and remove unintended overrides
(including arrays or extra resources); an inherited environment must not
expand this single-job cost envelope before launcher admission. That review
has not been performed for a new live submission; no new remote check ran.

Stdout/stderr: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v1/slurm-%j.stdout.txt` and `slurm-%j.stderr.txt`;
result `result.v1.json`,91-array container `result.v1.results.npz`, own ledger
`T7-summary-affine-paid-single-gpu-20261012-v1.ledger.json`. Request binds this exact false scope,
immutable code, reused recovery inputs and explicit single-GPU contract.

The initial n01 metadata showed all8GPUs allocated. The bounded eligibility
snapshot at2026-10-12T12:21:16UTC showed n04 MIXED,CPUAlloc48/112,CPULoad4.21,
GPUAlloc3/8; the other eligible mixed candidate had load5.27. No idle eligible
node was observed. n04 was eligible and lower-load, not idle or reserved.
Refresh scheduler eligibility before any separately approved submission; no
automatic substitution, reservation, free fallback or polling is authorized.

## Preservation and limits

Old remote58a HEAD and job15806646 closed-scope SHA
`5b6b24bbb54c98879d74033e6d1b21ed9bc9d4e9f77dafe5ae14fa7092c077d4`
were checked read-only on test01. Local historical report/scope/CSV/XML/archive
anchors match prior preserved identities. Unique Teacher/data/weights/results
were not generated, removed or modified. No old deployed checkout was updated.
The actual old cgroup rules remain missing; `c *:255 rw` is hypothetical,
not an assertion about job15806646. CPU mocks/dry-runs do not establish actual
GPU readiness, OS isolation, CUDA/PJRT identity, affine success or T7 repair.

The [package](../../../outputs/T7-paid-single-gpu-contract-20261012/)
contains `proposal.v1.json`, false scope/manifest, exact scheduler snapshots,
test/commit/push receipts, full code/docs diffs, independent deployment and
actual target shell/parser stdout/stderr, preservation and final versioning.
No live scope opening, allocation, preflight, probe, training, paid charge,
configuration repair or background monitoring has been performed. Preparation
ends here pending review and new explicit paid execution authorization.
