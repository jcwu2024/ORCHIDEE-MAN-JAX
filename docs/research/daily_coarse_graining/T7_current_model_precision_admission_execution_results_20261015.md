# Current-model T7 precision/FD execution outcome

Updated: 2026-10-15. **Job15866925 FAILED1:0 before GPU discovery or model execution.**
The single authorized attempt has been used; this precision/FD diagnostic line
is stopped. Local launcher repair and CPU tests do not restore GPU authorization.
No resubmission, new proposal, deployment, GPU work, training, automation or
watching follows.

## Actual terminal evidence

Captured sacct reports job15866925 and its batch step FAILED1:0, elapsed8 seconds,
start2026-10-15T13:33:33, end13:33:41, node ibc13b02n03, four allocated CPUs,
AllocTRES billing=4,cpu=4,gres/gpu=1,node=1. The extern step separately reports
COMPLETED0:0, elapsed9 seconds, end13:33:42; it does not change the job failure.

The2099-byte stderr preserves the actual chain read_live_allocation →
validate_allocation → _allocated_tres → paid.allocated_tres, ending at line194
with ValueError: legacy TRES lacks actual single-node allocation detail.
The current launcher checks RUNNING and NodeList↔hostname, but its downstream
legacy allocation helper retained the old task's fixed NODE ibc13b03n04.
That static node coupling conflicts with current automatic node selection;
sacct independently establishes actual allocated node ibc13b02n03.

The actual RUNNING scontrol response was not saved. Terminal scontrol returned
Invalid job id specified after purge. The exact lost Nodes fields are not
reconstructed: sacct and synthetic CPU fixtures are not live-response evidence.
The preserved stack, static helper defect and actual allocation node have
distinct evidence roles.

The captured output inventory contains approval.v2.json, server-preparation,
zero-byte stdout and stderr only: no result, supervisor, ledger, discovery or
preflight artifact. Failure was before discovery/preflight/model compilation/
model calls. The intended52-call diagnostic did not complete. This is a launcher
failure, not numerical evidence against FP64 or the model.

## Approved identity and historical submission

The executed source remains untouched at
30d7eb7675485fe3ce6bc3d877851feaea9854d6 in
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-20261014-v2.
The historical exact proposal SHA256 remains
59b09320d19470d2486ec7efe0e654cc749536a6fe2227debeba378b6ff2aa0f.

The approved existing66,123,713-parameter latent32/private-reader64 model was
B1/artifact299/window0/T7 public loss: one V100, four Slurm CPUs,5400 seconds,
24GiB/CNY3.72 maximum, at most one real FP64 compile and52 calls
(16 native state-VJP/JVP and36 scalar FD), no retry/training.
Directions434,2382,2621/control200 were planned;2621 is computational-coordinate
only, without a physical-manifold claim.

The exact approved argv was submitted once from gln01 without node pinning:
sbatch exit0, raw stdout `Submitted batch job 15866925`; first snapshot PENDING(Priority),
Requeue=0, OverSubscribe=OK. Queue time was outside the allocation clock.
The prior predicted start was only an estimate and is superseded by sacct actual
start/end. We incorrectly treated busy GPUs as a submission block before the
user clarified normal Slurm queueing was allowed; that historical closure was
wrong and did not revoke approval. It is separate from this actual terminal stop.

## Measurements, accounting and release limits

| Direction | New native FP64 JVP/VJP | Repeat/full3854/physics gates | Independent FD |
|---|---|---|---|
| 434 | Unmeasured | Unmeasured | Not executed |
| 2382 | Unmeasured | Unmeasured | Not executed |
| 2621 | Unmeasured | Unmeasured | Not executed |
| control200 | Unmeasured | Unmeasured | Not executed |

New precision/resource/model gates remain unmeasured. Saved FP32 projections
and conditional scalar-resolution bounds remain preparation evidence. Strict
state20 nonfinite failure remains separately unresolved, with no fill/exemption.
No derivative reference, precision adoption, B4 repair, complete-update speed
gain or training acceptance follows.

The8-second job allocation accounts for0.002222222222 GPU card-hours and
0.008888888889 allocated CPU-hours; estimated CNY0.005511111111, actual bill
unknown. The separate9-second extern record is retained without double counting.
These are scheduler allocation seconds, not seconds of GPU model execution.
Slurm is terminal; no owned GPU children started. No GPU UUID/process-memory
sample was taken, so no UUID-based release-pass claim is made.

## Local repair and CPU validation

The local launcher now passes its verified actual node to paid.allocated_tres;
the helper's omitted-argument default preserves the old fixed node for legacy
callers. The launcher also saves scheduler command/hostname/return code/raw
stdout/stderr before allocation validation can raise, retaining the actual error.
These patches are local and were not deployed to the immutable execution checkout.

The scoped CPU suite passed **144 tests in2.03 seconds**, with one existing
pytest unknown-cache_dir warning when its cache provider is disabled. The exact
command in local-validation-receipt.json is:

    conda run -n ORCJAX python -B -m pytest -p no:cacheprovider tests/unit/test_run_shared_current_T7_precision_admission.py tests/unit/test_shared_latent_T7_batch_numerics_paid.py

Reconstructed legacy-TRES fixtures exercise the real current→paid helper without
mocking that helper, accepting ibc13b02n03 and ibc13b03n04 allocations while the
legacy default rejects a wrong node. Regressions reject CPU/GPU/node counts,
hostname/Nodes conflicts, requested-only TRES, missing/invalid CPU IDs and absent
or mismatched allocated GPU detail. A reconstructed failure fixture verifies raw
diagnostics survive validation failure; it is not the lost actual RUNNING response.

Bounded current→paid.PaidSupervisor→free.Supervisor review found no additional
legacy time/scope/output dependency in the current path. The current plan uses
5400 seconds/180-second release reserve, a shared240-second discovery/preflight
deadline and probe cap≤4980 seconds. Outputs/ledger derive from the current
invocation; the current path bypasses old paid.run_live's fixed1200-second
scope/output/ledger. No further source change was needed for those concerns.
These are local CPU/startup results, not GPU or gradient admission, and restore
no execution authorization.

## Evidence

Actual terminal captures are under
outputs/T7-current-model-precision-admission-execution-20261015/terminal/:
collection-manifest.json, sacct.raw.stdout.txt,
scontrol-terminal.raw.stderr.txt, output-inventory.raw.stdout.txt and
slurm-15866925.stderr.txt / slurm-15866925.stdout.txt,
terminal-summary.json and local-validation-receipt.json.
The manifest records raw capture hashes and the unsaved RUNNING-response limit.

Submission/approval history remains under the sibling queued-execution/ directory:
submission.raw.stdout.txt, first-scheduler-snapshot.raw.stdout.txt,
submission-receipt.json and copied official approval. Earlier resource-blocked/
receipts remain unchanged as transient availability evidence, with their mistaken
terminal interpretation superseded. Accepted preparation remains CPU/source
evidence in outputs/T7-current-model-precision-admission-preparation-20261014/v2/server-preparation/.
Original assets/results are preserved. [NEXT_STEPS](../../NEXT_STEPS.md) owns
the current stop boundary.
