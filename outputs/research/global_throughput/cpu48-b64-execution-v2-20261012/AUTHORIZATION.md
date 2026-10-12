# CPU48 / FULL64 v2 repair and single-run authorization

The user replied “好的，继续” in supervisor chat
`01a128d2-3ec2-77a1-8d51-1665c220d2ca` on 2026-10-12 (Asia/Shanghai),
approving resource-field diagnosis and repair, preservation of raw command
errors, focused validation, and one new same-resource paid run after validation.
The supervisor delegated the complete work to this existing independent Sol
execution chat. This is a new explicit grant; the previous attempt remains
closed and its [failure evidence](../cpu48-b64-execution-20261012/CURRENT_RESULT.md)
must remain unchanged. No further confirmation is needed for this same plan.

## Scope and version

First collect bounded read-only Slurm command stdout, stderr and exit status.
The previous job's log alone does not prove why its `AllocTRES` read failed.
Implement only the necessary startup/resource repair and direct tests. Preserve
the raw command evidence on success and failure; fail closed on command errors,
missing/ambiguous records or incompatible allocated resources. Never substitute
requested resources, a default CPU count or fabricated TRES for an allocation.
Any supported legacy field must have observed and source-backed allocation
semantics, and its actual source must remain recorded.

Validate the real launcher resource-read-to-`--validate-allocation` branch,
including the observed success format and failure/rejection paths, under the
target GNU Bash. Synthetic/saved-output validation is not a new allocation
measurement. Small non-model checks may use existing restricted test01–test04;
login nodes are only metadata, scheduler and transfer surfaces. Reuse completed
unchanged scientific checks and environments; no full physical compilation is
required to repair logging or allocation parsing.

Build a new immutable execution version from
`4f6b2fc906c453b9030acff489637a8f83605dca`, with only the reviewed startup and
resource patch plus necessary helper/test changes. Prove the `jax_orchidee`
physical subtree is identical to that base; science remains
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`.
The research branch's carbon fixes remain separate. Record the complete new
SHA/ref in the [v2 repair evidence](../cpu48-b64-repair-v2-20261012/).
Deploy into a new `runtime/worktrees/global-throughput-cpu48-b64-v2` checkout;
preserve v1, its outputs, shared Git objects, data and canonical environments.

## One new allocation

- `cnall`, one node, one task, 48 allocated CPUs, zero GPUs, 5400 seconds.
  `--no-requeue`, nonexclusive, no new `--mem` reservation or overcommit.
- Select a verified healthy, matching CPU node, preferring low load. Normal
  PENDING and `IDLE+PLANNED` are allowed; n35 is not required. Exclude
  DRAIN/DOWN or incompatible nodes. A snapshot is not a reservation.
- Maximum planning scenarios: `48 * 1.5 * 0.05 = CNY3.60` primary and
  `48 * 1.5 * 0.07 = CNY5.04` conservative. These are unverified rates.
  The scalar phase also occupies the 48-CPU parent; count the complete parent
  once, including failure, using its exact terminal accounting row.
- Record the actual full command, cwd, new version, resources, rate scenarios,
  costs and node evidence in this execution chat before the unique submission.
  This is transparency under the existing approval, not another approval gate.
- Exactly one `sbatch` invocation. Persist its attempt marker first and recover
  ambiguous return through the marker and scheduler. Failure/timeout closes
  this attempt; no automatic second submission, requeue or old-budget reuse.

The [original protocol](../cpu48-b64-preparation-20261011/PROTOCOL.md) and
[proposal](../cpu48-b64-preparation-20261011/PAID_PROPOSAL.md) retain the scientific
and measurement contract, with the new reviewed execution version and paths:
64 unique FULL64 points, continuous Days2–8 then Days9–15, 48 half-hour steps/day,
896 point-days, 760 complete native outputs/masks, original dtypes and physics,
`atol=1e-8, rtol=1e-10`. JAX x64 does not imply every expression is FP64.
One native CPU JAX process/device uses B64 vmap over 48 distinct physical cores.
The scalar reference is capped at 2700 seconds; batch admission requires at
least 1800 seconds left. Each phase retains five core and five wrapper samples.
CPU-internal whole-tree, permutation, replacement/isolation, reset, structural
and nonfinite gates stay hard. Finite cross-GPU differences can retain
exploratory timing without scientific acceptance. No GPU rerun, precision,
physics, tolerance, ABI or intermediate-batch research change is authorized.

Preserve logs, arrays and the exact terminal parent sacct row and hashes.
Use the independent saved-array collector on a restricted diagnostic CPU node
only when real stage artifacts exist. GPU B64 job15579869 is the same896-point-day
comparator; GPU B1024 job15671732 is a different-total-work reference.
Report unavailable measurements explicitly. Finish compact current documents,
actual-artifact acceptance, and commit/push `research/global-throughput` here.
No supervisor messages/polling, new chat, heartbeat or automation is requested.

The execution owner handles code, focused checks, version pin, deployment,
single-run monitoring/collection and result receipts. Root owns this
authorization, current entry documents, actual-artifact acceptance and final
research delivery. Other agents share the workspace: preserve their changes,
main and other fixed research worktrees. Local admission passed at
`a3b618e848c7c8ede53d68209227780e5d204c10`: correct global cwd/top-level,
`research/global-throughput`, clean tree and actual ORCJAX import from this tree.
