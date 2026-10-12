# CPU48 / FULL64 continuation authorization

Disposition: the unique submission was job `15782454`, terminal
`FAILED / 2:0 / 00:00:01` during launcher startup. This single-job authorization
is consumed. See the [terminal result](CURRENT_RESULT.md); no retry is authorized.

The user explicitly replied “允许，继续” in supervisor chat
`01a128d2-3ec2-77a1-8d51-1665c220d2ca` on 2026-10-12 (Asia/Shanghai).
The supervisor delegated this continuation to the existing independent Sol
execution chat. This authorization supersedes the previous n35-only and
closed-session conditions for this assignment. The prior
[no-submission evidence](../cpu48-b64-execution-20261011/CURRENT_RESULT.md)
is preserved: zero sbatch invocations, no job and no paid allocation.

Authorized resources and ceilings remain unchanged:

- `cnall`, one node, one task, 48 allocated CPUs, zero GPUs, 5400 seconds;
  `--no-requeue`, nonexclusive, no new `--mem` reservation or overcommit.
- Select a verified healthy CPU node with matching hardware/resources,
  preferring low load; exclude DRAIN/DOWN and unsuitable hardware. The
  explicit sbatch node selection must override any old launcher default.
- `IDLE+PLANNED` and normal resource queueing are permitted. Normal PENDING
  is not a failure and does not authorize cancellation or resubmission.
- Maximum planning scenarios: `48 * 1.5 * 0.05 = CNY3.60` primary and
  `48 * 1.5 * 0.07 = CNY5.04` conservative. Rates are not a verified bill.
  The complete parent, including the scalar phase, is billed at 48 CPUs.
- Exactly one sbatch invocation. Recover an ambiguous submission through
  its persistent marker and scheduler records; never blindly retry.
  Failure or timeout closes the attempt, with no second submission,
  resource expansion, changed execution version or old-budget reuse.

Reuse immutable execution commit
`4f6b2fc906c453b9030acff489637a8f83605dca`, ref
`codex/global-cpu48-b64-v1`, and
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v1`.
Use the existing `scripts/hpc/slurm_global_server_cpu_48core.sh` and canonical
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`.
The only approved submission-command change is selecting an appropriate
matching node or scheduler node range instead of locking n35. Before submission,
display the actual command, cwd, resources, costs and node evidence in this
execution chat. This is a transparency record under approval already granted;
it does not require another permission question.

The [existing protocol](../cpu48-b64-preparation-20261011/PROTOCOL.md) and
[proposal](../cpu48-b64-preparation-20261011/PAID_PROPOSAL.md) remain the
scientific and measurement contract, with the node-rule change above:
science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`,
64 genuine FULL64 points, continuous Days2–8/9–15, 48 half-hour steps/day,
896 point-days, 760 complete native outputs/masks, original dtypes and
`atol=1e-8, rtol=1e-10`. x64 is not a claim that every expression is FP64.
Scalar reference is capped at 2700 seconds; batch admission requires at least
1800 seconds left. Each phase retains five core and five wrapper samples.
The candidate is one native CPU JAX device/process using B64 vmap and 48
distinct physical cores. Finite existing cross-GPU tolerance differences may
retain exploratory timing; CPU-internal whole-tree, permutation, replacement,
isolation, reset, structure and nonfinite gates remain hard.

Authorized work includes node selection, the unique submission, autonomous
monitoring through queue and terminal state, preservation of arrays/logs/failure
evidence and the exact terminal parent sacct row, restricted diagnostic-node
collection, necessary light harness/collection repairs within the unchanged
contract, compact current documents/receipts and research commit/push. Login
nodes are transfer/scheduler surfaces; computation uses allocated compute nodes
and existing restricted test-node checks/collection. Reuse completed deployment,
preflight and regression evidence; do not recompile merely for this authorization
or node-argument change.

The existing execution owner handles remote execution and collection. Root owns
this authorization, actual-artifact acceptance, current entry documents and
final research delivery. Others share the codebase; preserve their changes,
main, other branches, historical checkouts, shared objects, data and environments.
No supervisor polling/messages, new chat, heartbeat or automation is part of
this assignment. The result is delivered in this execution chat.

Local admission passed: fixed global cwd/top-level, branch
`research/global-throughput`, clean research HEAD
`44294a621f07ba6050c6b4b934cca9de7fceea0f`, and actual ORCJAX module import
from this global worktree.
