# CPU48 v4 unpaid repair and preparation scope

Current decision (2026-10-14): the supervisor accepted this preparation and
approved its exact fixed proposal for one deployment and paid submission under
the user's existing delegated authority. That sole attempt, job15853683, ended
FAILED1:0/50s and is closed. Read the separate
[terminal result](../cpu48-b64-execution-v4-20261014/CURRENT_RESULT.md) and
[execution authorization](../cpu48-b64-execution-v4-20261014/AUTHORIZATION.md).
The preparation records below remain historical; no paid grant is restored.

On 2026-10-14 (Asia/Shanghai), the supervisor dispatched this work after the
user said “怎么样了。继续。” and had authorized supervisor review and approval
of suitable new jobs. Dispatch source: `01a128d2-3ec2-77a1-8d51-1665c220d2ca`.

This stage authorizes the coordinator/resource-entry repair, real subprocess
regressions, independent admission of the saved v3 scalar arrays, necessary
reference/collector changes, and a new immutable execution pin, bundle and
concrete single-job proposal. Small process and saved-array checks may run on
restricted `test01`–`test04` CPUs. Login nodes are metadata/transfer surfaces.

This stage performs no `sbatch`, GPU execution, full physical compilation,
new production deployment, environment modification or physical-source change.
The v3 job `15820467` consumed its grant. A new proposal is submitted only after
the supervisor reviews its exact artifacts and approves it under the user's
existing delegated authority; the same proposal need not be referred back to
the user for duplicate approval.

The new code must start from execution pin
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`, with all 104 `jax_orchidee` files
and tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba` unchanged. Preserve science
SHA `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`, native
dtypes, all 760 outputs/masks and `atol=1e-8`, `rtol=1e-10`.

Prefer B64 alone, with the independently admitted scalar from old job
`15820467`; never silently regenerate it. Keep the three finite cross-GPU
failures and strict scientific status false. Preserve old data and identities.
The next proposal is one node/task, 48 CPUs, zero GPUs, no requeue, no exclusive
allocation, no new memory request or placement restriction. Its wall limit and
worst-case conservative charge must not exceed 5400 seconds and CNY 5.04.

Deliver in this execution chat, then stop and await dispatch. Do not contact or
poll the supervisor, create a chat, heartbeat or automation, or submit a retry.
