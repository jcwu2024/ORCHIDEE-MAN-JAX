# CPU48 / FULL64 B64 Execution Result

**Outcome: no submission.** The approved execution checkout was imported, the
existing canonical input and reference were verified, and the target Bash
launcher plus coordinator/scalar/batch runner dry-runs passed on `ibtest03`.
The dry-runs imported no JAX and executed no model.

The final approved-node snapshot and three bounded-wait snapshots reported
`ibc12b08n35` as `IDLE+PLANNED` (`sinfo` state `plnd`), not plain `IDLE`.
The final observation was local `2026-10-11T14:55:35Z` and target-host
`2026-10-11T14:47:15Z`; CPUAlloc was 0/56, load 0.01, scheduler FreeMem
177983 MB, AllocTRES empty, and no queued job targeted the node. A visible
running job used other nodes. The planned state persisted through the bounded
wait window, so the exact-node admission condition was not met. The three
recorded snapshots span 243 seconds by the workstation clock and 242 seconds
by the target clock. Three 60-second sleeps completed; SSH/snapshot overhead
is included in those spans. The exact wait-dispatch time was not recorded, so
this does not claim an exact 300-second wait.

`sbatch` invocation count: **0**. Slurm job ID: none. The paid allocation was
not started; no parent accounting, model execution, performance result, or
collection was produced. The CNY3.60/CNY5.04 figures were unspent planning
ceilings. This authorization ended without an allocation; future execution
requires a new explicit assignment. No replacement node or second attempt is
authorized by this run.

The immutable execution commit is `4f6b2fc906c453b9030acff489637a8f83605dca`.
The prepared input SHA-256 is
`be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`; the
FULL64 reference checkpoint SHA-256 is
`a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`.

See [pre-submit-receipt.json](pre-submit-receipt.json),
[submission-attempt.json](submission-attempt.json), and
[AUTHORIZATION.md](AUTHORIZATION.md) for the pinned contract and timestamped
evidence. Remote deployment registry entry:
`latest_global_cpu48_b64_deployment`.
