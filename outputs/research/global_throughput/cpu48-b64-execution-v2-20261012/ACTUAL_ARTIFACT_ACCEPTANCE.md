# CPU48/B64 v2 actual-artifact acceptance

Date: 2026-10-12. The repair and the evidence of one failed execution are
accepted. CPU48 performance and scientific acceptance remain unavailable.
This is the terminal delivery of the single-run grant, with no retry.

The lead confirmed the fixed global worktree, `research/global-throughput`,
and ORCJAX's actual `jax_orchidee` import from this checkout. The reviewed
repair is research commit `72a983fc3d7352e5bec0cc7266df87568e60cb84`.
Execution used `b1b5cf0835da298bddb388e61d754feeb6c14cf2` at
`codex/global-cpu48-b64-v2`, in the declared immutable v2 server checkout.
The package and deployment identities retain all 104 physical-source files
byte-identical to base `4f6b2fc906c453b9030acff489637a8f83605dca`.

The earlier accepted checks were 22 focused local tests and six unique target
GNU Bash fixture cases across seven launcher invocations. The two valid
synthetic invocations reached the controlled srun stub; the five rejection
cases stopped before dispatch. Those checks created no real allocation or
model execution. Their scope is separate from the actual job below.

The lead read the real saved scontrol stdout, stderr and exit status, the
submission marker/captures, orchestration, scalar log/report, empty stage
manifest and exact terminal accounting. The files indexed in
[terminal-evidence-manifest.json](terminal-evidence-manifest.json) were checked
against their SHA-256 digests. All 18 manifest files and five remote result
receipts (22 unique files) also matched the bytes staged in Git; JSON, scope,
resource and parent-cost checks passed. The compact verification record is
[git-artifact-verification.json](git-artifact-verification.json). The persisted pre-call marker hash remains
`70542ef71251c891781e2bccb839d3ca3a0d823c8cff8dc927aed08c91127469`.
The submission stdout is `Submitted batch job 15805735`, with status 0 and
empty stderr. The real RUNNING record identifies one cnall node/task,
48 CPUs and zero GPUs, using `TRES=cpu=48,node=1,billing=48`.

Job `15805735` ran on `ibc12b08n35` and ended `FAILED / 1:0 / 00:01:49`.
The scalar step used CPU affinity `[0]`; its report's load was
`[0.04, 0.03, 0.05]`. The coordinator's separate startup load was
`[0.08, 0.03, 0.05]`, with 48 CPUs in its affinity mask.

| Observed quantity | Value and scope |
| --- | --- |
| Scalar lowering | 54.8616076009348 seconds |
| Recorded benchmark interval | 63.17527768295258 seconds |
| Scalar srun interval | 105.38788199424744 seconds |
| GNU time process wall | 104.86 seconds, including process setup |
| GNU time CPU | User 39.45 seconds + system 27.01 seconds; 63% for this scalar process |
| Sampled scalar peak RSS | 1,791,643,648 bytes |
| Linux/GNU time scalar ru_maxrss | 1,811,177,472 bytes, about 1.69 GiB |
| Terminal parent elapsed | 109 seconds at 48 allocated CPUs; 5,232 CPU-count-seconds |

The scalar traceback matches the exact execution version:
`scripts/dev/benchmark_global_batch.py` lines 437, 443 and 518.
`describe_aot_signature` passes caller leaves to `_aval_signature`, which
reads `.shape` from a Python `float`. `lower_compile` calls this descriptor
before signature admission, IR capture or `.compile()`. The offending argument
leaf was not identified in saved evidence. The lead used exact historical
source after graph search returned no match and coverage marked this worktree
excluded (main graph generation `2026-10-10T08:46:50Z`); no graph completeness
claim is made.

The scalar report and stage manifest both have `stages={}`; the report has
`gates=[]`. Only the scalar step was dispatched. There is no completed scalar
reference, B64/48-core execution, compile/first-call/warm timing, scientific
gate, saved-output stage or independent collector result. B64 throughput,
utilization, RSS, warm unit cost, cold unit cost and CPU/GPU speed ratios are
unavailable. The observed scalar RSS and CPU percentage do not measure B64.
The Slurm `.batch` accounting row is the script step, not a B64 candidate.

The exact parent row hash is
`b1adac477661bdaeae9cf35de9ddb89136c9bc7d4b81815ba59b2fa95343f371`.
Count that parent once: `48 * 109 / 3600 * 0.05 = CNY0.072666667` primary,
or `48 * 109 / 3600 * 0.07 = CNY0.101733333` conservative. Rates are unverified.
The coordinator's 105.487334-second interval and its cost estimates cover a
smaller interval and do not replace terminal Slurm accounting.

The accepted GPU B64 job `15579869` remains the 896-point-day comparator;
GPU B1024 job `15671732` has different total work. Neither produces a new
speed ratio without CPU48 measurements. The old v1 and zero-submission
records remain unchanged. Final server registration is documented in
[registry-update-receipt.json](registry-update-receipt.json), preserving the
earlier execution and deployment entries. Its verified final registry hash is
`c83ed058fe526a65ad2568bad219d0f08378a0471d4fb619e9a7c3ffd2dfd698`.
