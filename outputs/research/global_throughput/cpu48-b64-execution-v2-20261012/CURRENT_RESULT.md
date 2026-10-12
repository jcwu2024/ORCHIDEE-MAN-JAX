# CPU48 B64 v2 Execution Result

The one authorized v2 submission, job `15805735`, ended `FAILED` with exit
`1:0` after `00:01:49` on `ibc12b08n35`. The real allocation validation
passed: `cnall`, one node, one task, 48 CPUs, zero GPUs, with validated
`TRES=cpu=48,node=1,billing=48`. The exact terminal parent accounting row is
preserved in [terminal-parent.sacct.raw](terminal-parent.sacct.raw); the full
step rows are in [sacct-job-15805735.raw](sacct-job-15805735.raw).

The one-CPU scalar step reached `lower_compile` and failed in
`benchmark_global_batch.py` while `describe_aot_signature` called
`_aval_signature`. That helper accessed `.shape` on a Python `float`, raising
`AttributeError` before `.compile()` was called. The trace does not identify
which argument leaf contained the float. Lowering took `54.8616` seconds;
the benchmark process reported `63.1753` seconds. No stage, scientific gate,
warm sample, B64 batch step, throughput result, or collector result exists.
B64/48-core candidate memory and CPU utilization were not measured because
that step was never dispatched.

The full parent allocation is counted once at 48 CPUs for 109 seconds. At the
unverified planning rates of CNY `0.05` and `0.07` per CPU-count-hour, the cost
scenarios are CNY `0.072666667` and `0.101733333`; these are not verified bills.
The coordinator's lower-scope estimates (CNY `0.070324889` and `0.098454845`)
cover its observed interval only and exclude Slurm submission/startup and
terminal scheduler accounting.

This was the unique submission under the v2 authorization; the attempt is
closed and no retry is authorized. CPU48 throughput, numerical acceptance,
and a same-workload CPU/GPU ratio remain unavailable. The GPU B64 result for
job `15579869` remains the same-workload comparator but cannot be compared for
throughput without CPU measurements. The v1 failure record is unchanged.

Raw scheduler captures, the attempt marker, Slurm log, and scalar artifacts
are indexed with SHA-256 digests in [execution-result.json](execution-result.json).
The marker hash is
`70542ef71251c891781e2bccb839d3ca3a0d823c8cff8dc927aed08c91127469`; the
parent row hash is
`b1adac477661bdaeae9cf35de9ddb89136c9bc7d4b81815ba59b2fa95343f371`; and
the complete `sacct` output hash is
`19f421e6e8c8cf3c05bb7855d6e505c4dc559e5ccfbb744de5eb9ea37f532dc8`.
