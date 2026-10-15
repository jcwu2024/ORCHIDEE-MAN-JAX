# CPU scalar pool v3: closed terminal result

Job15869035: FAILED1:0 after4 parent seconds, cnall/ibc11b02n24,
4allocated CPUs. No JAX import or physical worker. The actual affinity
24,25,79,80 maps to three unique socket/core keys, so the physical-core hard
gate correctly stopped execution. Optional NVIDIA telemetry failure continued
through real srun as intended. No retry and no remaining paid allowance.

Full current report:
[CPU_SCALAR_POOL_PILOT_V3_RESULT.md](../../../../docs/research/global_throughput/CPU_SCALAR_POOL_PILOT_V3_RESULT.md).
Raw receipts and compact `result-analysis.json` are in this directory.
`integrity-verification.json` records final local/target byte equality.

All per-worker compile/cache/memory, complete760 strict/reset, first/warm and
model utilization metrics are **not reached**. No evidence establishes whether
48workers reside in192GB or provides CPU/GPU equal-workload throughput.

Full parent charge counted once:4×4/3600=0.004444444 core-hours;
unverified CNY0.000222222/0.000311111 at0.05/0.07 per allocated core-hour.
No historical job or step charge is added. A separate non-model binding and
billing diagnostic is the minimum next decision, and was not executed.
