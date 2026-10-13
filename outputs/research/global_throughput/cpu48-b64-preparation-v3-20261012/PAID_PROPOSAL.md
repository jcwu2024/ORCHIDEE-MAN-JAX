# CPU48 / FULL64 B64 v3 single-run proposal

**Approval state: the one authorized attempt is terminal and consumed (2026-10-13 Asia/Shanghai).**
The user delegated approval to the supervisor, who reviewed the actual artifacts
and approved this exact version, command, deployment and single allocation.
See the [authorization](../cpu48-b64-execution-v3-20261012/AUTHORIZATION.md)
and [terminal result](../cpu48-b64-execution-v3-20261012/CURRENT_RESULT.md).
Job15820467 ended FAILED1:0/2197s: the scalar phase finished, then B64 failed
the inherited OMP thread-budget guard before lowering/compilation or timing.
Deployment and pre-submit receipts are retained. No retry or CPU48 acceptance follows.

Partition `cnall`; one node, one task, 48 allocated CPUs, zero GPUs; 5400-second
wall limit; no requeue; no node pin, node exclusion, exclusive allocation,
overcommit, or added memory request. Work directory for submission:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. The script runs the immutable v3
checkout and writes only below the declared CPU48/B64 v3 runtime output root.

Worst-case requested-time scenarios, counting the whole parent allocation once:

- Primary: `1 node * 48 CPUs * 1.5 hours * CNY 0.05/core-hour = CNY 3.60`.
- Conservative: `1 node * 48 CPUs * 1.5 hours * CNY 0.07/core-hour = CNY 5.04`.

Rates and additive billing treatment are unverified. These are ceilings, not
expected spend. No GPU charge applies to this CPU-only proposal.

After deploying and verifying commit
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`, and after fresh scheduler and
occupancy inspection, the exact command should clear inherited Slurm option
environment variables before invoking the script. This prevents inherited
`SBATCH_*` values from adding node, exclusive, memory or GPU constraints:

```bash
(
  set -euo pipefail
  cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
  for name in $(compgen -A variable SBATCH_); do unset "$name"; done
  for name in CODE_DIR RUN_ROOT EVIDENCE_DIR PYTHON ENTRY COORDINATOR SLURM_BIN PREPARED_INPUT GPU_REFERENCE_DIR CONFIG; do unset "$name"; done
  export CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3
  export EXPECTED_CODE_COMMIT=03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8
  exec /rmprog/slurm/v22.05.7/bin/sbatch \
    /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3/scripts/hpc/slurm_global_server_cpu_48core.sh
)
```

The deployment input is the verified Git bundle
`cpu48-b64-03026d013626.bundle` (SHA-256
`940a6f7e2aa6bf545661ef1c815cbd0156fa92b62b18452cbad7398d7c49d47c`), whose
prerequisite commit is `b1b5cf0835da298bddb388e61d754feeb6c14cf2`. Import it
fast-forward-only, verify ref `codex/global-cpu48-b64-v3` at the exact commit,
and require a clean target checkout before submission. Use the canonical
runtime `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`.
The prepared input must be
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`
with SHA-256 `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`;
the GPU reference directory is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model`
with reference SHA-256
`a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`.
The config is `<CODE_DIR>/configs/orchidee_man_250919.yaml`; results may be
written only beneath
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/<job-id>`.

Stop without submission if the bundle prerequisite is unavailable, any
commit/ref/tree or frozen input/reference hash differs, the canonical runtime
or declared paths are unavailable, the checkout is dirty, scheduler or
occupancy inspection is stale, the login endpoint is not `cln01` or `cln02`,
or the reviewed allocation/command/cost envelope changes. This exact single
submission was approved and is now terminal; its budget is not renewed.

The script's directives request only the resources listed above. Slurm selects
the eligible node automatically. Submit only from the approved `cln01` or
`cln02` login endpoint under the recorded authorization. Never submit from
`test01`-`test04`; those are restricted to small CPU diagnostics.

The single workload preserves 64 unique FULL64 points, 896 point-days, 760
native output leaves, source masks/dtypes, x64, and fixed tolerances
`atol=1e-8`, `rtol=1e-10`. Scalar reference cap is 2700 seconds; B64 requires
at least 1800 seconds remaining. Each phase collects five core and five
wrapper samples. GPU B64 is the same-workload comparator; GPU B1024 is only a
normalized reference. No scientific acceptance or throughput result is
implied by this preparation.

This proposal is for exactly one submission attempt. The coordinator stops
within the 5400-second parent window, bills all 48 allocated parent CPUs for
the full allocation including the scalar phase, and never automatically
re-submits or reuses this budget after a failure or timeout.
