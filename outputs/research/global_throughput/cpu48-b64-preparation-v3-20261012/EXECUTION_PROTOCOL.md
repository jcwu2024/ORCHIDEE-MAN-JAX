# CPU48 / FULL64 B64 v3 execution protocol

## Immutable execution identity

- Ref: `codex/global-cpu48-b64-v3`
- Commit: `03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`
- Base: `b1b5cf0835da298bddb388e61d754feeb6c14cf2`
- Intended checkout: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3`
- Bundle: `cpu48-b64-03026d013626.bundle`, SHA-256 `940a6f7e2aa6bf545661ef1c815cbd0156fa92b62b18452cbad7398d7c49d47c`
- Bundle prerequisite: `b1b5cf0835da298bddb388e61d754feeb6c14cf2`; size: 3380 bytes.
- Certified science identity: `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`
- Actual science source identity: `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`
- Baseline ref: `133e35aec7b83b9d79888c46092f64d518e97299`
- Frozen prepared-input SHA-256: `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`
- GPU FULL64 reference SHA-256: `a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`

The immutable pin contains only the reviewed execution harness/test overlay.
The `jax_orchidee` subtree is byte-identical to the declared base. Do not
overlay the current research branch or modify the pinned checkout.

## Workload and gate

Use 64 unique points through continuous Days 2-8 and Days 9-15, 48 half-hour
steps per day, totaling 896 point-days. Preserve all 760 native outputs,
masks, shapes and dtypes with x64 enabled and fixed tolerances `atol=1e-8`,
`rtol=1e-10`. Keep all CPU-internal whole-tree, permutation,
replacement/isolation, reset, structural and nonfinite gates hard. Finite
cross-GPU differences may preserve exploratory timing but do not establish
scientific acceptance.

The coordinator first runs one scalar FULL64 reference using one step CPU,
then a single native CPU JAX process/device using B64 vmap on 48 allocated CPU
cores. Each phase retains five synchronized core samples and five wrapper
samples. Scalar time is capped at 2700 seconds. Dispatch B64 only when at least
1800 seconds remain in the total 5400-second parent allocation. Stop at the
allocation wall limit. Do not add a GPU, change precision or tolerance, or
alter physics/ABI/intermediate batch sizes.

The canonical server environment is documented as Python 3.11.15 with JAX and
JAXLIB 0.4.38; the launcher sets `JAX_ENABLE_X64=true`. The test01 dry-run
deliberately did not import JAX, so this smoke verifies the requested
environment flag and CLI expansion, not the installed remote JAX versions or
active x64 configuration. A separate local ORCJAX metadata probe found Python
3.11.15, JAX/JAXLIB 0.10.1 and default x64 false; this is not the target
runtime. See `LOCAL_ADMISSION.json` for the boundary.

Measure the full same-workload outputs. Report cold lowering/compilation,
first call, hot core and wrapper times, serialization/I/O, RSS and process CPU
time separately. Compare throughput as 896 point-days divided by synchronized
seconds. Do not infer CPU speed or cost from the dry-run or normalized B1024
reference.

## Execution paths and runtime

- Canonical Python: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`
- Prepared input: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`
- GPU reference: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model`
- Config: `<CODE_DIR>/configs/orchidee_man_250919.yaml`
- New result root: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/<job-id>`
- Coordinator: `<CODE_DIR>/scripts/hpc/global_cpu48_b64_coordinator.py`
- Entry: `<CODE_DIR>/scripts/dev/benchmark_global_server_cpu.py`

The target production checkout must be created as a new immutable version and
verified at the exact commit with a clean tree before submission. Do not point
the launcher at the research checkout or local test staging files.

The proposed future run is one attempt only. All 48 parent CPUs are billed for the
full allocation, including scalar reference time. Any scalar failure, timeout,
insufficient candidate window or batch failure closes that attempt. There is
no automatic requeue, retry, second submission or budget reuse.
