# One additional original-T7 FP64 compile with parallel4

2026-10-02. **The user approved one additional free attempt, at most75min.
Execution is complete: one binary and32 calls passed the unchanged strict
gates and independent CPU verification in2783.953854s /0.773320515023card-hours.**
See the [actual results](shared_state_reference_fp64_parallel_results_20261002.md).

Pre-execution rationale: one final bounded free attempt was proposed to
obtain the originally specified derivative diagnostic. The previous
[FP64 run](shared_state_reference_fp64_results_20261002.md) passed exact
preparation and full dtype lowering, then reached its1800s backend timer.
It produced no numerical result. The proposed change addresses that compile
barrier; it is not a candidate for steady-state complete-update throughput.

## Evidence and uncertainty

The existing independent nvlink12.6.85 is already installed under the jcwu
runtime toolchains tree. Its official archive identity and CPU/container
version checks are documented in
`outputs/diagnostics/shared-nvlink-compatibility-20261001.md`. With LLVM module
parallelism enabled and force4, the synthetic GPU runtime check completed in
25.375s. That result proves compatibility only: it neither proves XLA used
nvlink nor measures real-model compile acceleration.

This is more than a blind extension of the same timer: enable the existing
parallel-module route, expose the existing matching linker through an
isolated process PATH, and record actual wrapper invocations. The execution
and data stayed fixed. The actual result now records compilation completion,
one public-backend nvlink call and32 strict numerical passes. Requested
parallel4 does not demonstrate four concurrent threads or a compiler speed
ratio; see the terminal result for the measured limits.

The old FP64 lowering measured201.938510s. Its backend time is censored at
1800s, with the compiler still active before termination. A different original
public T30 FP32 graph previously completed its backend in2545.616s on this
free V100, as recorded in
[T30 reference](shared_original_t30_dual_reference_results_20261001.md).
That different graph cannot predict or lower-bound this FP64 graph's time;
it supplies concrete context that a1800s cap can exclude a valid compile.
The proposed3600s cap covers that observed time scale while avoiding an
open-ended compile. There is no measured FP64/parallel4 completion estimate,
so3600s is a stopping budget, not a promise. If this final bounded attempt
also fails to yield admitted results, close it and return to speed work with
the unresolved reference uncertainty stated.

## Exact additional allocation and commands

- Platform: free `gln01` / `ibc13b02n04`, **one physical GPU1 V100**, OMP4,
  OpenBLAS1; no Slurm allocation or whole-node reservation. Inspect actual
  occupancy/load immediately before launch and wait for a suitable free node.
- New family only: `shared-state-reference-fp64-parallel-20261002` under
  `runtime/transfer/` and `runtime/outputs/diagnostics/`.
- Checkout/working directory:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`,
  HEAD`c599236bcf57b745e9bafed678298010f4c3b58f`.
- Canonical GPU environment and existing Singularity image/driver bindings.
  Existing nvlink toolchain is bound read-only. No installation is needed.
- Compiler change only:
  `--xla_gpu_enable_llvm_module_compilation_parallelism=true`
  `--xla_gpu_force_compilation_parallelism=4`; the first is also supplied
  explicitly to `lowered.compile(compiler_options=...)`.
- **Additional cumulative GPU-stage exposure≤4500s (75min,1.25 card-hours),
  including new preflight, failed starts, setup and timeout grace.** Paid cost
  isCNY0; there is no paid fallback. This new approval would cover up to1.25
  additional card-hours, separate from the closed0.647810511 already spent.
  Actual new-family exposure was0.773320515023card-hours; the allowance is closed.
- Phase ceilings: preflight180s; lowering360s; backend3600s; postcompile
  serialization/memory/results≤600s. Their sum is not an entitlement: the
 4500s cumulative cap and dynamically shortened process timer win.
- Exactly **one additional expensive model compile attempt**. A failed
  precompile start may be repaired within remaining cumulative time; a recorded
  backend attempt blocks another expensive compile, even if time remains.

These are the complete proposed execution commands, after that new approval:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-state-reference-fp64-parallel-20261002/run_shared_state_reference_fp64_parallel.sh preflight
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-state-reference-fp64-parallel-20261002/run_shared_state_reference_fp64_parallel.sh model
```

The launcher sums this family's earlier GPU-started exit records and clips
the next timeout to the remaining4500s minus setup/grace margins. A family
lock prevents concurrent stages. Missing prior exit accounting stops for
review. Both output preservation and the one-backend-attempt guard fail
closed. The new harness reports`model_compiles=0` until compilation actually
returns, then1; an attempt marker is recorded separately.

## Same science and acceptance

The harness is a fork of the **executed** FP64 source snapshot, with the
`combined` function AST unchanged. The FP64 policy file is a byte-identical
copy of its executed snapshot. It retains exact widened update0 values for
all64,884,481 dynamic parameters/300 leaves, artifact742/window0/T7, all3854
state derivatives and the same public loss. The268 input leaves,15 calibration
arrays and112 boolean mask records must match the previous admitted
preparation. Physical formulas, the exact2**-23 R4 threshold, precision,
architecture, data, losses, weights, masks and directions are unchanged.

Before backend compilation, the full public lowering again requires zero
FP16/BF16/FP32 tensors. Its raw StableHLO is compared with the previous saved
successful lowering. If needed, the small CPU helper permits only source
locations and bijective SSA/block/symbol alpha-renaming; it preserves all
operations, types, constants, attributes, strings, order and control flow.
The raw/normalized hashes and comparison mode are saved. Actual computation
differences stop before compilation; harmless textual metadata is not an
independent scientific rejection. No extensive IR-debugging project is part
of this preparation.

If a binary is obtained, save it and all32 complete result pickles: eight
calls for each dynamic basis direction33,199,428,200 in the original order.
Apply unchanged`rtol=1e-5, atol=1e-6` to native JVP/VJP consistency, same
coordinate repeats and complete3854-state VJP direction invariance. Require
finite outputs, all14 hard physical counts0, unchanged source/science/PFT
masks, declared-mask count14, exactly zero derivatives at the21 undefined
source-state coordinates, and estimated/observed active memory strictly
below24GiB. Report original-FP32/FP64 primal and derivative differences;
matching FP32 is not an admission requirement. Independently verify the saved
arrays on CPU without another model compile.

Failure of any numerical gate closes this bounded diagnostic. Success would
support only these bounded internally consistent derivatives; it cannot
adopt a new reference, reopen a candidate or establish a training speed gain.
No parameter AD, real optimizer update, training, T30, FD, precision sweep,
second compile or paid fallback is included.

## Reviewable assets and checks already run

Prepared sources:

- `scripts/hpc/probe_shared_state_reference_fp64_parallel.py`
- `scripts/hpc/run_shared_state_reference_fp64_parallel.sh`
- `scripts/hpc/shared_fp64_parallel_admission.py`
- `scripts/hpc/shared_fp64_parallel_nvlink.sh`
- Frozen executed policy and verification records under
  `outputs/diagnostics/shared-state-reference-fp64-parallel-20261002/`.

The new assets are uploaded to the declared new transfer family. Local ORCJAX
CPU checks passed Python syntax, unchanged combined AST, exactly one compile
call and the360/3600/600 phase caps. Small IR fixtures accept location/name
changes and reject operation/type/constant/operand/computational-string
changes. They import no JAX and start no GPU process.

On test01, actual Bash budget fixtures passed: only GPU-started exits count;
prior compile attempts, exhausted budget and missing exit records stop.
CPU nvlink`--version` and a deliberately invalid option produced identical
stdout/stderr and original exit0/1 through the transparent wrapper. The two
CPU-only log records do not prove XLA invocation. On gln01, `bash -n` and
actual unchanged-launcher`--dry-run preflight`/`--dry-run model` passed, using
real driver paths. The fresh dry-run model printed external4479s/internal4459s;
actual preflight exposure will reduce those limits before execution.

The wrapper uses`exec "$REAL" "$@"`, forwards every argument and raw exit,
and appends escaped non-secret argument paths to one log per stage. Model
`--version` records must be distinguished from non-version compiler calls;
PATH discovery alone is insufficient evidence of XLA linker use. No full
XLA dump or per-kernel files are enabled. Sources, logs, phase exits,
preparation arrays, graph, binary and result arrays are preserved within this
family. See`artifact-manifest.json`, `source-preparation.json`,
`cpu-preparation-check.json`, `target-cpu-check.json`, and
`target-shell-dry-run.txt` for actual preparation evidence.

**Authorization satisfied:** the user explicitly approved the additional
free-GPU commands and4500s/one-backend-attempt budget above. The preparation
evidence in this section predates execution; actual stage outcomes and saved
arrays are recorded in the linked result report. The approved single attempt
is complete; no further GPU execution is included.
