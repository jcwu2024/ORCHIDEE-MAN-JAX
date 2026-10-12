# Local Checks

Worktree: `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` on
`research/global-throughput`. The local `jax_orchidee` import resolved to
`E:\ORCHIDEE-MAN-JAX\runtime\worktrees\global-throughput\jax_orchidee\__init__.py`;
both `git status --short -- jax_orchidee` and
`git diff --name-only -- jax_orchidee` were empty.

Focused AOT tests:

```text
conda run -n ORCJAX python -m pytest -q tests/unit/test_global_gpu_capacity.py -k aot
6 passed, 54 deselected in 6.52s
```

This `-k aot` selection includes three of the four added regressions. It does
not select `test_raw_python_scalar_lower_compile_and_bound_call_interfaces`
because `raw` rather than `aot` appears in that test name. That fourth
regression passed in the full-module run below; its 50 passes include all four
added tests.

Targeted lint:

```text
conda run -n ORCJAX ruff check --no-cache scripts/dev/benchmark_global_batch.py tests/unit/test_global_gpu_capacity.py outputs/research/global_throughput/cpu-aot-signature-repair-20261012/aot_signature_diagnostic.py
All checks passed!
```

Diff whitespace check:

```text
git diff --check
passed
```

Full capacity module:

```text
conda run -n ORCJAX python -m pytest -q tests/unit/test_global_gpu_capacity.py
50 passed, 10 failed in 13.88s
```

Each of the ten failures raises the same existing guard error from
`global_source_compatibility.source_identity`: `science source changed outside
certified execution-only cache repair`. None reaches the AOT helper. Failed
cases:

- `test_real_Bash_and_Python_capacity_dryrun_without_GPU_initialization`
- `test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline[False]`
- `test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline[True]`
- `test_continuation_measures_only256_then512_and_seals_first`
- `test_continuation_launcher_actual_Bash_expansion[False]`
- `test_continuation_launcher_actual_Bash_expansion[True]`
- `test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines[True]`
- `test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines[False]`
- `test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout[False]`
- `test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout[True]`
