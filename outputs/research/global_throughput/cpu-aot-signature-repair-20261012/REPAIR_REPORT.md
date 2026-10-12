# AOT Signature Repair

The shared AOT signature helper now calls JAX `shaped_abstractify` for each
caller leaf. It imports from `jax._src.api_util` on JAX 0.4.38 and falls back
to `jax._src.core` on the local JAX 0.10.1. Admission still compares pytree,
shape, dtype and weak type; no JAX internals or physical processes changed.

The frozen payload SHA matched
`be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`. It has
64 scalar lanes and 64 second-block inputs with 536 dynamic leaves each. The
first four lanes contain 40 raw scalar paths each (39 Python floats and one
Python integer); lane 4 has no raw scalar leaves. In the complete first scalar
argument leaf list, the old helper fails on the first non-array leaf at
`.values_by_component[4][67]` (`[0][0][4][67]` in the flattened argument tree)
with `AttributeError: 'float' object has no attribute 'shape'`. This is
reproduced without physical model lowering. The final diagnostic records
unique non-array paths, types and counts, then compares raw and scalar-only
device-put metadata for both the 64 scalar and 64 second-block trees. Each
lane/group has 536 leaves; no array leaves are transferred for this metadata
pass.

The exact candidate file was staged separately, SHA-checked against the local
LF file (full SHA-256:
`bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`), and
imported directly on `ibtest01`. The diagnostic verified that
`scale.base` referenced this candidate and that its abstractifier came from
`jax._src.api_util`. Frozen class and physical-process imports remained from
the immutable v2 checkout at commit
`b1b5cf0835da298bddb388e61d754feeb6c14cf2`. The canonical interpreter was
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`.
The exact target command used `env JAX_ENABLE_X64=true
ORCHJAX_AOT_DIAGNOSTIC_LAUNCHER="taskset -c 0" taskset -c 0
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-aot-signature-repair-20261012/aot_signature_diagnostic_v3.py`
with the `--code-dir`, `--candidate-source`, `--expected-candidate-sha256`,
`--payload`, and `--output` values recorded in the final JSON. It records
`sys.executable`, supporting checkout commit, candidate SHA, payload SHA and
command arguments.

On Python 3.11.15/JAX+jaxlib 0.4.38 with x64 enabled and CPU affinity `[0]`,
candidate scalar lower/compile/call accepted the same value as a raw Python
float and as a device-put float; tree, shape and dtype mismatches were
rejected. Actual `SingleRunner` scalar and B2 core/wrapper flows passed with
frozen lanes 0 and 4 (`001.0-071.0`, `215.0-115.0`), 536 input leaves, and
complete output trees equal. Only `make_scalar_transition` and
`make_batch_transition` were replaced by small state-carrying diagnostic
functions. Raw/device signatures matched within each selected lane, and
shape/dtype matched across lanes despite 42 weak-type differences. Across all
64 scalar lanes and all 64 second-block lanes, raw/device metadata matched
exactly at all 536 leaves, including tree structure. JAX's native
`Compiled.call` accepted a weak-only difference; the explicit audit rejected
that same-shape, same-dtype strong input. This behavior was observed, not
changed.

The candidate source SHA is
`bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`. The
final full-metadata result has SHA-256
`976b3cf45c3621249b212a9aad3f11c95c297244bdd9876523905b1a49b6ff0c`.

Local targeted checks were run in the `ORCJAX` environment:

- `conda run -n ORCJAX python -m pytest -q tests/unit/test_global_gpu_capacity.py -k aot` — 6 passed, 54 deselected.
- `conda run -n ORCJAX ruff check --no-cache scripts/dev/benchmark_global_batch.py tests/unit/test_global_gpu_capacity.py outputs/research/global_throughput/cpu-aot-signature-repair-20261012/aot_signature_diagnostic.py` — passed.
- `git diff --check` — passed.
- `conda run -n ORCJAX python -m pytest -q tests/unit/test_global_gpu_capacity.py` — 50 passed, 10 failed.

All ten full-module failures stop at the existing
`global_source_compatibility.source_identity` guard with
`science source changed outside certified execution-only cache repair`;
none reaches the repaired AOT helper. The failing parametrized cases are
`test_real_Bash_and_Python_capacity_dryrun_without_GPU_initialization`,
`test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline[False]`,
`test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline[True]`,
`test_continuation_measures_only256_then512_and_seals_first`,
`test_continuation_launcher_actual_Bash_expansion[False]`,
`test_continuation_launcher_actual_Bash_expansion[True]`,
`test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines[True]`,
`test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines[False]`,
`test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout[False]`,
and `test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout[True]`.
The source-identity failure is the same across all ten. Both
`git status --short -- jax_orchidee` and `git diff --name-only -- jax_orchidee`
were empty. No physical model lower/compile, GPU execution, Slurm submission or
production deployment was performed. See [LOCAL_CHECKS.md](LOCAL_CHECKS.md) for
exact local commands/results and the full list of guard-failing tests. The
`-k aot` selection passed six tests; all four added regressions passed in the
full-module run. The raw-Python-scalar regression is not selected by `-k aot`,
but is among the 50 full-module passes.

Artifacts:

- `aot_signature_diagnostic.py`: portable bounded target diagnostic.
- [linux-jax-0.4.38-candidate-full-metadata.json](linux-jax-0.4.38-candidate-full-metadata.json): final exact-candidate result with both all-64 metadata groups, interpreter, support commit, command arguments and hashes.
- [LOCAL_CHECKS.md](LOCAL_CHECKS.md): exact local test/lint commands, results and guard-failure classification.
- [linux-jax-0.4.38-candidate.json](linux-jax-0.4.38-candidate.json): earlier exact-candidate run, retained as separate evidence.
- `linux-jax-0.4.38-preliminary-injection.json`: earlier preliminary helper
  injection run, retained separately from exact-candidate evidence.
