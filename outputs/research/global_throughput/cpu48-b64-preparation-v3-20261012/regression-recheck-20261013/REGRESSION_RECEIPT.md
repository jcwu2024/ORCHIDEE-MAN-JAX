# v3 local regression recheck

Date: 2026-10-13. The checks below ran in the detached, clean local
verification checkout at commit `03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`.
They cover the pinned harness and tests only. This is not acceptance on the
server's canonical JAX/JAXLIB 0.4.38 runtime and is not model, scientific, or
throughput evidence.

## Runtime and commands

Runtime metadata is captured in `source-admission.stdout.txt`: Python 3.11.15,
JAX/JAXLIB 0.10.1, `JAX_ENABLE_X64=true`, and the module imported from this
detached checkout. The local environment defaults x64 to false; the source
admission and regression commands enabled it. Python bytecode and pytest's cache provider
were disabled to keep the checkout clean. Pytest 9.0.3 emitted one existing
configuration warning because `cache_dir` is unknown with the cache plugin
disabled; the tests passed.

The preparation suite command was:

```powershell
$env:JAX_ENABLE_X64 = "true"
$env:PYTHONDONTWRITEBYTECODE = "1"
conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_global_cpu48_b64_preparation.py
```

An additional bounded AOT/source-certificate selection command was:

```powershell
$env:JAX_ENABLE_X64 = "true"
$env:PYTHONDONTWRITEBYTECODE = "1"
conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_global_gpu_capacity.py -k "aot_signature_abstractifies_python_numpy_jax_and_aval_leaves or raw_python_scalar_lower_compile_and_bound_call_interfaces or aot_audit_keeps_weak_type_strict or scale_compile_batch_uses_shared_raw_leaf_safe_aot_path or source_certificate_rejects_every_unapproved_diffuco_byte_change or source_certificate_rejects_unapproved_change_to_another_source_file"
```

That supplemental selection passed four AOT interface tests and six
source-certificate parameter cases: four DIFFUCO byte-change variants and two
other-source-file change variants. It does not replace the original ten
capacity protocol/launcher cases below. The direct Ruff command was:

```powershell
& 'C:\Users\admin\.conda\envs\ORCJAX\Scripts\ruff.exe' check --output-format concise scripts/dev/benchmark_global_batch.py scripts/dev/prepare_global_cpu48_execution.py scripts/hpc/global_cpu48_b64_coordinator.py tests/unit/test_global_cpu48_b64_preparation.py tests/unit/test_global_gpu_capacity.py
```

The shell launcher syntax command was:

```bash
bash -n scripts/hpc/slurm_global_server_cpu_48core.sh
```

## Results and evidence

- Preparation tests: 15 passed, 1 warning; exit code 0.
- Original capacity protocol/launcher cases: 10 passed, 1 warning; exit code 0.
- Additional AOT/source-certificate selection: 10 passed, 50 deselected, 1 warning; exit code 0. This is supplemental coverage.
- Ruff on the five pinned Python files: passed; exit code 0.
- Bash syntax check: passed; exit code 0.
- Source admission: passed for the pinned source/module identity; no model execution.
- Target smoke: the separately recorded `test01` launcher/coordinator dry-runs passed, with `JAX_imported=false` and `model_execution=false`.
- No `sbatch`, GPU execution, package installation, production deployment, or remote pytest run occurred.

The test, lint, and syntax checks have adjacent `.stdout.txt`, `.stderr.txt`,
and `.exitcode.txt` files. SHA-256 values below use the exact
captured bytes; the empty stderr files have the standard SHA-256 of an empty
file.

The original capacity protocol/launcher command ran from the pinned checkout
with `JAX_ENABLE_X64=true` and `PYTHONDONTWRITEBYTECODE=1`:

```powershell
conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider `
  tests/unit/test_global_gpu_capacity.py::test_real_Bash_and_Python_capacity_dryrun_without_GPU_initialization `
  'tests/unit/test_global_gpu_capacity.py::test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline[False]' `
  'tests/unit/test_global_gpu_capacity.py::test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline[True]' `
  tests/unit/test_global_gpu_capacity.py::test_continuation_measures_only256_then512_and_seals_first `
  'tests/unit/test_global_gpu_capacity.py::test_continuation_launcher_actual_Bash_expansion[False]' `
  'tests/unit/test_global_gpu_capacity.py::test_continuation_launcher_actual_Bash_expansion[True]' `
  'tests/unit/test_global_gpu_capacity.py::test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines[True]' `
  'tests/unit/test_global_gpu_capacity.py::test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines[False]' `
  'tests/unit/test_global_gpu_capacity.py::test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout[False]' `
  'tests/unit/test_global_gpu_capacity.py::test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout[True]'
```

Its raw result, stderr, exit code and structured selector/runtime metadata are
in `original-capacity-protocol-tests.stdout.txt`,
`original-capacity-protocol-tests.stderr.txt`,
`original-capacity-protocol-tests.exitcode.txt`, and
`original-capacity-protocol-tests.json`.

| Evidence file | SHA-256 |
| --- | --- |
| `preparation-tests.stdout.txt` | `5b2dd1a9fa71e55c6a5cabeb17aca7294579162664820b4957c4ad063db6980f` |
| `preparation-tests.stderr.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preparation-tests.exitcode.txt` | `13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354` |
| `capacity-source-guard-tests.stdout.txt` | `56331a2a293528da3a946f821694b3d7ad36dfed1caa24cf058f453c81f4d83d` |
| `capacity-source-guard-tests.stderr.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `capacity-source-guard-tests.exitcode.txt` | `13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354` |
| `original-capacity-protocol-tests.stdout.txt` | `acb99ac406c1904f625fdb16b77d48b3afc1fb6a3936cf5009a31aa60850a35a` |
| `original-capacity-protocol-tests.stderr.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `original-capacity-protocol-tests.exitcode.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `original-capacity-protocol-tests.json` | `1d58649a5cf313a4cf9d2277d90175de7e162a28c47558f87485162aff2d1604` |
| `ruff-python.stdout.txt` | `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18` |
| `ruff-python.stderr.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `ruff-python.exitcode.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `bash-syntax.stdout.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `bash-syntax.stderr.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `bash-syntax.exitcode.txt` | `13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354` |
| `source-admission.stdout.txt` | `f22f6c5ba46f3b49a78dea7a1c5b114fe538c5c77664566f56bc8a9e77d7dc34` |

The earlier `ruff.stdout.txt`/`ruff.stderr.txt` files preserve the failed
Conda-mediated lint attempt, whose output capture hit a GBK encoding error.
That attempt is not a Ruff result; the direct Python-only Ruff run above is the
lint result. The shell file is covered by Bash syntax validation.
