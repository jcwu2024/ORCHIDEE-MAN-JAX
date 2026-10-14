# Startup assertions v3 — local validation

Date: 2026-10-15 (Asia/Shanghai). Scope: only the three Git HEAD/date/hostname
pipeline hard-failure tests in `tests/unit/test_global_server_cpu_startup.py`.
The production launcher was not edited for v3. Original local report and
`LOCAL_VALIDATION.md` remain preserved.

The parent's target GNU Bash 4.2.46 run of v2 returned status1 for these three
fatal `group | tee` failures, whereas local GNU Bash 5.2.37 returned45/46/48.
V3 requires each case to exit nonzero, dispatch zero times, and contain the
corresponding failing command in its trace. Exact srun47, scontrol44 and
resource/path gate2 expectations are unchanged.

Executed command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/admin/.conda/envs/ORCJAX/python.exe' -B -X utf8 tests/unit/test_global_server_cpu_startup.py --report-json outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/local-startup-matrix-v3.json
git diff --check
```

Result: **14/14 passed**, 15.448 seconds. Diff whitespace check returned zero.
No focused18/full81/scientific tests were repeated. No JAX import, real model,
real scheduler or scientific input/reference read occurred. Target v3 Linux
validation and a new immutable execution pin remain with the parent.

Launcher SHA256: `4bb816f3e1c586bc6fa59caa1cce8d894aafe2bb504a1bfcf8572ac6a7799507`.

Startup test SHA256: `5ee1dbc8d28cffecc49dcabf881f286a6ae46d459631bf40ad229d8496faa4ae`.

| Actual case | Exit code | Record-only dispatches |
|---|---:|---:|
| test_context_tee_IO_failure_stays_fatal | 49 | 0 |
| test_dirty_checkout_blocks_dispatch | 2 | 0 |
| test_dispatch_failure_stays_fatal_even_when_GPU_telemetry_fails | 47 | 1 |
| test_git_HEAD_failure_stays_fatal | 45 | 0 |
| test_missing_allocated_TRES_blocks_dispatch | 2 | 0 |
| test_missing_prepared_input_blocks_dispatch | 2 | 0 |
| test_nvidia_absent_warns_and_dispatches | 0 | 1 |
| test_nvidia_present_failure_warns_and_dispatches | 0 | 1 |
| test_nvidia_present_success_dispatches | 0 | 1 |
| test_optional_CPU_queries_fail_without_blocking_dispatch | 0 | 1 |
| test_required_date_failure_stays_fatal | 46 | 0 |
| test_required_hostname_failure_stays_fatal | 48 | 0 |
| test_scontrol_failure_stays_fatal | 44 | 0 |
| test_wrong_CPU_budget_blocks_dispatch | 2 | 0 |

Source line-ending verification: both launcher and startup test are LF only.
