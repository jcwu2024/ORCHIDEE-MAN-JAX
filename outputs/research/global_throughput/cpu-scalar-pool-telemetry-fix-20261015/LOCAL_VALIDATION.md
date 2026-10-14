# CPU pool optional telemetry startup fix — local receipt

Date: 2026-10-15 (Asia/Shanghai). Worktree: `research/global-throughput`.

The production launcher keeps `set -euo pipefail`. Only optional `uptime`,
`taskset -pc $$`, `lscpu -e=CPU,CORE,SOCKET,ONLINE`, and `nvidia-smi` now warn
on unavailable/failed telemetry. Date, hostname, Git HEAD, scontrol/TRES,
required paths/resources, tee IO and srun retain fatal behavior.

## Executed commands

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/admin/.conda/envs/ORCJAX/python.exe' -B -X utf8 tests/unit/test_global_server_cpu_startup.py --report-json outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/local-startup-matrix.json
& 'C:/Users/admin/.conda/envs/ORCJAX/python.exe' -B -X utf8 -m pytest tests/unit/test_global_server_cpu_launchers.py -q -p no:cacheprovider
& 'C:/Program Files/Git/bin/bash.exe' -n scripts/hpc/slurm_global_server_cpu_4core.sh
git diff --check
```

Startup matrix: **14 passed**; focused launcher regression: **18 passed**.
The latter emitted one PytestConfigWarning for `cache_dir` because the cache
plugin was disabled. Bash syntax and diff whitespace checks returned zero.
Local Bash: GNU Bash 5.2.37(1)-release (x86_64-pc-msys).

The startup harness uses a temporary project path containing spaces and a copy
of the real launcher with only PROJECT and SLURM_BIN assignments changed. It
executes actual startup logic with controlled metadata/telemetry commands and
record-only scheduler stubs. It never executes downstream Python, JAX, model,
real scheduler or prepared/reference content. On Windows it assigns PATH
inside Bash before exec to defeat Git Bash's host-path insertion. Both source
files have zero CRLF bytes (LF only). Git blob equality awaits the parent's
new execution pin.

## Source identities

- Launcher SHA256: `4bb816f3e1c586bc6fa59caa1cce8d894aafe2bb504a1bfcf8572ac6a7799507`
- Startup test SHA256: `a28e4e5a7063e384009287ca3ba3b79e362a86f838b8028d6a1e13f928d7bcae`

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

The JSON report preserves exact stdout/stderr, command traces and dispatched
argv for every case. Failed NVIDIA telemetry reaches srun; that same case
with srun status47 exits47. Required failures stop before dispatch.

## Target Linux validation pending

The parent owns deployment and target execution. Run the standalone stdlib
entry using the canonical CPU Python and the immutable candidate checkout:

```bash
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python -B tests/unit/test_global_server_cpu_startup.py --scratch-root /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/telemetry-startup-scratch --report-json /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/telemetry-startup-target.json
```

This is startup control-flow evidence, not scientific numerical acceptance,
physical-core/resource admission, or approval for a new paid job.
