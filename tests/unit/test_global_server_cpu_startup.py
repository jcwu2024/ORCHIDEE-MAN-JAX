"""Real launcher startup with record-only scheduler stubs; stdlib, no model.

Direct Linux/Windows entry (no pytest required):
  python -B tests/unit/test_global_server_cpu_startup.py --report-json REPORT.json
On Explore1000 provide --scratch-root under /WORK/liwei_work/jcwu/. The test
simulates a four-CPU context; it never submits a job or calls real srun/Python.
Only PROJECT and SLURM_BIN assignments in a temporary launcher copy change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/hpc/slurm_global_server_cpu_4core.sh"
BASH = shutil.which("bash")
SCRATCH_ROOT = None
OBSERVATIONS = []


def shell_path(path):
    if os.name != "nt":
        return str(path)
    return subprocess.check_output([BASH, "-c", 'cygpath -u "$1"', "fixture", str(path)],
        text=True).strip()


def write_executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


class LauncherStartupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BASH, "Bash is required for actual launcher startup control flow")
        self.temp = tempfile.TemporaryDirectory(prefix="pool-startup-", dir=SCRATCH_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.sandbox = Path(self.temp.name)
        self.project = self.sandbox / "project with spaces"
        self.code = self.project / "runtime/worktrees/immutable-fixture"
        self.bin = self.sandbox / "stub-bin"
        self.slurm = self.sandbox / "scheduler-stubs"
        self.trace = self.sandbox / "trace.txt"
        self.dispatch = self.sandbox / "dispatch-arguments.txt"
        self.run_root = self.project / "runtime/outputs/research/global_throughput/server-benchmark/startup-fixture"
        self.bin.mkdir()
        self.slurm.mkdir()
        # Expose only these actual non-compute utilities. Host NVIDIA/git/srun
        # executables cannot accidentally leak into the controlled PATH.
        for name in ("realpath", "awk", "mkdir", "tee"):
            actual = subprocess.check_output([BASH, "-c", 'command -v "$1"', "fixture", name], text=True).strip()
            self.assertTrue(actual)
            write_executable(self.bin / name, '#!/bin/bash\nexec ' + shlex.quote(actual) + ' "$@"\n')
        for name in ("date", "hostname", "uptime", "taskset", "lscpu", "nvidia-smi"):
            variable = "STUB_" + name.upper().replace("-", "_") + "_STATUS"
            write_executable(self.bin / name, '#!/bin/bash\n'
                + "printf '%s\\n' " + shlex.quote(name) + ' >> "$STUB_TRACE"\n'
                + "printf '%s\\n' " + shlex.quote("stub " + name) + "\n"
                + 'exit "${' + variable + ':-0}"\n')
        write_executable(self.bin / "git", '''#!/bin/bash
printf 'git %s\n' "$1" >> "$STUB_TRACE"
if [[ $1 == status ]]; then
  if [[ ${STUB_DIRTY:-0} == 1 ]]; then printf ' M dirty-fixture\n'; fi
elif [[ $1 == rev-parse && $2 == HEAD ]]; then
  printf '1111111111111111111111111111111111111111\n'
  exit "${STUB_GIT_HEAD_STATUS:-0}"
else
  exit 96
fi
''')
        write_executable(self.slurm / "scontrol", '''#!/bin/bash
printf 'scontrol %s\n' "$*" >> "$STUB_TRACE"
if [[ ${STUB_SCONTROL_STATUS:-0} != 0 ]]; then exit "$STUB_SCONTROL_STATUS"; fi
if [[ ${STUB_MISSING_TRES:-0} == 1 ]]; then
  printf 'JobId=mock-startup NumCPUs=4\n'
else
  printf 'JobId=mock-startup NumCPUs=4 NumTasks=1 Partition=cnall TRES=cpu=4,node=1,billing=4\n'
fi
''')
        write_executable(self.slurm / "srun", '''#!/bin/bash
printf 'srun\n' >> "$STUB_TRACE"
printf '%s\n' "$@" > "$STUB_DISPATCH_ARGUMENTS"
printf 'record-only srun: no Python, model, or allocation executed\n'
exit "${STUB_SRUN_STATUS:-0}"
''')
        # Required paths exist, but neither Python entry can execute. The srun
        # stub deliberately records its argv without invoking any downstream.
        python = self.project / ".venvs/orcjax_cpu/bin/python"
        write_executable(python, '#!/bin/bash\nprintf "UNEXPECTED_PYTHON_EXECUTION\\n" >> "$STUB_TRACE"\nexit 97\n')
        prepared = self.project / "runtime/outputs/frozen-placeholder.pkl"
        config = self.code / "configs/orchidee_man_250919.yaml"
        for path in (prepared, config, self.code / "scripts/dev/benchmark_global_server_cpu.py",
                     self.code / "scripts/dev/global_cpu_child_environment.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture only; never read or executed\n", encoding="utf-8")
        for path in (self.project / "runtime/outputs/gpu-reference", self.project / "runtime/outputs/cpu-reference"):
            path.mkdir(parents=True)
        source = LAUNCHER.read_text(encoding="utf-8")
        project_assignment = "PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX"
        slurm_assignment = "SLURM_BIN=/rmprog/slurm/v22.05.7/bin"
        self.assertEqual(source.count(project_assignment), 1)
        self.assertEqual(source.count(slurm_assignment), 1)
        adjusted = source.replace(project_assignment, "PROJECT=" + shlex.quote(shell_path(self.project)))
        adjusted = adjusted.replace(slurm_assignment, "SLURM_BIN=" + shlex.quote(shell_path(self.slurm)))
        self.script = self.sandbox / "launcher-under-test.sh"
        write_executable(self.script, adjusted)
        self.prepared = prepared
        # Explicit settings isolate the fixture from inherited sbatch/pool policy.
        self.env = {key: value for key, value in os.environ.items()
            if not key.startswith(("SLURM_", "SBATCH_", "POOL_", "STUB_", "CPU_REFERENCE_", "EXPECTED_", "CACHE_"))
            and key not in {"CPU4_METHODS", "VALIDATION_MODE", "BENCHMARK_SIZES", "RUN_ROOT", "CONFIG", "PREPARED_INPUT", "GPU_REFERENCE_DIR", "CODE_DIR"}}
        self.env.update(PATH=shell_path(self.bin), CODE_DIR=shell_path(self.code),
            RUN_ROOT=shell_path(self.run_root), PREPARED_INPUT=shell_path(prepared), CONFIG=shell_path(config),
            GPU_REFERENCE_DIR=shell_path(self.project / "runtime/outputs/gpu-reference"),
            CPU_REFERENCE_DIR=shell_path(self.project / "runtime/outputs/cpu-reference"),
            CPU_REFERENCE_ROLE="accepted", VALIDATION_MODE="strict", BENCHMARK_SIZES="64", CPU4_METHODS="pool",
            POOL_WORKERS="4", POOL_START_CONCURRENCY="1", SLURM_JOB_ID="mock-startup",
            SLURM_CPUS_PER_TASK="4", SLURM_NTASKS="1", SLURM_JOB_PARTITION="cnall",
            STUB_TRACE=shell_path(self.trace), STUB_DISPATCH_ARGUMENTS=shell_path(self.dispatch),
            PYTHONDONTWRITEBYTECODE="1")

    def launch(self, **changes):
        # Git Bash on Windows prepends host utilities to the inherited PATH at
        # process startup. Set it inside Bash before executing the unchanged
        # launcher so git/date/hostname resolve to the same stubs on all hosts.
        result = subprocess.run([BASH, "-c", 'export PATH="$1"; exec /bin/bash "$2"',
            "fixture", shell_path(self.bin), shell_path(self.script)], env={**self.env, **changes},
            text=True, capture_output=True, timeout=20)
        trace = self.trace.read_text().splitlines() if self.trace.exists() else []
        arguments = self.dispatch.read_text().splitlines() if self.dispatch.exists() else []
        OBSERVATIONS.append({"case": self.id().rsplit(".", 1)[-1], "exit_code": result.returncode,
            "dispatch_count": trace.count("srun"), "trace": trace, "dispatch_arguments": arguments,
            "stdout": result.stdout, "stderr": result.stderr, "real_model_or_scheduler_execution": False})
        self.assertNotIn("UNEXPECTED_PYTHON_EXECUTION", trace)
        return result, trace, arguments

    def assert_dispatch(self, result, trace, arguments, code=0):
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertEqual(trace.count("srun"), 1)
        self.assertEqual(arguments[:3], ["--ntasks=1", "--cpus-per-task=4", "--cpu-bind=cores"])
        self.assertIn("global_cpu_child_environment.py", arguments[6])
        self.assertEqual(arguments[7:10], ["--thread-budget", "1", "--"])
        self.assertEqual(arguments[arguments.index("--mode") + 1], "pool")
        self.assertIn("starting mode=pool", result.stdout)

    def assert_blocked(self, result, trace, code=None):
        self.assertNotEqual(result.returncode, 0)
        if code is not None:
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertNotIn("srun", trace)

    def test_nvidia_present_failure_warns_and_dispatches(self):
        result, trace, arguments = self.launch(STUB_NVIDIA_SMI_STATUS="9")
        self.assert_dispatch(result, trace, arguments)
        self.assertIn("optional GPU telemetry unavailable: nvidia-smi failed", result.stderr)

    def test_nvidia_present_success_dispatches(self):
        self.assert_dispatch(*self.launch())

    def test_nvidia_absent_warns_and_dispatches(self):
        (self.bin / "nvidia-smi").unlink()
        result, trace, arguments = self.launch()
        self.assert_dispatch(result, trace, arguments)
        self.assertIn("optional GPU telemetry unavailable: nvidia-smi not found", result.stderr)

    def test_optional_CPU_queries_fail_without_blocking_dispatch(self):
        result, trace, arguments = self.launch(STUB_UPTIME_STATUS="41", STUB_TASKSET_STATUS="42", STUB_LSCPU_STATUS="43")
        self.assert_dispatch(result, trace, arguments)
        for name in ("uptime failed", "taskset query failed", "lscpu failed"):
            self.assertIn("optional CPU telemetry unavailable: " + name, result.stderr)

    def test_dispatch_failure_stays_fatal_even_when_GPU_telemetry_fails(self):
        result, trace, arguments = self.launch(STUB_NVIDIA_SMI_STATUS="9", STUB_SRUN_STATUS="47")
        self.assert_dispatch(result, trace, arguments, code=47)

    def test_missing_prepared_input_blocks_dispatch(self):
        self.prepared.unlink()
        result, trace, _ = self.launch()
        self.assert_blocked(result, trace, code=2)
        self.assertIn("required path missing", result.stderr)

    def test_dirty_checkout_blocks_dispatch(self):
        result, trace, _ = self.launch(STUB_DIRTY="1")
        self.assert_blocked(result, trace, code=2)
        self.assertIn("clean and immutable", result.stderr)

    def test_wrong_CPU_budget_blocks_dispatch(self):
        result, trace, _ = self.launch(SLURM_CPUS_PER_TASK="1")
        self.assert_blocked(result, trace, code=2)
        self.assertIn("exactly four allocated CPUs", result.stderr)

    def test_missing_allocated_TRES_blocks_dispatch(self):
        result, trace, _ = self.launch(STUB_MISSING_TRES="1")
        self.assert_blocked(result, trace, code=2)
        self.assertIn("Slurm did not report allocated TRES", result.stderr)

    def test_scontrol_failure_stays_fatal(self):
        result, trace, _ = self.launch(STUB_SCONTROL_STATUS="44")
        self.assert_blocked(result, trace, code=44)

    def test_git_HEAD_failure_stays_fatal(self):
        result, trace, _ = self.launch(STUB_GIT_HEAD_STATUS="45")
        # Bash versions can report a different nonzero status for group|tee.
        self.assert_blocked(result, trace)
        self.assertIn("git rev-parse", trace)

    def test_required_date_failure_stays_fatal(self):
        result, trace, _ = self.launch(STUB_DATE_STATUS="46")
        self.assert_blocked(result, trace)
        self.assertIn("date", trace)

    def test_required_hostname_failure_stays_fatal(self):
        result, trace, _ = self.launch(STUB_HOSTNAME_STATUS="48")
        self.assert_blocked(result, trace)
        self.assertIn("hostname", trace)

    def test_context_tee_IO_failure_stays_fatal(self):
        write_executable(self.bin / "tee", '#!/bin/bash\nprintf "stub tee IO failure\\n" >&2\nexit 49\n')
        result, trace, _ = self.launch()
        self.assert_blocked(result, trace)
        self.assertIn("stub tee IO failure", result.stderr)


def main():
    global BASH, LAUNCHER, SCRATCH_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--bash", default=BASH)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    BASH, LAUNCHER, SCRATCH_ROOT = args.bash, args.launcher.resolve(), args.scratch_root
    if SCRATCH_ROOT is not None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LauncherStartupTests))
    if args.report_json:
        report = {"schema": "global_cpu_pool_launcher_startup_matrix_v1", "passed": result.wasSuccessful(),
            "test_count": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "launcher": str(LAUNCHER), "launcher_sha256": hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bash": BASH, "python": sys.executable, "JAX_imported": "jax" in sys.modules,
            "model_execution": False, "real_Slurm_execution": False,
            "resource_context": "stub four-CPU metadata only; no allocation, binding or srun request executed",
            "temporary_source_changes": ["PROJECT assignment", "SLURM_BIN assignment"],
            "cases": OBSERVATIONS}
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
