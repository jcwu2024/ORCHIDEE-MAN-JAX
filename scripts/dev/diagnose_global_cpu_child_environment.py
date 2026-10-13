"""Tiny real-process CPU environment diagnostic; no model, JAX or fake allocation.

On Linux it restricts itself and every descendant to one available CPU. A
48-thread policy check is only an environment/CLI check, never CPU48 admission.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.dev import global_cpu_child_environment as boundary  # noqa: E402
from scripts.hpc.global_cpu48_b64_coordinator import run_process  # noqa: E402

CONFLICTS = {"OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "MKL_NUM_THREADS": "64",
    "OPENBLAS_NUM_THREADS": "64", "NUMEXPR_NUM_THREADS": "64",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=48 --xla_cpu_multi_thread_eigen=false",
    "JAX_PLATFORMS": "cuda", "JAX_ENABLE_X64": "false", "JAX_DISABLE_JIT": "true",
    "JAX_NUM_CPU_DEVICES": "48", "TF_NUM_INTRAOP_THREADS": "1"}


def diagnose(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for key in list(os.environ):
        if key.startswith(("SLURM_", "ORCHJAX_SLURM_")):
            os.environ.pop(key)
    os.environ.update(CONFLICTS)
    runner = ROOT / "scripts/dev/benchmark_global_server_cpu.py"
    child = runner.with_name("global_cpu_child_environment.py")
    transport = output / "conflicting_transport.py"
    transport.write_text("import os,sys,subprocess\n" + f"os.environ.update({CONFLICTS!r})\n" +
        "command=sys.argv[1:]\n" +
        "if os.name=='nt': raise SystemExit(subprocess.call(command,env=dict(os.environ)))\n" +
        "os.execvpe(command[0],command,dict(os.environ))\n", encoding="utf-8")
    common = ["--sizes", "64", "--prepared-input", str(output / "not-read.pkl"),
        "--gpu-reference-dir", str(output / "not-read-gpu"), "--output-dir", str(output / "not-created"),
        "--require-child-environment"]
    checks = {}
    for mode, budget in (("scalar", 1), ("batch", 48)):
        args = [sys.executable, str(runner), "--mode", mode, "--cpu-budget", str(budget), *common, "--dry-run"]
        if mode == "batch":
            args += ["--cpu-reference-dir", str(output / "not-read-scalar")]
        command = [sys.executable, str(transport), *boundary.wrapped_command(sys.executable, child, budget, args)]
        log = output / f"{mode}-policy.log"
        result = run_process(command, log, 20)
        if result["returncode"] != 0:
            raise RuntimeError(f"actual {mode} child failed; see {log}")
        raw = log.read_text(encoding="utf-8")
        start = raw.index('{\n  "status": "dry_run"')
        dry, _ = json.JSONDecoder().raw_decode(raw[start:])
        actual = dry["actual_child_prejax_receipt"]
        expected = {key: boundary.policy(budget).get(key) for key in boundary.WHITELIST}
        if actual["actual_environment"] != expected or actual["unexpected_policy_keys"] or actual["JAX_imported"]:
            raise RuntimeError("actual post-transport child policy mismatch")
        if actual["inherited_transport_policy"]["environment"]["OMP_NUM_THREADS"] != "1":
            raise RuntimeError("conflicting inherited transport was not exercised")
        if actual["affinity"] is not None and len(actual["affinity"]) != 1:
            raise RuntimeError("diagnostic widened beyond one available CPU")
        checks[mode] = {"dispatch": result, "actual_child_receipt": actual}
    real = [sys.executable, str(runner), "--mode", "batch", "--cpu-budget", "48", *common,
            "--cpu-reference-dir", str(output / "not-read-scalar")]
    for name, env in (("wrong-policy", dict(os.environ)),
                      ("no-allocation", boundary.normalized_environment(48))):
        result = subprocess.run(real, env=env, capture_output=True, text=True, timeout=20)
        (output / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        receipt = json.loads(result.stderr.splitlines()[0])
        if result.returncode == 0 or receipt["JAX_imported"] or receipt["status"] != "resource_admission_failed":
            raise RuntimeError("wrong policy/unallocated process did not fail before JAX")
        checks[name] = {"returncode": result.returncode, "actual_child_receipt": receipt}
    return {"schema": "global_cpu48_environment_diagnostic_v1", "status": "passed_nonmodel_process_checks",
        "python": sys.executable, "python_version": sys.version, "code_dir": str(ROOT),
        "hostname": __import__("platform").node(), "checks": checks,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "JAX_imported": "jax" in sys.modules, "model_execution": False,
        "physical_allocation_claim": False,
        "scope": "actual run_process/exec children only; no model compilation or CPU48 allocation admission"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.output_dir.resolve())
    raw = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "environment-diagnostic.json").write_text(raw, encoding="utf-8")
    print(raw, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
