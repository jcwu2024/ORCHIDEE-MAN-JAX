"""Resolve the isolated raw neural prefix JVP command and preserve actual child status."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime")
CODE = ROOT / "checkouts/shared-split-width-v1"
PREFIX = "T7-raw-neural-jvp-20261010"
OVERLAY = ROOT / "transfer" / PREFIX
COMMON = ROOT / "transfer/shared-latent-reader-20261004"
DIAG = ROOT / "outputs/diagnostics/shared-latent-reader-20261004"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cpu-readiness", "execute"), required=True)
    p.add_argument("--approval-report", type=Path, default=DIAG / (PREFIX+".approval.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    return p


def child_command(args):
    # Reuse only the historical argument paths; neither its runner nor funding.
    try:
        from scripts.hpc import run_shared_latent_T7_state_jvp as paths
    except ImportError:
        import run_shared_latent_T7_state_jvp as paths
    namespace = argparse.Namespace(stage=args.stage, output_prefix=PREFIX,
        approval_report=args.approval_report, runtime_preflight=DIAG/(PREFIX+".runtime-preflight.json"),
        state_input_metadata=DIAG/"T7-state-inputs-20261009.json", repair_record=None,
        b1_completion=False, plan_only=args.plan_only or args.dry_run)
    command = paths.child_command(namespace)
    command[2] = str(OVERLAY/"probe_shared_latent_T7_raw_neural_jvp.py")
    values = {"--wall-seconds": "4800", "--gpu-seconds-cap": "5400",
              "--lower-seconds": "180", "--backend-seconds": "1800", "--route-seconds": "2100"}
    for flag, value in values.items():
        command[command.index(flag)+1] = value
    return command


def preflight_command():
    return [sys.executable, "-B", str(CODE/"scripts/hpc/check_daily_training_runtime.py"),
            "--backend", "gpu", "--output", str(DIAG/(PREFIX+".runtime-preflight.json"))]


def load_probe():
    path = OVERLAY / "probe_shared_latent_T7_raw_neural_jvp.py"
    spec = importlib.util.spec_from_file_location("first_day_probe_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    args = parser().parse_args(argv)
    command = child_command(args)
    if args.dry_run:
        print(json.dumps({"command": shlex.join(command), "argv": command,
            "cwd": str(CODE), "runtime_preflight_command": shlex.join(preflight_command()),
            "runtime_preflight_action": "new check after approval only",
            "runtime_preflight_cap_seconds": 240, "outer_cap_seconds": 5400,
            "probe_cap_seconds": 4800, "exit_reserve_seconds": 360,
            "GPU_executed": False, "planned_JVP_calls": 24, "saved_executable_load_count": 0,
            "planned_model_compiles": {"B1": 1, "B4": 1}, "optimizer_updates": 0}, indent=2))
        return 0
    if not CODE.is_dir() or not OVERLAY.is_dir():
        raise FileNotFoundError("declared immutable checkout and isolated overlay required")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(CODE), str(OVERLAY), str(COMMON), env.get("PYTHONPATH", "")))
    os.chdir(CODE)
    if args.stage == "execute" and not args.plan_only:
        guard = load_probe()
        child_args = guard.parser().parse_args(command[3:])
        guard.validate_caps(child_args)
        guard.validate_proposal(child_args, execution=True)
        output = Path(preflight_command()[-1])
        if output.exists():
            raise FileExistsError("new boundary preflight output already exists")
        result = subprocess.run(preflight_command(), cwd=CODE, env=env, check=False, timeout=240)
        if result.returncode:
            return result.returncode
    return subprocess.run(command, cwd=CODE, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
