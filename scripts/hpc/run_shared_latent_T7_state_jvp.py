"""Resolve the isolated Explore1000 overlay command for the T7 state-JVP probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime")
CODE = ROOT / "checkouts/shared-split-width-v1"
OVERLAY = ROOT / "transfer/T7-state-jvp-20261009"
COMPLETION_OVERLAY = ROOT / "transfer/T7-B1-state-jvp-completion-20261009"
COMMON = ROOT / "transfer/shared-latent-reader-20261004"
DIAG = ROOT / "outputs/diagnostics/shared-latent-reader-20261004"
TRAIN = ROOT / "outputs/training"
PREFIX = "T7-state-jvp-20261009"
RECOVERY_PREFIX = "T7-state-jvp-20261009-B4-recovery"
COMPLETION_PREFIX = "T7-B1-state-jvp-completion-20261009"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cpu-readiness", "execute"), required=True)
    p.add_argument("--output-prefix", default=PREFIX)
    p.add_argument("--runtime-preflight", type=Path,
                   default=DIAG / (PREFIX + ".runtime-preflight.json"))
    p.add_argument("--state-input-metadata", type=Path,
                   default=DIAG / "T7-state-inputs-20261009.json")
    p.add_argument("--approval-report", type=Path, default=DIAG / (PREFIX + ".approval.json"))
    p.add_argument("--repair-record", type=Path)
    p.add_argument("--reuse-runtime-preflight", action="store_true")
    p.add_argument("--b1-completion", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def child_command(args):
    completion = bool(getattr(args, "b1_completion", False))
    prefix = COMPLETION_PREFIX if completion and args.output_prefix == PREFIX else args.output_prefix
    approval_path = args.approval_report
    if completion and approval_path == DIAG / (PREFIX + ".approval.json"):
        suffix = ".approval.json" if args.stage == "cpu-readiness" else ".approved.json"
        approval_path = DIAG / (COMPLETION_PREFIX + suffix)
    output = DIAG / (prefix + (".cpu-readiness.json" if args.stage == "cpu-readiness" else ".result.json"))
    prepared = ROOT / "outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
    dataset = ROOT / "outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
    flags = {
        "--manifest": COMMON / "latent-reader-T7-20261005.manifest.json",
        "--prepared-context": ROOT / "outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl",
        "--parameters": ROOT / "outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl",
        "--field-adapter": ROOT / "transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py",
        "--backbone-adapter": ROOT / "transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py",
        "--transformer-module": COMMON / "shared_transformer.py",
        "--latent-module": COMMON / "shared_latent_reader.py",
        "--turnover-overlay": COMMON / "turnover_leaf_age_fall.guarded-source.py",
        "--runtime-preflight": args.runtime_preflight,
        "--state-input-metadata": args.state_input_metadata,
        "--training-prepared-manifest": prepared,
        "--dataset-manifest": dataset,
        "--b1-executable": ROOT / "outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-compile-candidate.executable.pkl",
        "--b4-executable": ROOT / "outputs/diagnostics/latent-reader-T7-batch-20261009-paid.replacement.executable.pkl",
        "--paid-pair-envelope": DIAG / "T7-state-repeats-20261009.analysis.json",
        "--approval-report": approval_path,
        "--output": output,
    }
    repair_record = getattr(args, "repair_record", None)
    if repair_record is not None:
        flags["--repair-record"] = repair_record
    route_mode = "B4-only" if repair_record is not None else ("B1-completion" if completion else "both")
    if completion:
        flags["--b4-jvp-executable"] = ROOT / "outputs/diagnostics/shared-latent-reader-20261004/T7-state-jvp-20261009-B4-recovery.result.b4.jvp.executable.pkl"
        flags["--output"] = output
    overlay = COMPLETION_OVERLAY if completion else OVERLAY
    command = [sys.executable, "-B", str(overlay / "probe_shared_latent_T7_state_jvp.py"),
               "--stage", args.stage, "--route-mode", route_mode,
               "--horizon", "7", "--force-compilation-parallelism", "4"]
    for flag, path in flags.items():
        command.extend((flag, str(path)))
    if getattr(args, "plan_only", False):
        command.append("--plan-only")
    if route_mode == "B4-only":
        command.extend(("--wall-seconds", "2100", "--gpu-seconds-cap", "2400", "--lower-seconds", "180",
                        "--backend-seconds", "1500", "--route-seconds", "1800", "--load-seconds", "300"))
    elif route_mode == "B1-completion":
        command.extend(("--wall-seconds", "3300", "--gpu-seconds-cap", "3600", "--lower-seconds", "180",
                        "--backend-seconds", "1800", "--route-seconds", "2400", "--load-seconds", "300"))
    else:
        command.extend(("--wall-seconds", "3060", "--gpu-seconds-cap", "3600", "--lower-seconds", "180",
                        "--backend-seconds", "900", "--route-seconds", "1500", "--load-seconds", "300"))
    return command


def main():
    args = parser().parse_args()
    completion = bool(getattr(args, "b1_completion", False))
    if args.dry_run:
        args.plan_only = True
    if completion and args.output_prefix == PREFIX:
        args.output_prefix = COMPLETION_PREFIX
    if completion and args.approval_report == DIAG / (PREFIX + ".approval.json"):
        suffix = ".approval.json" if args.stage == "cpu-readiness" else ".approved.json"
        args.approval_report = DIAG / (COMPLETION_PREFIX + suffix)
    command = child_command(args)
    if args.dry_run:
        route_mode = ("B4-only" if getattr(args, "repair_record", None) is not None else
                      ("B1-completion" if getattr(args, "b1_completion", False) else "both"))
        preflight = None if args.reuse_runtime_preflight else [sys.executable, "-B",
            str(CODE / "scripts/hpc/check_daily_training_runtime.py"), "--backend", "gpu",
            "--output", str(args.runtime_preflight)]
        print(json.dumps({"command": shlex.join(command), "runtime_preflight_command":
                          shlex.join(preflight) if preflight else None,
                          "runtime_preflight_action": "reuse-existing" if preflight is None else "execute",
                          "runtime_preflight_path": str(args.runtime_preflight), "cwd": str(CODE),
                          "gpu_execution": False, "route_mode": route_mode,
                          "probe_model_compiles": {"B1_JVP": int(route_mode in ("both", "B1-completion") and args.stage == "execute"),
                              "B4_JVP": int(route_mode in ("both", "B4-only") and args.stage == "execute")},
                          "saved_executables_to_load": ["B1_VJP", "B4_VJP", "B4_JVP"]
                              if route_mode == "B1-completion" else ["B1_VJP", "B4_VJP"],
                          "JVP_calls": 0, "planned_JVP_calls": 16 if route_mode == "B4-only" else 32,
                          "planned_saved_VJP_calls": 4, "optimizer_updates": 0}, indent=2))
        return 0
    overlay = COMPLETION_OVERLAY if completion else OVERLAY
    if not CODE.is_dir() or not overlay.is_dir():
        raise FileNotFoundError("immutable checkout and isolated JVP overlay must be staged first")
    os.chdir(CODE)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(CODE), str(overlay), str(COMMON), environment.get("PYTHONPATH", "")))
    environment.setdefault("XLA_FLAGS", "")
    if args.stage == "execute":
        from types import SimpleNamespace
        try:
            from scripts.hpc import probe_shared_latent_T7_state_jvp as probe
        except ImportError:
            import probe_shared_latent_T7_state_jvp as probe
        probe._approval(SimpleNamespace(approval_report=args.approval_report,
            output=DIAG / (args.output_prefix + ".result.json"),
            route_mode=("B4-only" if args.repair_record is not None else
                        ("B1-completion" if getattr(args, "b1_completion", False) else "both")),
            repair_record=args.repair_record))
        preflight = [sys.executable, "-B", str(CODE / "scripts/hpc/check_daily_training_runtime.py"),
                     "--backend", "gpu", "--output", str(args.runtime_preflight)]
        if args.reuse_runtime_preflight:
            if not args.runtime_preflight.is_file():
                raise FileNotFoundError("requested prior runtime preflight is missing")
        else:
            if args.runtime_preflight.exists():
                raise FileExistsError(args.runtime_preflight)
            try:
                ready = subprocess.run(preflight, cwd=CODE, env=environment, check=False, timeout=240)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("actual JAX/CUDA runtime preflight exceeded240s") from error
            if ready.returncode != 0:
                raise RuntimeError("actual JAX/CUDA runtime preflight failed; no model graph compile started")
    return subprocess.run(command, cwd=CODE, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
