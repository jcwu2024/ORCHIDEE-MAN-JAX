"""Stdlib-only environment boundary, shared by dry runs and real CPU children.

Run after srun: Slurm's ALL export precedence must not choose the thread policy.
This module sets execution policy only and never imports JAX/model code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PREFIXES = ("JAX_", "XLA_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_", "TF_NUM_")
WHITELIST = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS", "JAX_ENABLE_X64",
             "PYTHONDONTWRITEBYTECODE")


def policy(budget: int) -> dict[str, str]:
    if budget not in (1, 4, 48):
        raise ValueError("CPU thread budget must be 1, 4 or 48")
    result = {"OMP_NUM_THREADS": str(budget), "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
              "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true",
              "PYTHONDONTWRITEBYTECODE": "1"}
    if budget > 1:
        result["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    return result


def normalized_environment(budget: int, environ=None) -> dict[str, str]:
    inherited = os.environ if environ is None else environ
    result = {key: value for key, value in inherited.items() if not key.startswith(PREFIXES)}
    result.update(policy(budget))
    return result


def snapshot(budget: int, *, status: str, error: str | None = None) -> dict:
    expected = policy(budget)
    inherited = None
    if os.environ.get("ORCHJAX_CHILD_INHERITED_POLICY_JSON"):
        try:
            inherited = json.loads(os.environ["ORCHJAX_CHILD_INHERITED_POLICY_JSON"])
        except (TypeError, ValueError):
            inherited = {"unavailable": "inherited receipt is not valid JSON"}
    return {"schema": "global_cpu_child_prejax_v1", "status": status,
            "pid": os.getpid(), "thread_budget": budget,
            "thread_policy": expected,
            "actual_environment": {key: os.environ.get(key) for key in WHITELIST},
            "unexpected_policy_keys": sorted(key for key in os.environ
                if key.startswith(PREFIXES) and key not in expected),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "JAX_imported": "jax" in sys.modules, "model_execution": False,
            "inherited_transport_policy": inherited,
            "error": error,
            "scope": "pre-JAX child entry; thread budget does not prove allocation or effective parallelism"}


def validate_environment(budget: int) -> None:
    actual = snapshot(budget, status="checking")
    expected = policy(budget)
    mismatches = [key for key in WHITELIST if os.environ.get(key) != expected.get(key)]
    if actual["JAX_imported"] or mismatches or actual["unexpected_policy_keys"]:
        raise RuntimeError("pre-JAX child environment mismatch: " + ",".join(mismatches + actual["unexpected_policy_keys"]))


def wrapped_command(python: str, entry, budget: int, command: list[str]) -> list[str]:
    return [python, str(entry), "--thread-budget", str(budget), "--", *command]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-budget", type=int, choices=(1, 4, 48), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("child command is required")
    inherited = {"environment": {key: os.environ.get(key) for key in WHITELIST},
                 "unexpected_policy_keys": sorted(key for key in os.environ
                     if key.startswith(PREFIXES) and key not in policy(args.thread_budget))}
    clean = normalized_environment(args.thread_budget)
    clean["ORCHJAX_CHILD_INHERITED_POLICY_JSON"] = json.dumps(inherited)
    for key in list(os.environ):
        if key.startswith(PREFIXES):
            os.environ.pop(key)
    os.environ.update(clean)
    validate_environment(args.thread_budget)
    print(json.dumps(snapshot(args.thread_budget, status="exec_boundary_ready")), file=sys.stderr, flush=True)
    try:
        if os.name == "nt":
            # Windows exec emulation does not preserve subprocess capture lifetimes.
            return subprocess.call(command, env=clean)
        os.execvpe(command[0], command, clean)
    except Exception as exc:
        print(json.dumps(snapshot(args.thread_budget, status="exec_boundary_failed", error=str(exc))),
              file=sys.stderr, flush=True)
        raise
    return 1  # exec never returns successfully.


if __name__ == "__main__":
    raise SystemExit(main())
