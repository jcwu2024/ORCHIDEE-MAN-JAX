"""Isolated 7200 card-second grant for one-card leaf full-network verification."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

LIMIT = 7200
SCOPE = "leaf_fraction_full_network_cuda_7200_20261005"
LEDGER_NAME = "leaf-full-state-cuda-20261005.gpu-budget.json"
APPROVAL = ("User authorization 2026-10-05: independent7200 GPU-seconds; one idle free V100; "
            "original network, frozen old/new step1 snapshots, artifact275 rows16/17; "
            "one corrected full parameter/state AD compilation; no optimizer or paid jobs. "
            "Startup, compilation, repairs, persistence and cleanup are charged. "
            "Historical funding is excluded.")


def validate_grant(data):
    if (data.get("limit_gpu_seconds") != LIMIT or data.get("session_limit_seconds") != LIMIT
            or data.get("independent_funding_scope") != SCOPE or data.get("approval_note") != APPROVAL):
        raise ValueError("independent leaf grant identity differs; historical funds cannot be used")
    for row in data.get("sessions", ()):
        if row.get("cards") != 1 or not 20 <= row.get("session_cap_seconds", 0) <= LIMIT:
            raise ValueError("leaf grant contains an invalid multi-card/session record")


def run(args):
    if args.ledger.name != LEDGER_NAME or not 20 <= args.seconds <= LIMIT:
        raise ValueError("exact independent leaf ledger and bounded session required")
    source = Path(__file__).with_name("shared_gpu_budget.py")
    name = "leaf_isolated_occupied_gpu_budget"
    spec = importlib.util.spec_from_file_location(name, source)
    budget = importlib.util.module_from_spec(spec)
    sys.modules[name] = budget
    spec.loader.exec_module(budget)
    # Reuse audited reservation/TERM/KILL/settlement logic in an isolated module
    # instance; neither the historical module file nor its ledgers are modified.
    budget.LIMIT = LIMIT
    with budget.locked(args.ledger) as data:
        if not data.get("sessions") and "independent_funding_scope" not in data:
            data.update(independent_funding_scope=SCOPE, approval_note=APPROVAL,
                        authorized_gpu_seconds=LIMIT, historical_funds_excluded=True)
        validate_grant(data)
    args.cards = 1
    args.approved_limit_gpu_seconds = LIMIT
    args.approval_note = APPROVAL
    args.funding_scope = None
    return budget.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--stage", choices=("runtime-preflight", "full-state", "saved-recovery", "saved-original-control"), required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    raise SystemExit(run(parser.parse_args()))
