"""Audit daily-surrogate carbon ownership against a dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.daily_coarse_graining.carbon_budget_ownership import (
    audit_carbon_budget_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    report = audit_carbon_budget_contract(manifest["markov_contract"])
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
