from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.daily_coarse_graining.daily_flux_label_inventory import (
    DEFAULT_INVENTORY_PATH,
    audit_daily_flux_label_inventory,
    load_daily_flux_label_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen daily flux label inventory")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset_manifest.resolve().read_text(encoding="utf-8"))
    report = audit_daily_flux_label_inventory(
        load_daily_flux_label_inventory(args.inventory),
        dataset["markov_contract"],
        contract_sha256=str(dataset["markov_contract_sha256"]),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
