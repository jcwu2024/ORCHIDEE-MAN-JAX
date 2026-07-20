from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.paper_validation import (  # noqa: E402
    FEATURE_NAMES,
    load_paper_validation_points,
    select_representative_paper_points,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a deterministic paper-landpoint validation matrix.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Landpoint ID to remove before maximin selection; repeat as needed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reference_mode" / "paper_validation_selection.json",
    )
    args = parser.parse_args()

    excluded = frozenset(args.exclude)
    points = tuple(
        point
        for point in load_paper_validation_points(args.root)
        if point.landpoint_id not in excluded
    )
    selected = select_representative_paper_points(points, count=args.count, include_ids=args.include)
    payload = {
        "description": (
            "Deterministic maximin sample over landpoint grid indices, four calibrated PFT14 parameters, "
            "and archived Fortran AGB/BGB/GPP/NPP targets. This is a runtime validation sample, not a "
            "substitute for source-driven branch coverage."
        ),
        "population_count": len(points),
        "excluded_landpoint_ids": sorted(excluded),
        "selected_count": len(selected),
        "feature_names": list(FEATURE_NAMES),
        "selected_landpoint_ids": [point.landpoint_id for point in selected],
        "selected": [point.to_json_row() for point in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
