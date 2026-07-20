from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.paper_binding import (  # noqa: E402
    PAPER_POINT_BINDING_OWNERS,
    PAPER_POINT_VARYING_RUN_DEF_KEYS,
    audit_all_paper_landpoint_bindings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit all paper landpoint input and parameter bindings.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "acceptance" / "paper_landpoint_binding_audit",
    )
    args = parser.parse_args()

    rows = audit_all_paper_landpoint_bindings(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_rows = [row.to_json_row() for row in rows]
    failed = [row for row in rows if not row.passed]
    summary = {
        "status": "passed" if len(rows) == 669 and not failed else "failed",
        "landpoint_count": len(rows),
        "passed_count": sum(row.passed for row in rows),
        "failed_count": len(failed),
        "failed_landpoint_ids": [row.landpoint_id for row in failed],
        "varying_run_def_keys": list(PAPER_POINT_VARYING_RUN_DEF_KEYS),
        "binding_owners": PAPER_POINT_BINDING_OWNERS,
        "scope": (
            "Finite paper-package binding gate: run.def -> Fortran used_run.def -> JAX consumer, "
            "forcing-domain identity, and per-landpoint cold-start/restart asset routing."
        ),
        "not_proven_by_this_gate": "full-chain numerical equivalence",
    }
    (args.output_dir / "points.json").write_text(json.dumps(point_rows, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("landpoint_id", "passed", "package_signature_sha256", "errors"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "landpoint_id": row.landpoint_id,
                    "passed": row.passed,
                    "package_signature_sha256": row.package_signature_sha256,
                    "errors": " | ".join(row.errors),
                }
            )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
