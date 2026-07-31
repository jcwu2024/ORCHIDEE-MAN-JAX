"""Fail-closed summary of a completed Experiment C all-sample screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SUMMARY_SCHEMA_VERSION = "causal_carbon_adapter_screening_summary_v1"
EXPECTED_REPORT_SCHEMA_VERSION = "causal_carbon_adapter_screening_report_v1"
GATE_FAMILIES = (
    "interface",
    "global_state",
    "primary_state",
    "flux_state_bias_all_leads",
    "stock_tendency_bias_all_leads",
    "guard",
)
SELECTION_SLICES = ("temporal", "spatial", "joint")
HORIZONS = (1, 7, 30)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio_coordinates(gate_id: str) -> tuple[str, str, int]:
    parts = gate_id.split("/")
    family = parts[0]
    if family not in GATE_FAMILIES or len(parts) < 3:
        raise ValueError(f"unknown Experiment C ratio gate id {gate_id}")
    slice_id = parts[1]
    if slice_id not in SELECTION_SLICES:
        raise ValueError(f"unknown Experiment C screening slice {slice_id}")
    if family == "interface":
        horizon = 1
    else:
        horizon_record = parts[2]
        if not horizon_record.startswith("horizon_"):
            raise ValueError(f"missing Experiment C gate horizon in {gate_id}")
        horizon = int(horizon_record.removeprefix("horizon_"))
    if horizon not in HORIZONS:
        raise ValueError(f"unknown Experiment C screening horizon {horizon}")
    return family, slice_id, horizon


def _counter_records(
    totals: Counter[Any],
    failures: Counter[Any],
    keys: Sequence[Any],
) -> list[Mapping[str, Any]]:
    return [
        {
            "id": str(key),
            "total": int(totals[key]),
            "failed": int(failures[key]),
            "passed": int(totals[key] - failures[key]),
        }
        for key in keys
    ]


def summarize_screening(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Summarize only a complete promotion-eligible all-sample report."""

    if report.get("schema_version") != EXPECTED_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported Experiment C screening report schema")
    if report.get("promotion_eligible") is not True:
        raise ValueError("Experiment C summary refuses smoke or partial evidence")
    if report.get("sealed_test_used") is not False:
        raise ValueError("Experiment C summary refuses sealed-test evidence")
    classification = report.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError("Experiment C summary requires a completed classification")

    ratios = classification.get("ratio_gates")
    hard = classification.get("hard_constraints")
    structural = classification.get("structural_constraints")
    if not isinstance(ratios, list) or len(ratios) != 165:
        raise ValueError("Experiment C summary requires all 165 ratio gates")
    if not isinstance(hard, list) or len(hard) != 108:
        raise ValueError("Experiment C summary requires all 108 hard gates")
    if not isinstance(structural, list) or len(structural) != 6:
        raise ValueError("Experiment C summary requires all six structural gates")

    family_totals: Counter[str] = Counter()
    family_failures: Counter[str] = Counter()
    slice_totals: Counter[str] = Counter()
    slice_failures: Counter[str] = Counter()
    horizon_totals: Counter[int] = Counter()
    horizon_failures: Counter[int] = Counter()
    failed_ratio_records = []
    failed_coordinates: set[tuple[str, str, int]] = set()
    for record in ratios:
        gate_id = str(record["id"])
        family, slice_id, horizon = _ratio_coordinates(gate_id)
        family_totals[family] += 1
        slice_totals[slice_id] += 1
        horizon_totals[horizon] += 1
        if record.get("passed") is not True:
            family_failures[family] += 1
            slice_failures[slice_id] += 1
            horizon_failures[horizon] += 1
            failed_coordinates.add((family, slice_id, horizon))
            failed_ratio_records.append(
                {
                    "id": gate_id,
                    "ratio": record.get("ratio"),
                    "threshold_max": record.get("threshold_max"),
                    "candidate": record.get("candidate"),
                    "control": record.get("control"),
                }
            )

    def _severity(record: Mapping[str, Any]) -> float:
        ratio = record["ratio"]
        if ratio is None:
            return float("inf")
        threshold = float(record["threshold_max"])
        return float(ratio) / threshold

    failed_ratio_records.sort(key=_severity, reverse=True)
    hard_failures = [str(record["id"]) for record in hard if record.get("passed") is not True]
    structural_failures = [str(record["id"]) for record in structural if record.get("passed") is not True]

    def failed(families: set[str], *, horizon: int | None = None, slices: set[str] | None = None) -> bool:
        return any(
            family in families
            and (horizon is None or gate_horizon == horizon)
            and (slices is None or slice_id in slices)
            for family, slice_id, gate_horizon in failed_coordinates
        )

    flags = {
        "execution_or_state_integrity_failure": bool(hard_failures or structural_failures),
        "one_day_causal_interface_failure": failed({"interface"}),
        "one_day_primary_state_failure": failed({"global_state", "primary_state"}, horizon=1),
        "seven_day_recursive_failure": failed(set(GATE_FAMILIES) - {"interface"}, horizon=7),
        "thirty_day_recursive_failure": failed(set(GATE_FAMILIES) - {"interface"}, horizon=30),
        "spatial_generalization_failure": failed(set(GATE_FAMILIES), slices={"spatial"}),
        "joint_generalization_failure": failed(set(GATE_FAMILIES), slices={"joint"}),
        "carbon_flux_or_stock_bias_failure": failed({"flux_state_bias_all_leads", "stock_tendency_bias_all_leads"}),
        "litter_or_doc_guard_regression": failed({"guard"}),
    }
    status = str(classification.get("status"))
    expected_status = (
        "passed" if not failed_ratio_records and not hard_failures and not structural_failures else "rejected"
    )
    if status != expected_status:
        raise ValueError("Experiment C classification status contradicts its gates")

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "screening_status": status,
        "screening_decision": classification.get("decision"),
        "ratio_gates": {
            "total": len(ratios),
            "failed": len(failed_ratio_records),
            "by_family": _counter_records(
                family_totals,
                family_failures,
                GATE_FAMILIES,
            ),
            "by_slice": _counter_records(
                slice_totals,
                slice_failures,
                SELECTION_SLICES,
            ),
            "by_horizon": _counter_records(
                horizon_totals,
                horizon_failures,
                HORIZONS,
            ),
            "worst_failures": failed_ratio_records[:20],
        },
        "hard_gates": {
            "total": len(hard),
            "failed": len(hard_failures),
            "failed_ids": hard_failures,
        },
        "structural_gates": {
            "total": len(structural),
            "failed": len(structural_failures),
            "failed_ids": structural_failures,
        },
        "interpretation_flags": flags,
        "sealed_test_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = dict(summarize_screening(report))
    summary["source_report"] = {
        "path": str(report_path),
        "sha256": _sha256_file(report_path),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
