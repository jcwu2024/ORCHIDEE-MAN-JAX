from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_summaries(paths: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    summaries = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    if not summaries:
        raise ValueError("no acceptance summaries found")
    policy = summaries[0]["tolerance_policy"]
    for summary in summaries[1:]:
        if summary["tolerance_policy"] != policy:
            raise ValueError("acceptance shards use different tolerance policies")
    return summaries


def aggregate_summaries(
    summaries: tuple[dict[str, object], ...],
    *,
    expected_landpoints: int | None,
) -> dict[str, object]:
    rows = [row for summary in summaries for row in summary.get("results", [])]
    ids = [str(row["landpoint_id"]) for row in rows]
    duplicates = sorted(landpoint_id for landpoint_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate landpoints across acceptance shards: {duplicates[:5]}")
    rows.sort(key=lambda row: str(row["landpoint_id"]))
    classifications = Counter(str(row.get("classification", "missing")) for row in rows)
    accepted = [row for row in rows if bool(row.get("scientific_acceptance_pass"))]
    failed = [row for row in rows if not bool(row.get("scientific_acceptance_pass"))]

    metric_rows: dict[str, list[tuple[float, str]]] = {}
    for row in rows:
        evaluation = row.get("compiled_evaluation") or {}
        for variable, values in evaluation.get("series", {}).items():
            for metric in ("max_abs_error", "normalized_bias", "normalized_rmse"):
                value = values.get(metric)
                if value is not None:
                    metric_rows.setdefault(f"{variable}.{metric}", []).append(
                        (abs(float(value)), str(row["landpoint_id"]))
                    )
    worst = {
        metric: {"absolute_value": max(values)[0], "landpoint_id": max(values)[1]}
        for metric, values in sorted(metric_rows.items())
    }
    complete = expected_landpoints is None or len(rows) == int(expected_landpoints)
    return {
        "tolerance_policy": summaries[0]["tolerance_policy"],
        "shard_count": len(summaries),
        "expected_landpoint_count": expected_landpoints,
        "landpoint_count": len(rows),
        "complete_landpoint_set": complete,
        "scientific_acceptance_count": len(accepted),
        "scientific_failure_count": len(failed),
        "all_scientifically_accepted": bool(complete and len(failed) == 0),
        "classification_counts": dict(sorted(classifications.items())),
        "failed_landpoint_ids": [str(row["landpoint_id"]) for row in failed],
        "worst_compiled_scientific_metrics": worst,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate and validate paper-landpoint acceptance shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="summary*.json")
    parser.add_argument("--expected-landpoints", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = tuple(sorted(args.input_dir.glob(args.pattern)))
    summaries = _load_summaries(paths)
    payload = aggregate_summaries(summaries, expected_landpoints=args.expected_landpoints)
    output = args.output or args.input_dir / "aggregate.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "landpoint_count": payload["landpoint_count"],
                "scientific_acceptance_count": payload["scientific_acceptance_count"],
                "scientific_failure_count": payload["scientific_failure_count"],
                "complete_landpoint_set": payload["complete_landpoint_set"],
            },
            indent=2,
        )
    )
    return 0 if payload["all_scientifically_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
