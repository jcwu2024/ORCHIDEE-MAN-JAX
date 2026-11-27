from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jax_orchidee.runners.run_identity import (  # noqa: E402
    RESULT_PROTOCOL,
    assigned_site_ids,
    read_selection_ids,
    write_json,
)


def _load_summaries(paths: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)


def _check_protocol(summaries: tuple[dict[str, object], ...]) -> int | None:
    if not summaries:
        raise ValueError("no acceptance summaries found")
    keys = ("tolerance_policy", "start_year", "years", "days_per_year", "initial_state", "result_protocol")
    protocol = {key: summaries[0].get(key) for key in keys}
    for summary in summaries[1:]:
        if any(summary.get(key) != value for key, value in protocol.items()):
            raise ValueError("acceptance shards use different scientific requests or result protocols")
    result_protocol = protocol["result_protocol"]
    if result_protocol not in (None, 1, RESULT_PROTOCOL):
        raise ValueError("unknown acceptance result protocol")
    batch_ids = set()
    if result_protocol is not None:
        for summary in summaries:
            for row in summary.get("results", []):
                run = row.get("compiled_run") or {}
                batch_id = run.get("batch_identity")
                if batch_id:
                    batch_ids.add(batch_id)
                if row.get("scientific_acceptance_pass") and (
                    not batch_id or run.get("returncode") != 0 or row.get("classification") != "accepted_compiled"
                ):
                    raise ValueError(f"accepted point {row.get('landpoint_id')} lacks a successful identified compiled run")
    if len(batch_ids) > 1:
        raise ValueError("acceptance shards use different production environments or shared inputs")
    return result_protocol


def _complete_shard_set(summaries: tuple[dict[str, object], ...]) -> bool:
    counts = {summary.get("shard_count", 1) for summary in summaries}
    if counts == {1}:
        # An array of independent unsharded point jobs is also supported.
        return True
    if len(counts) != 1:
        raise ValueError("acceptance shards use different shard counts")
    count = counts.pop()
    indices = {summary.get("shard_index") for summary in summaries}
    return len(summaries) == count and indices == set(range(count))


def _valid_roster(ids: object) -> bool:
    return (
        isinstance(ids, (tuple, list))
        and bool(ids)
        and all(isinstance(site, str) and site for site in ids)
        and len(set(ids)) == len(ids)
    )


def _selection_errors(
    summaries: tuple[dict[str, object], ...], expected: tuple[str, ...] | None,
) -> list[str]:
    errors: list[str] = []
    sharded_plan = None
    for summary in summaries:
        declared = summary.get("selection_ids")
        index, count = summary.get("shard_index"), summary.get("shard_count")
        limit = summary.get("max_points")
        if not (
            _valid_roster(declared)
            and type(index) is int and type(count) is int and 0 <= index < count
            and "max_points" in summary
            and (limit is None or (type(limit) is int and limit > 0))
        ):
            errors.append("summary has an invalid selection or shard assignment")
            continue
        roster = tuple(declared)
        received = tuple(row["landpoint_id"] for row in summary.get("results", []))
        assigned = assigned_site_ids(roster, index, count, limit)
        if received != assigned:
            errors.append(f"shard index {index} has results that differ from its assigned site IDs or order")
        if count > 1:
            plan = (roster, count, limit)
            if sharded_plan is None:
                sharded_plan = plan
                # Applying a per-shard cap after round-robin assignment
                # retains the first count * limit sites across all shards.
                capped_roster = roster if limit is None else roster[:count * limit]
                if expected is not None and expected not in (roster, capped_roster):
                    errors.append("declared sharded selection differs from the expected ordered roster")
            elif plan != sharded_plan:
                errors.append(f"shard index {index} declares a different ordered selection or assignment rule")
    return errors


def aggregate_summaries(
    summaries: tuple[dict[str, object], ...],
    *,
    expected_landpoints: int | None = None,
    expected_landpoint_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, object]:
    result_protocol = _check_protocol(summaries)
    expected = None
    if expected_landpoint_ids is not None:
        if not _valid_roster(expected_landpoint_ids):
            raise ValueError("expected roster must contain unique nonempty site IDs")
        expected = tuple(expected_landpoint_ids)
        if expected_landpoints is not None and expected_landpoints != len(expected):
            raise ValueError("expected landpoint count differs from the expected roster size")
        expected_landpoints = len(expected)

    rows = [row for summary in summaries for row in summary.get("results", [])]
    ids = [str(row["landpoint_id"]) for row in rows]
    duplicates = sorted(landpoint_id for landpoint_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate landpoints across acceptance shards: {duplicates[:5]}")
    missing = [] if expected is None else sorted(set(expected) - set(ids))
    unexpected = [] if expected is None else sorted(set(ids) - set(expected))
    current_protocol = result_protocol == RESULT_PROTOCOL
    selection_errors = _selection_errors(summaries, expected) if current_protocol else []
    complete = (
        bool(rows)
        and _complete_shard_set(summaries)
        and (expected_landpoints is None or len(rows) == int(expected_landpoints))
        and not (missing or unexpected or selection_errors)
    )
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
    return {
        "tolerance_policy": summaries[0]["tolerance_policy"],
        "shard_count": len(summaries),
        "expected_landpoint_count": expected_landpoints,
        "expected_landpoint_ids": expected,
        "landpoint_count": len(rows),
        "complete_landpoint_set": complete,
        "result_protocol": result_protocol,
        "identified_result_protocol": result_protocol is not None,
        "missing_landpoint_ids": missing,
        "unexpected_landpoint_ids": unexpected,
        "selection_errors": selection_errors,
        "scientific_acceptance_count": len(accepted),
        "scientific_failure_count": len(failed),
        "all_scientifically_accepted": current_protocol and expected is not None and complete and not failed,
        "classification_counts": dict(sorted(classifications.items())),
        "failed_landpoint_ids": [str(row["landpoint_id"]) for row in failed],
        "worst_compiled_scientific_metrics": worst,
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate and validate paper-landpoint acceptance shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="summary*.json")
    parser.add_argument("--expected-landpoints", type=int, default=None)
    parser.add_argument(
        "--expected-selection", type=Path,
        help="Manifest with the expected selected_landpoint_ids, required for formal acceptance.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = tuple(sorted(args.input_dir.glob(args.pattern)))
    summaries = _load_summaries(paths)
    expected_ids = None if args.expected_selection is None else read_selection_ids(args.expected_selection)
    payload = aggregate_summaries(
        summaries, expected_landpoints=args.expected_landpoints, expected_landpoint_ids=expected_ids,
    )
    output = args.output or args.input_dir / "aggregate.json"
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "landpoint_count": payload["landpoint_count"],
                "scientific_acceptance_count": payload["scientific_acceptance_count"],
                "scientific_failure_count": payload["scientific_failure_count"],
                "complete_landpoint_set": payload["complete_landpoint_set"],
                "all_scientifically_accepted": payload["all_scientifically_accepted"],
                "missing_landpoint_ids": payload["missing_landpoint_ids"],
                "unexpected_landpoint_ids": payload["unexpected_landpoint_ids"],
                "selection_errors": payload["selection_errors"],
            },
            indent=2,
        )
    )
    return 0 if payload["all_scientifically_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
