"""Build a process- and field-level diagnostic from a canonical rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    daily_markov_contract_from_metadata,
)


def _series_summary(series: list[tuple[int, Mapping[str, Any]]]) -> dict[str, Any]:
    first_day, first = series[0]
    final_day, final = series[-1]
    thresholds = (0.1, 0.25, 0.5, 1.0)
    return {
        "first_day": first_day,
        "first_normalized_rmse": first["normalized_rmse"],
        "first_uniform_loss_share": first.get("current_uniform_loss_share"),
        "final_day": final_day,
        "final_normalized_rmse": final["normalized_rmse"],
        "final_uniform_loss_share": final.get("current_uniform_loss_share"),
        "peak_normalized_rmse": max(
            float(metrics["normalized_rmse"]) for _, metrics in series
        ),
        "first_threshold_crossings": {
            str(threshold): next(
                (
                    day
                    for day, metrics in series
                    if float(metrics["normalized_rmse"]) >= threshold
                ),
                None,
            )
            for threshold in thresholds
        },
    }


def _nominal_state_weights(contract: DailyMarkovContract) -> dict[str, Any]:
    width = contract.continuous_state_width
    components: dict[str, dict[str, int | float]] = {}
    for leaf in contract.state_leaves:
        if leaf.discrete:
            continue
        record = components.setdefault(
            leaf.component,
            {"state_values": 0, "state_leaves": 0},
        )
        record["state_values"] = int(record["state_values"]) + int(
            leaf.stop - leaf.start
        )
        record["state_leaves"] = int(record["state_leaves"]) + 1
    for record in components.values():
        record["nominal_weight_share"] = float(record["state_values"]) / width
    return {
        "policy": "uniform_per_continuous_state_value",
        "continuous_state_width": width,
        "per_value_weight": 1.0 / width,
        "components": components,
    }


def build_drift_report(
    rollout: Mapping[str, Any],
    contract: DailyMarkovContract,
) -> dict[str, Any]:
    if rollout["contract_sha256"] != contract.sha256:
        raise ValueError("rollout and dataset contract hashes differ")
    days = rollout.get("days", [])
    if not days:
        raise ValueError("rollout contains no days")
    if any("leaves" not in day["state"] for day in days):
        raise ValueError("rollout predates complete field-level diagnostics")

    component_series: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    leaf_series: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    leaf_metadata: dict[str, dict[str, Any]] = {}
    for record in days:
        day = int(record["day_index"])
        for component, metrics in record["state"]["components"].items():
            if metrics["normalized_rmse"] is not None:
                component_series.setdefault(component, []).append((day, metrics))
        for leaf in record["state"]["leaves"]:
            if leaf["normalized_rmse"] is None:
                continue
            key = str(leaf["key"])
            leaf_series.setdefault(key, []).append((day, leaf))
            leaf_metadata[key] = {
                "component": leaf["component"],
                "source_ref": leaf["source_ref"],
                "width": leaf["width"],
                "nominal_weight_share": leaf["width"]
                / contract.continuous_state_width,
            }

    components = {
        key: _series_summary(series) for key, series in component_series.items()
    }
    leaves = {
        key: leaf_metadata[key] | _series_summary(series)
        for key, series in leaf_series.items()
    }
    first_rank = sorted(
        leaves,
        key=lambda key: leaves[key]["first_normalized_rmse"],
        reverse=True,
    )
    final_rank = sorted(
        leaves,
        key=lambda key: leaves[key]["final_normalized_rmse"],
        reverse=True,
    )
    return {
        "schema_version": "canonical_rollout_drift_report_v1",
        "rollout": {
            "dataset_id": rollout["dataset_id"],
            "checkpoint": rollout["checkpoint"],
            "selection": rollout["selection"],
            "summary": rollout["summary"],
        },
        "state_loss_weighting": _nominal_state_weights(contract),
        "daily_overall_normalized_rmse": [
            {
                "day_index": day["day_index"],
                "normalized_rmse": day["state"]["normalized_rmse"],
            }
            for day in days
        ],
        "components": components,
        "leaves": leaves,
        "largest_first_day_leaves": first_rank[:20],
        "largest_final_day_leaves": final_rank[:20],
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    selection = report["rollout"]["selection"]
    lines = [
        "# Canonical Seven-Day Rollout Drift Diagnostic",
        "",
        "## Scope",
        "",
        f"- Landpoint: `{selection['landpoint_id']}`",
        f"- Year and days: `{selection['year']}`, "
        f"`{selection['start_day']}`-"
        f"`{selection['start_day'] + selection['days'] - 1}`",
        f"- Feedback: `{selection['state_feedback']}`",
        "- Sealed test split used: `false`",
        "",
        "## Daily Overall Error",
        "",
        "| Day | Normalized RMSE |",
        "|---:|---:|",
    ]
    for day in report["daily_overall_normalized_rmse"]:
        lines.append(f"| {day['day_index']} | {day['normalized_rmse']:.6f} |")

    lines.extend(
        [
            "",
            "## Current State-Loss Weighting",
            "",
            "The current multistep objective assigns equal nominal weight to each "
            "continuous state value. Wider components therefore receive more total "
            "nominal weight. Actual shares below are the fractions of the uniform "
            "Huber state loss on the first and final rollout days.",
            "",
            "| Component | Values | Nominal share | First-day loss | Final-day loss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    weighting = report["state_loss_weighting"]["components"]
    for component, values in sorted(
        weighting.items(),
        key=lambda item: item[1]["nominal_weight_share"],
        reverse=True,
    ):
        metrics = report["components"][component]
        lines.append(
            f"| `{component}` | {values['state_values']} | "
            f"{_percent(values['nominal_weight_share'])} | "
            f"{_percent(metrics['first_uniform_loss_share'])} | "
            f"{_percent(metrics['final_uniform_loss_share'])} |"
        )

    def add_leaf_table(title: str, keys: Sequence[str], day_prefix: str) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| State field | Source | Width | Normalized RMSE | Loss share |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for key in keys:
            leaf = report["leaves"][key]
            lines.append(
                f"| `{key}` | `{leaf['source_ref']}` | {leaf['width']} | "
                f"{leaf[f'{day_prefix}_normalized_rmse']:.6f} | "
                f"{_percent(leaf[f'{day_prefix}_uniform_loss_share'])} |"
            )

    add_leaf_table(
        "Largest First-Day Field Errors",
        report["largest_first_day_leaves"][:15],
        "first",
    )
    add_leaf_table(
        "Largest Final-Day Field Errors",
        report["largest_final_day_leaves"][:15],
        "final",
    )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This report identifies where recursive error first appears and how the "
            "current objective weights it. It does not by itself prescribe new "
            "scientific weights or establish long-horizon stability.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(dataset["markov_contract"])
    report = build_drift_report(rollout, contract)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
