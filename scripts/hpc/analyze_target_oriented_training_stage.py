"""Summarize target-oriented training dynamics and milestone decisions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

COMPONENTS = (
    "one_day_state_anchor",
    "seven_day_science",
    "thirty_day_science",
    "necessary_process_supervision",
)
SHARED_FAMILIES = (
    "base_source_encoder",
    "domain_refinement",
    "science_cross_attention",
    "science_token_mixer",
)
HARD_FAILURE_KEYS = (
    "budget_failure",
    "discrete_mismatch",
    "prediction_nonfinite",
    "state_nonfinite",
    "stock_negative",
    "unexpected_state_mask_mismatch",
)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(statistics.fmean(materialized)) if materialized else math.nan


def _rss(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _stage_start(report: Mapping[str, Any]) -> int:
    identity = report.get("identity", {})
    policy = report.get("policy", {})
    return int(identity.get("stage_start", policy.get("stage_start", 0)))


def _stage_records(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    start = _stage_start(report)
    records = [
        record
        for record in report.get("updates", ())
        if int(record["update"]) > start
    ]
    return sorted(records, key=lambda record: int(record["update"]))


def _record_means(records: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    result = {
        "loss": _mean(record["loss"] for record in records),
        "unclipped_gradient_norm": _mean(
            record["unclipped_gradient_norm"] for record in records
        ),
        "parameter_step_norm": _mean(record["parameter_step_norm"] for record in records),
        "clip_fraction": _mean(float(record["clipped"]) for record in records),
    }
    result.update(
        {
            name: _mean(record["components"][name] for record in records)
            for name in COMPONENTS
        }
    )
    return result


def _detail_means(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    first = records[0]["component_details"]
    groups = sorted(first["one_day_state_anchor"]["group_loss"])
    science_terms = (
        "flux_mse",
        "flux_bias_squared",
        "stock_mse",
        "stock_bias_squared",
    )
    return {
        "one_day_group_loss": {
            group: _mean(
                record["component_details"]["one_day_state_anchor"]["group_loss"][
                    group
                ]
                for record in records
            )
            for group in groups
        },
        "seven_day_science": {
            term: _mean(
                record["component_details"]["seven_day_science"][term]
                for record in records
            )
            for term in science_terms
        },
        "thirty_day_science": {
            term: _mean(
                record["component_details"]["thirty_day_science"][term]
                for record in records
            )
            for term in science_terms
        },
    }


def _block_summaries(
    records: Sequence[Mapping[str, Any]], *, block_size: int
) -> list[Mapping[str, Any]]:
    blocks = []
    for offset in range(0, len(records), block_size):
        block = records[offset : offset + block_size]
        blocks.append(
            {
                "update_start": int(block[0]["update"]),
                "update_end": int(block[-1]["update"]),
                "count": len(block),
                "means": _record_means(block),
                "detail_means": _detail_means(block),
            }
        )
    return blocks


def _window_change(
    records: Sequence[Mapping[str, Any]], *, window_size: int
) -> Mapping[str, Any]:
    window = min(window_size, max(1, len(records) // 2))
    first = _record_means(records[:window])
    last = _record_means(records[-window:])
    relative_change = {
        name: (last[name] - first[name]) / max(abs(first[name]), 1.0e-30)
        for name in first
    }
    return {
        "window_size": window,
        "first": first,
        "last": last,
        "relative_change": relative_change,
    }


def _conditioning_snapshots(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    snapshots = []
    for record in records:
        conditioning = record.get("gradient_conditioning")
        if not conditioning:
            continue
        sources = {}
        for source, families in conditioning.items():
            family_norms = {
                family: float(values["l2_norm"])
                for family, values in families.items()
            }
            sources[source] = {
                "shared_l2_norm": _rss(
                    family_norms.get(family, 0.0) for family in SHARED_FAMILIES
                ),
                "primary_head_l2_norm": family_norms.get("primary_heads", 0.0),
                "state_head_l2_norm": family_norms.get("state_heads", 0.0),
                "auxiliary_head_l2_norm": family_norms.get("auxiliary_heads", 0.0),
                "families": family_norms,
            }
        snapshots.append({"update": int(record["update"]), "sources": sources})
    return snapshots


def _validation_summary(validation: Mapping[str, Any]) -> Mapping[str, Any]:
    key_values: dict[str, list[Mapping[str, Any]]] = {}
    key_horizon_values: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    cells = []
    for slice_name, horizons in validation.get("slices", {}).items():
        for horizon, cell in horizons.items():
            cells.append(
                {
                    "slice": slice_name,
                    "horizon": int(horizon),
                    "passed_keys": sum(
                        bool(value["passed"]) for value in cell["keys"].values()
                    ),
                    "total_keys": len(cell["keys"]),
                    "all_key_gates_passed": bool(cell["all_key_gates_passed"]),
                }
            )
            for name, value in cell["keys"].items():
                key_values.setdefault(name, []).append(value)
                key_horizon_values.setdefault(name, {}).setdefault(
                    str(int(horizon)), []
                ).append(value)
    keys = {}
    for name, values in sorted(key_values.items()):
        keys[name] = {
            "mean_normalized_rmse": _mean(value["normalized_rmse"] for value in values),
            "maximum_normalized_rmse": max(
                float(value["normalized_rmse"]) for value in values
            ),
            "mean_absolute_normalized_bias": _mean(
                abs(float(value["normalized_signed_bias"])) for value in values
            ),
            "maximum_absolute_normalized_bias": max(
                abs(float(value["normalized_signed_bias"])) for value in values
            ),
            "passed_cells": sum(bool(value["passed"]) for value in values),
            "total_cells": len(values),
        }
    hard_counts = {
        name: int(value) for name, value in validation.get("hard_counts", {}).items()
    }
    keys_by_horizon = {
        name: {
            horizon: {
                "mean_normalized_rmse": _mean(
                    value["normalized_rmse"] for value in values
                ),
                "mean_absolute_normalized_bias": _mean(
                    abs(float(value["normalized_signed_bias"])) for value in values
                ),
                "passed_cells": sum(bool(value["passed"]) for value in values),
                "total_cells": len(values),
            }
            for horizon, values in sorted(horizons.items(), key=lambda item: int(item[0]))
        }
        for name, horizons in sorted(key_horizon_values.items())
    }
    return {
        "label": str(validation.get("label", "unknown")),
        "science_macro_normalized_rmse": float(
            validation["science_macro_normalized_rmse"]
        ),
        "absolute_short_horizon_passed": bool(
            validation["absolute_short_horizon_passed"]
        ),
        "hard_failure_count": sum(hard_counts.get(name, 0) for name in HARD_FAILURE_KEYS),
        "hard_counts": hard_counts,
        "cells": cells,
        "keys": keys,
        "keys_by_horizon": keys_by_horizon,
    }


def _validation_timeline(
    report: Mapping[str, Any], parent: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    values = []
    if parent is not None:
        values.extend(
            (
                _validation_summary(parent["initial_validation"]),
                _validation_summary(parent["final_validation"]),
            )
        )
    else:
        values.append(_validation_summary(report["initial_validation"]))
    final = _validation_summary(report["final_validation"])
    if not values or final["label"] != values[-1]["label"]:
        values.append(final)
    return values


def _validation_changes(
    timeline: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    changes = []
    for before, after in zip(timeline, timeline[1:]):
        keys = {}
        for name in sorted(set(before["keys"]) & set(after["keys"])):
            first = before["keys"][name]
            last = after["keys"][name]
            initial_rmse = float(first["mean_normalized_rmse"])
            final_rmse = float(last["mean_normalized_rmse"])
            keys[name] = {
                "initial_mean_normalized_rmse": initial_rmse,
                "final_mean_normalized_rmse": final_rmse,
                "relative_rmse_improvement": (
                    initial_rmse - final_rmse
                ) / max(initial_rmse, 1.0e-30),
                "initial_mean_absolute_normalized_bias": float(
                    first["mean_absolute_normalized_bias"]
                ),
                "final_mean_absolute_normalized_bias": float(
                    last["mean_absolute_normalized_bias"]
                ),
                "final_passed_cells": int(last["passed_cells"]),
                "total_cells": int(last["total_cells"]),
            }
        keys_by_horizon = {}
        for name in sorted(
            set(before["keys_by_horizon"]) & set(after["keys_by_horizon"])
        ):
            keys_by_horizon[name] = {}
            before_horizons = before["keys_by_horizon"][name]
            after_horizons = after["keys_by_horizon"][name]
            for horizon in sorted(
                set(before_horizons) & set(after_horizons), key=int
            ):
                first = before_horizons[horizon]
                last = after_horizons[horizon]
                initial_rmse = float(first["mean_normalized_rmse"])
                final_rmse = float(last["mean_normalized_rmse"])
                keys_by_horizon[name][horizon] = {
                    "initial_mean_normalized_rmse": initial_rmse,
                    "final_mean_normalized_rmse": final_rmse,
                    "relative_rmse_improvement": (
                        initial_rmse - final_rmse
                    ) / max(initial_rmse, 1.0e-30),
                    "final_mean_absolute_normalized_bias": float(
                        last["mean_absolute_normalized_bias"]
                    ),
                    "final_passed_cells": int(last["passed_cells"]),
                    "total_cells": int(last["total_cells"]),
                }
        initial_macro = float(before["science_macro_normalized_rmse"])
        final_macro = float(after["science_macro_normalized_rmse"])
        changes.append(
            {
                "from_label": before["label"],
                "to_label": after["label"],
                "relative_macro_improvement": (
                    initial_macro - final_macro
                ) / max(initial_macro, 1.0e-30),
                "keys": keys,
                "keys_by_horizon": keys_by_horizon,
            }
        )
    return changes


def _milestone_decision(report: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = _validation_summary(report["initial_validation"])
    final = _validation_summary(report["final_validation"])
    initial_macro = baseline["science_macro_normalized_rmse"]
    final_macro = final["science_macro_normalized_rmse"]
    improvement = (initial_macro - final_macro) / max(initial_macro, 1.0e-30)
    hard_passed = final["hard_failure_count"] == 0
    absolute_passed = final["absolute_short_horizon_passed"] and hard_passed
    pacing_passed = improvement >= 0.05
    stage_end = int(report.get("policy", {}).get("stage_end", 0))
    if stage_end <= 0:
        stage_end = max(int(record["update"]) for record in report.get("updates", ()))
    if absolute_passed:
        action = "review_separately_authorized_365_day_gate"
    elif hard_passed and pacing_passed:
        action = (
            "review_update_251_to_1000_continuation"
            if _stage_start(report) == 0 and stage_end <= 250
            else "review_updates_1001_to_3000_continuation"
        )
    else:
        action = "stop_training_and_attribute_failure"
    return {
        "baseline_macro": initial_macro,
        "final_macro": final_macro,
        "five_percent_pacing_macro_maximum": initial_macro * 0.95,
        "relative_macro_improvement": improvement,
        "hard_gates_passed": hard_passed,
        "absolute_short_horizon_passed": absolute_passed,
        "pacing_improvement_passed": pacing_passed,
        "action": action,
    }


def analyze(
    report: Mapping[str, Any],
    *,
    parent: Mapping[str, Any] | None = None,
    block_size: int = 50,
) -> Mapping[str, Any]:
    records = _stage_records(report)
    if not records:
        raise ValueError("target-oriented report contains no records for this stage")
    timeline = _validation_timeline(report, parent)
    return {
        "schema_version": "target_oriented_training_analysis_v1",
        "stage": {
            "start_update": _stage_start(report),
            "end_update": int(records[-1]["update"]),
            "record_count": len(records),
        },
        "training_dynamics": {
            "block_size": block_size,
            "blocks": _block_summaries(records, block_size=block_size),
            "first_to_last_window": _window_change(records, window_size=block_size),
            "gradient_norm_median": float(
                statistics.median(
                    float(record["unclipped_gradient_norm"]) for record in records
                )
            ),
            "gradient_norm_maximum": max(
                float(record["unclipped_gradient_norm"]) for record in records
            ),
            "clip_fraction": _mean(float(record["clipped"]) for record in records),
            "learning_rate_first": float(records[0]["learning_rate"]),
            "learning_rate_last": float(records[-1]["learning_rate"]),
        },
        "gradient_conditioning": _conditioning_snapshots(records),
        "validation_timeline": timeline,
        "validation_changes": _validation_changes(timeline),
        "decision": _milestone_decision(report),
    }


def _markdown(analysis: Mapping[str, Any]) -> str:
    stage = analysis["stage"]
    dynamics = analysis["training_dynamics"]
    decision = analysis["decision"]
    lines = [
        "# Target-Oriented Training Analysis",
        "",
        f"Stage: updates `{stage['start_update'] + 1}..{stage['end_update']}` "
        f"({stage['record_count']} records).",
        "",
        "## Training Blocks",
        "",
        "| Updates | Loss | One-day state | 7-day science | 30-day science | Process | Gradient norm | Clip fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for block in dynamics["blocks"]:
        means = block["means"]
        lines.append(
            f"| {block['update_start']}..{block['update_end']} | {means['loss']:.6g} | "
            f"{means['one_day_state_anchor']:.6g} | {means['seven_day_science']:.6g} | "
            f"{means['thirty_day_science']:.6g} | "
            f"{means['necessary_process_supervision']:.6g} | "
            f"{means['unclipped_gradient_norm']:.6g} | {means['clip_fraction']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Science Loss Geometry",
            "",
            "| Updates | 7d flux MSE | 7d stock MSE | 30d flux MSE | 30d stock MSE |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for block in dynamics["blocks"]:
        details = block["detail_means"]
        seven = details["seven_day_science"]
        thirty = details["thirty_day_science"]
        lines.append(
            f"| {block['update_start']}..{block['update_end']} | "
            f"{seven['flux_mse']:.6g} | {seven['stock_mse']:.6g} | "
            f"{thirty['flux_mse']:.6g} | {thirty['stock_mse']:.6g} |"
        )
    first_groups = dynamics["blocks"][0]["detail_means"]["one_day_group_loss"]
    last_groups = dynamics["blocks"][-1]["detail_means"]["one_day_group_loss"]
    lines.extend(
        (
            "",
            "## One-Day State Groups",
            "",
            "| Group | First block | Last block | Reduction |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    for name in sorted(first_groups):
        first = float(first_groups[name])
        last = float(last_groups[name])
        reduction = (first - last) / max(abs(first), 1.0e-30)
        lines.append(f"| {name} | {first:.6g} | {last:.6g} | {reduction:.3%} |")
    lines.extend(
        (
            "",
            "## Gradient Conditioning",
            "",
            "| Update | Source | Shared backbone | Primary heads | State heads |",
            "| ---: | --- | ---: | ---: | ---: |",
        )
    )
    for snapshot in analysis["gradient_conditioning"]:
        for source, values in snapshot["sources"].items():
            lines.append(
                f"| {snapshot['update']} | {source} | {values['shared_l2_norm']:.6g} | "
                f"{values['primary_head_l2_norm']:.6g} | "
                f"{values['state_head_l2_norm']:.6g} |"
            )
    lines.extend(
        (
            "",
            "## Validation Timeline",
            "",
            "| Label | Science macro | Absolute gate | Hard failures |",
            "| --- | ---: | :---: | ---: |",
        )
    )
    for validation in analysis["validation_timeline"]:
        lines.append(
            f"| {validation['label']} | "
            f"{validation['science_macro_normalized_rmse']:.6g} | "
            f"{'yes' if validation['absolute_short_horizon_passed'] else 'no'} | "
            f"{validation['hard_failure_count']} |"
        )
    if analysis["validation_changes"]:
        latest = analysis["validation_changes"][-1]
        lines.extend(
            (
                "",
                f"## Key Changes: {latest['from_label']} To {latest['to_label']}",
                "",
                "| Key | Mean NRMSE before | Mean NRMSE after | Improvement | Passed cells |",
                "| --- | ---: | ---: | ---: | ---: |",
            )
        )
        horizon_rows = [
            (name, horizon, values)
            for name, horizons in latest["keys_by_horizon"].items()
            for horizon, values in horizons.items()
        ]
        for name, horizon, values in sorted(
            horizon_rows,
            key=lambda item: item[2]["final_mean_normalized_rmse"],
            reverse=True,
        ):
            lines.append(
                f"| {name} ({horizon}d) | "
                f"{values['initial_mean_normalized_rmse']:.6g} | "
                f"{values['final_mean_normalized_rmse']:.6g} | "
                f"{values['relative_rmse_improvement']:.3%} | "
                f"{values['final_passed_cells']}/{values['total_cells']} |"
            )
    lines.extend(
        (
            "",
            "## Milestone Decision",
            "",
            f"- baseline macro: `{decision['baseline_macro']:.9g}`",
            f"- final macro: `{decision['final_macro']:.9g}`",
            f"- relative improvement: `{decision['relative_macro_improvement']:.6%}`",
            f"- five-percent pacing ceiling: `{decision['five_percent_pacing_macro_maximum']:.9g}`",
            f"- action: `{decision['action']}`",
            "",
        )
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=50)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.block_size <= 0:
        raise ValueError("block size must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    parent = (
        json.loads(args.parent_report.read_text(encoding="utf-8"))
        if args.parent_report is not None
        else None
    )
    result = analyze(report, parent=parent, block_size=args.block_size)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
