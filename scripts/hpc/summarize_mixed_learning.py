"""Create read-only learning/science summaries from mixed-training reports.

This script reads report JSON only.  It does not load checkpoints, run model
inference, select a checkpoint, or modify a training report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = {
    "development": "Reused development panel",
    "fixed_training_recurrent": "Fixed training recurrent panel",
}

METRICS = {
    "gpp": {
        "label": "GPP accumulated",
        "family": "fluxes",
        "source": "gpp",
        "aggregate": "cumulative_gC_m2",
        "teacher": "teacher_total",
        "predicted": "predicted_total",
        "semantics": "accumulated_flux",
    },
    "npp": {
        "label": "NPP accumulated",
        "family": "fluxes",
        "source": "npp",
        "aggregate": "cumulative_gC_m2",
        "teacher": "teacher_total",
        "predicted": "predicted_total",
        "semantics": "accumulated_flux",
    },
    "ra": {
        "label": "RA accumulated",
        "family": "fluxes",
        "source": "autotrophic_respiration",
        "aggregate": "cumulative_gC_m2",
        "teacher": "teacher_total",
        "predicted": "predicted_total",
        "semantics": "accumulated_flux",
    },
    "rh": {
        "label": "RH accumulated",
        "family": "fluxes",
        "source": "heterotrophic_respiration",
        "aggregate": "cumulative_gC_m2",
        "teacher": "teacher_total",
        "predicted": "predicted_total",
        "semantics": "accumulated_flux",
    },
    "agb_growth": {
        "label": "AGB window change",
        "family": "stocks",
        "source": "aboveground_biomass_carbon",
        "aggregate": "change_gC_m2",
        "teacher": "teacher_change",
        "predicted": "predicted_change",
        "semantics": "stock_change_not_inventory",
    },
    "bgb_growth": {
        "label": "BGB window change",
        "family": "stocks",
        "source": "belowground_biomass_carbon",
        "aggregate": "change_gC_m2",
        "teacher": "teacher_change",
        "predicted": "predicted_change",
        "semantics": "stock_change_not_inventory",
    },
    "canopy_change": {
        "label": "Canopy carbon change",
        "family": "canopy",
        "source": "canopy_physical",
        "aggregate": "change_gC_m2",
        "teacher": "teacher_change",
        "predicted": "predicted_change",
        "semantics": "stock_change_not_inventory",
    },
    "nonliving_change": {
        "label": "Nonliving carbon change",
        "family": "stocks",
        "source": "nonliving_ecosystem_carbon",
        "aggregate": "change_gC_m2",
        "teacher": "teacher_change",
        "predicted": "predicted_change",
        "semantics": "stock_change_not_inventory",
    },
}

FLUX_METRICS = ("gpp", "npp", "ra", "rh")
CHANGE_METRICS = ("agb_growth", "bgb_growth", "canopy_change", "nonliving_change")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _population_id(windows: Iterable[Mapping[str, Any]], horizon: int) -> str:
    population = []
    for window in windows:
        selection = window.get("selection", {})
        if int(selection.get("length", -1)) != horizon:
            continue
        teacher_reference = {}
        for metric_name, spec in METRICS.items():
            if spec["family"] == "canopy":
                source = window.get("canopy_physical", {})
            else:
                source = (
                    window.get("physical", {})
                    .get(spec["family"], {})
                    .get(spec["source"], {})
                )
            teacher_reference[metric_name] = source.get(spec["teacher"])
        population.append({"selection": selection, "teacher_reference": teacher_reference})
    return hashlib.sha256(_canonical(population).encode("utf-8")).hexdigest()[:16]


def _numbers(value: Any) -> list[float]:
    if value is None:
        return []
    array = np.asarray(value, dtype=float).reshape(-1)
    return [float(item) for item in array]


def _display_value(value: Any) -> float | list[float] | None:
    values = _numbers(value)
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _metric_container(container: Mapping[str, Any], spec: Mapping[str, str]) -> Mapping[str, Any]:
    if spec["family"] == "canopy":
        return container.get("canopy_physical", {})
    return (
        container.get("physical", {})
        .get(spec["family"], {})
        .get(spec["source"], {})
    )


def _metric_summary(horizon: Mapping[str, Any], spec: Mapping[str, str]) -> dict[str, Any] | None:
    source = _metric_container(horizon, spec)
    statistics = source.get(spec["aggregate"])
    if not isinstance(statistics, Mapping):
        return None
    result = {
        key: statistics.get(key)
        for key in ("rmse", "bias", "mae", "p95_absolute_error", "count", "nonfinite_defined")
        if key in statistics
    }
    if spec["family"] == "fluxes":
        result["teacher_nonzero_count"] = source.get("teacher_nonzero_totals")
    result["source_field"] = spec["aggregate"]
    result["semantics"] = spec["semantics"]
    result["unit"] = "gC m-2"
    return result


def _window_metric(window: Mapping[str, Any], spec: Mapping[str, str]) -> dict[str, Any] | None:
    source = _metric_container(window, spec)
    if spec["teacher"] not in source or spec["predicted"] not in source:
        return None
    result = {
        "teacher": _display_value(source.get(spec["teacher"])),
        "predicted": _display_value(source.get(spec["predicted"])),
        "semantics": spec["semantics"],
        "unit": "gC m-2",
    }
    if spec["semantics"] == "stock_change_not_inventory":
        result["inventory_teacher"] = _display_value(source.get("teacher_final"))
        result["inventory_predicted"] = _display_value(source.get("predicted_final"))
        result["inventory_unit"] = "gC m-2"
    return result


def _update_time(report: Mapping[str, Any], update: int) -> dict[str, Any]:
    records = [row for row in report.get("updates", []) if isinstance(row, Mapping)]
    by_update = {int(row["update"]): row for row in records if "update" in row}
    required = set(range(1, update + 1))
    available = required.intersection(by_update)
    observed = sum(float(by_update[index].get("seconds", 0.0)) for index in available)
    complete = available == required
    return {
        "measured_update_seconds": observed if complete else None,
        "observed_update_seconds": observed,
        "timing_complete_through_monitor": complete,
        "timed_updates": len(available),
        "required_updates": update,
    }


def _daily_monitor(monitor: Mapping[str, Any]) -> dict[str, Any] | None:
    source = monitor.get("excluded_train_daily")
    if not isinstance(source, Mapping):
        return None
    fields = (
        source.get("diagnostics", {})
        .get("physical", {})
        .get("fluxes", {})
        .get("fields", {})
    )
    extracted = {}
    for name, source_name in (
        ("gpp", "gpp"),
        ("npp", "npp"),
        ("ra", "autotrophic_respiration"),
        ("rh", "heterotrophic_respiration"),
    ):
        if source_name in fields:
            extracted[name] = {
                key: fields[source_name].get(key)
                for key in ("rmse", "bias", "mae", "n", "teacher_mean", "nonfinite_defined")
            }
    return {
        "role": "excluded TRAIN daily monitor; not independent validation",
        "count": source.get("count"),
        "mean_loss": source.get("mean_loss"),
        "metrics": extracted,
    }


def summarize_reports(
    reports: Mapping[str, Mapping[str, Any]],
    source_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary without changing input reports."""
    source_paths = source_paths or {}
    arms: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for name, report in reports.items():
        monitors = report.get("monitors", [])
        if not isinstance(monitors, list):
            monitors = []
            warnings.append(f"{name}: monitors is not a list")
        if not monitors:
            warnings.append(f"{name}: no recorded monitor snapshots yet")
        monitor_updates = [int(m.get("update", 0)) for m in monitors if isinstance(m, Mapping)]
        if len(monitor_updates) != len(set(monitor_updates)):
            warnings.append(f"{name}: duplicate monitor updates are retained")
        completed_updates = int(report.get("completed_updates", 0) or 0)
        latest_monitor_update = max(monitor_updates) if monitor_updates else None
        unmonitored_completed_updates = (
            completed_updates - latest_monitor_update if latest_monitor_update is not None else completed_updates
        )
        if unmonitored_completed_updates > 0:
            warnings.append(
                f"{name}: training reached update {completed_updates}, but the latest recorded science "
                f"monitor is update {latest_monitor_update if latest_monitor_update is not None else 'none'}; "
                "no post-monitor learning outcome is inferred"
            )
        arms[name] = {
            "source_path": source_paths.get(name),
            "schema_version": report.get("schema_version"),
            "source_git_head": report.get("source_git_head"),
            "status": report.get("status"),
            "stage": report.get("stage"),
            "completed_updates": completed_updates,
            "declared_total_updates": report.get("learning_schedule", {}).get(
                "total_updates", report.get("configuration", {}).get("updates")
            ),
            "recorded_monitor_updates": monitor_updates,
            "recorded_monitor_count": len(monitors),
            "latest_recorded_monitor_update": latest_monitor_update,
            "unmonitored_completed_updates": unmonitored_completed_updates,
            "training_seconds_reported": report.get("training_seconds"),
            "elapsed_seconds_reported": report.get("elapsed_seconds"),
            "scientific_acceptance_reported": report.get("scientific_acceptance"),
            "data_consumption": report.get("data_consumption"),
            "learning_schedule": report.get("learning_schedule"),
        }
        seen_populations: dict[tuple[str, int], set[str]] = defaultdict(set)
        for monitor in monitors:
            if not isinstance(monitor, Mapping):
                warnings.append(f"{name}: ignored non-object monitor entry")
                continue
            update = int(monitor.get("update", 0))
            timing = _update_time(report, update)
            if not timing["timing_complete_through_monitor"]:
                warnings.append(f"{name}: incomplete update timing through monitor {update}")
            daily = _daily_monitor(monitor)
            if daily is not None:
                daily_records.append({"arm": name, "update": update, **timing, **daily})
            for panel_name, panel_label in PANELS.items():
                panel = monitor.get(panel_name)
                if not isinstance(panel, Mapping):
                    continue
                windows = panel.get("windows", [])
                if not isinstance(windows, list):
                    windows = []
                horizons = panel.get("horizons", {})
                if not isinstance(horizons, Mapping):
                    continue
                for horizon_key, horizon_summary in horizons.items():
                    try:
                        horizon = int(horizon_key)
                    except (TypeError, ValueError):
                        warnings.append(f"{name}: ignored non-integer horizon {horizon_key!r}")
                        continue
                    if not isinstance(horizon_summary, Mapping):
                        continue
                    population_id = _population_id(windows, horizon)
                    seen_populations[(panel_name, horizon)].add(population_id)
                    metric_values = {}
                    for metric_name, spec in METRICS.items():
                        value = _metric_summary(horizon_summary, spec)
                        if value is not None:
                            metric_values[metric_name] = value
                    window_values = []
                    for window in windows:
                        selection = window.get("selection", {})
                        if int(selection.get("length", -1)) != horizon:
                            continue
                        values = {}
                        for metric_name, spec in METRICS.items():
                            value = _window_metric(window, spec)
                            if value is not None:
                                values[metric_name] = value
                        window_values.append({"selection": selection, "metrics": values})
                    records.append(
                        {
                            "arm": name,
                            "update": update,
                            "panel": panel_name,
                            "panel_label": panel_label,
                            "panel_role_reported": panel.get("role"),
                            "horizon_days": horizon,
                            "population_id": population_id,
                            "window_count": len(window_values),
                            **timing,
                            "metrics": metric_values,
                            "windows": window_values,
                        }
                    )
        for (panel_name, horizon), populations in seen_populations.items():
            if len(populations) > 1:
                warnings.append(
                    f"{name}: {panel_name} T{horizon} selection changed across monitors; "
                    "records are split by population"
                )

    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        panel = record["panel"]
        horizon = str(record["horizon_days"])
        population_id = record["population_id"]
        group = groups[panel][horizon].setdefault(
            population_id, {"arms": set(), "arm_updates": defaultdict(list), "window_count": record["window_count"]}
        )
        group["arms"].add(record["arm"])
        group["arm_updates"][record["arm"]].append(record["update"])
    population_groups = {}
    for panel, horizons in groups.items():
        population_groups[panel] = {}
        for horizon, populations in horizons.items():
            population_groups[panel][horizon] = {}
            for population_id, group in populations.items():
                population_groups[panel][horizon][population_id] = {
                    "arms": sorted(group["arms"]),
                    "arm_updates": {
                        arm: sorted(updates) for arm, updates in sorted(group["arm_updates"].items())
                    },
                    "window_count": group["window_count"],
                    "overlay_is_population_comparable": len(group["arms"]) > 1,
                }

    rh_records = [record["metrics"].get("rh") for record in records]
    rh_records = [item for item in rh_records if item is not None]
    rh_all_zero = bool(rh_records) and all(item.get("teacher_nonzero_count") == 0 for item in rh_records)
    limitations = [
        "Development results use a reused development panel, not final independent validation.",
        "The excluded TRAIN daily monitor is not independent validation.",
        "Fixed-training recurrent results describe a fixed training panel and are reported separately.",
        "No checkpoint is selected: every recorded monitor milestone is retained.",
        "No report here establishes final scientific acceptance or 0-to-365-day extrapolation.",
        "Measured update time accumulates recorded optimizer-update seconds only; it excludes startup, monitors, saves, and is not wall time.",
        "Stock change (growth/loss) is distinct from endpoint inventory; both are retained per window.",
        "RMSE and bias are retained with per-window Teacher/prediction values because signed means can cancel.",
    ]
    if rh_all_zero:
        limitations.append(
            "RH Teacher values are zero in every summarized population; zero RH error does not validate nonzero heterotrophic respiration behavior."
        )
    return {
        "schema_version": "mixed_learning_science_summary_v1",
        "read_only_derivation": True,
        "metric_definitions": METRICS,
        "panel_definitions": {
            "development": "reused development panel; not final independent validation",
            "fixed_training_recurrent": "fixed training recurrent panel; not a development or validation population",
            "excluded_train_daily": "excluded TRAIN daily monitor; not independent validation",
        },
        "population_fingerprint_definition": (
            "SHA-256 prefix over ordered exact selection metadata and summarized Teacher references; "
            "only identical fingerprints may share an overlay group."
        ),
        "time_axis_definition": (
            "Accumulated measured update time sums recorded optimizer-update seconds; "
            "it excludes startup, monitors, saves, and is not wall time."
        ),
        "arms": arms,
        "population_groups": population_groups,
        "records": records,
        "excluded_train_daily_records": daily_records,
        "rh_teacher_all_zero": rh_all_zero,
        "limitations": limitations,
        "warnings": sorted(set(warnings)),
    }


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in text.lower()).strip("-")


def _scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def _plot_learning_group(
    summary: Mapping[str, Any],
    output_dir: Path,
    panel: str,
    horizon: int,
    population_id: str,
    arms: list[str],
    x_mode: str,
) -> str | None:
    records = [
        record
        for record in summary["records"]
        if record["panel"] == panel
        and record["horizon_days"] == horizon
        and record["population_id"] == population_id
        and record["arm"] in arms
    ]
    if x_mode == "compute":
        records = [record for record in records if record["measured_update_seconds"] is not None]
    if not records:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), layout="constrained")
    for column, metric_name in enumerate(FLUX_METRICS):
        axes[0, column].set_title(METRICS[metric_name]["label"])
    for column, metric_name in enumerate(CHANGE_METRICS):
        axes[1, column].set_title(METRICS[metric_name]["label"])
    for row, metric_names in enumerate((FLUX_METRICS, CHANGE_METRICS)):
        for column, metric_name in enumerate(metric_names):
            axis = axes[row, column]
            for arm_index, arm in enumerate(arms):
                arm_records = sorted(
                    [record for record in records if record["arm"] == arm], key=lambda item: item["update"]
                )
                points = []
                for record in arm_records:
                    metric = record["metrics"].get(metric_name, {})
                    y = _scalar(metric.get("rmse"))
                    x = (
                        record["update"]
                        if x_mode == "update"
                        else float(record["measured_update_seconds"]) / 3600.0
                    )
                    if y is not None:
                        points.append((x, y))
                if points:
                    arm_summary = summary["arms"][arm]
                    label = arm
                    if (
                        arm_summary["completed_updates"] > 0
                        and arm_summary["latest_recorded_monitor_update"] == 0
                    ):
                        label = f"{arm}: monitor 0 only (no learned snapshot)"
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        linewidth=1.7,
                        color=COLORS[arm_index % len(COLORS)],
                        label=label,
                    )
            axis.set_ylabel("RMSE (gC m$^{-2}$)")
            axis.set_xlabel(
                "Optimizer updates"
                if x_mode == "update"
                else "Accumulated measured update time (hours)"
            )
            axis.grid(alpha=0.25)
            axis.set_yscale("linear")
            if metric_name == "rh":
                rh = [record["metrics"].get("rh", {}) for record in records]
                if rh and all(item.get("teacher_nonzero_count") == 0 for item in rh):
                    axis.text(
                        0.5,
                        0.88,
                        "Teacher RH = 0\nnonzero RH untested",
                        transform=axis.transAxes,
                        ha="center",
                        va="top",
                        fontsize=8,
                    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=max(1, len(labels)))
    fig.suptitle(
        f"{PANELS[panel]}, T{horizon}: all recorded milestones\n"
        + (
            "Update count; no checkpoint selection or final acceptance claim"
            if x_mode == "update"
            else "Measured update compute excludes startup, monitors, saves, and wall-time overhead"
        ),
        fontsize=12,
    )
    suffix = f"-{population_id[:8]}" if len(summary["population_groups"][panel][str(horizon)]) > 1 else ""
    filename = f"{_slug(panel)}-t{horizon}{suffix}-learning-by-{x_mode}.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def _plot_window_group(
    summary: Mapping[str, Any],
    output_dir: Path,
    horizon: int,
    population_id: str,
    arms: list[str],
) -> str | None:
    candidates = [
        record
        for record in summary["records"]
        if record["panel"] == "development"
        and record["horizon_days"] == horizon
        and record["population_id"] == population_id
        and record["arm"] in arms
    ]
    latest = {}
    for arm in arms:
        rows = [record for record in candidates if record["arm"] == arm]
        if rows:
            candidate = max(rows, key=lambda item: item["update"])
            arm_summary = summary["arms"][arm]
            # A running arm with only its update-0 monitor has no learned
            # prediction to compare with completed pilot endpoints.
            if not (
                candidate["update"] == 0
                and arm_summary["completed_updates"] > 0
                and arm_summary["latest_recorded_monitor_update"] == 0
            ):
                latest[arm] = candidate
    if not latest:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), layout="constrained")
    for row, metric_names in enumerate((FLUX_METRICS, CHANGE_METRICS)):
        for column, metric_name in enumerate(metric_names):
            axis = axes[row, column]
            teacher_drawn = False
            for arm_index, (arm, record) in enumerate(latest.items()):
                pairs = [window["metrics"].get(metric_name, {}) for window in record["windows"]]
                teacher = [_scalar(pair.get("teacher")) for pair in pairs]
                predicted = [_scalar(pair.get("predicted")) for pair in pairs]
                valid = [index for index, values in enumerate(zip(teacher, predicted)) if None not in values]
                if not valid:
                    continue
                if not teacher_drawn:
                    axis.plot(
                        valid,
                        [teacher[index] for index in valid],
                        color="#222222",
                        marker="o",
                        linewidth=1.4,
                        label="Teacher",
                    )
                    teacher_drawn = True
                axis.plot(
                    valid,
                    [predicted[index] for index in valid],
                    color=COLORS[arm_index % len(COLORS)],
                    marker="x",
                    linewidth=1.4,
                    label=f"{arm} @ update {record['update']}",
                )
            axis.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.55)
            axis.set_title(METRICS[metric_name]["label"])
            axis.set_xlabel("Fixed window index")
            axis.set_ylabel("gC m$^{-2}$")
            axis.grid(alpha=0.25)
            axis.set_yscale("linear")
            if metric_name == "rh":
                teacher_values = [
                    _scalar(pair.get("teacher"))
                    for record in latest.values()
                    for pair in (window["metrics"].get("rh", {}) for window in record["windows"])
                ]
                teacher_values = [value for value in teacher_values if value is not None]
                if teacher_values and all(value == 0.0 for value in teacher_values):
                    axis.text(
                        0.5,
                        0.88,
                        "Teacher RH = 0\nnonzero RH untested",
                        transform=axis.transAxes,
                        ha="center",
                        va="top",
                        fontsize=8,
                    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(labels)))
    fig.suptitle(
        f"Reused development panel, T{horizon}: per-window Teacher and prediction\n"
        "Stock panels show signed window change, not inventory; linear axes; not final validation",
        fontsize=12,
    )
    suffix = (
        f"-{population_id[:8]}"
        if len(summary["population_groups"]["development"][str(horizon)]) > 1
        else ""
    )
    filename = f"development-t{horizon}{suffix}-per-window-science.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def create_plots(summary: Mapping[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for panel, horizons in summary["population_groups"].items():
        for horizon_text, populations in horizons.items():
            horizon = int(horizon_text)
            for population_id, group in populations.items():
                arms = group["arms"]
                for x_mode in ("update", "compute"):
                    filename = _plot_learning_group(
                        summary, output_dir, panel, horizon, population_id, arms, x_mode
                    )
                    if filename:
                        files.append(filename)
                if panel == "development":
                    filename = _plot_window_group(
                        summary, output_dir, horizon, population_id, arms
                    )
                    if filename:
                        files.append(filename)
    return files


def _format_number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6g}"


def _markdown(summary: Mapping[str, Any], plot_files: Iterable[str], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "This is a read-only summary of recorded report snapshots. It performs no inference, training, or checkpoint selection.",
        "",
        "## Inputs and coverage",
        "",
        "| Arm | Status | Completed / declared updates | Recorded monitors | Scientific acceptance in source |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for name, arm in summary["arms"].items():
        source_acceptance = arm.get("scientific_acceptance_reported")
        lines.append(
            f"| {name} | {arm.get('status')} | {arm.get('completed_updates')} / "
            f"{arm.get('declared_total_updates')} | {', '.join(map(str, arm['recorded_monitor_updates'])) or 'none'} | "
            f"{source_acceptance} |"
        )
    lines.extend(
        [
            "",
            "The development panel is reused development evidence, not final independent validation. "
            "The fixed-training recurrent panel is reported separately. The excluded TRAIN daily monitor is not independent validation.",
            "",
            "## Latest recorded development metrics",
            "",
            "RMSE and signed bias are in gC m-2. Fluxes are accumulated over the stated horizon; AGB/BGB, canopy, and nonliving rows are signed window changes, not endpoint inventories.",
            "",
            "| Arm | Update | Horizon | Quantity | RMSE | Bias |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    development = [record for record in summary["records"] if record["panel"] == "development"]
    for arm in summary["arms"]:
        for horizon in sorted({record["horizon_days"] for record in development}):
            candidates = [
                record
                for record in development
                if record["arm"] == arm and record["horizon_days"] == horizon
            ]
            if not candidates:
                continue
            latest = max(candidates, key=lambda item: item["update"])
            for metric_name in (*FLUX_METRICS, *CHANGE_METRICS):
                metric = latest["metrics"].get(metric_name)
                if metric is None:
                    continue
                lines.append(
                    f"| {arm} | {latest['update']} | {horizon} | {METRICS[metric_name]['label']} | "
                    f"{_format_number(metric.get('rmse'))} | {_format_number(metric.get('bias'))} |"
                )
    lines.extend(["", "## Population comparability", ""])
    for panel, horizons in summary["population_groups"].items():
        for horizon, populations in horizons.items():
            for population_id, group in populations.items():
                lines.append(
                    f"- {PANELS[panel]}, T{horizon}, population `{population_id}`: "
                    f"{group['window_count']} windows; arms {', '.join(group['arms'])}."
                )
    lines.extend(["", "Different population fingerprints are never overlaid in one plotted series group.", ""])
    if plot_files:
        lines.extend(["## Figures", ""])
        lines.extend(f"- [{filename}]({filename})" for filename in plot_files)
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    if summary["warnings"]:
        lines.extend(["", "## Snapshot warnings", ""])
        lines.extend(f"- {item}" for item in summary["warnings"])
    lines.append("")
    return "\n".join(lines)


def write_bundle(summary: dict[str, Any], output_dir: Path, title: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = create_plots(summary, output_dir)
    summary["generated_plots"] = plots
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "README.md"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(summary, plots, title), encoding="utf-8")
    return [json_path, markdown_path, *(output_dir / name for name in plots)]


def _parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("report must contain a nonempty NAME and PATH")
    return name.strip(), Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        type=_parse_report,
        metavar="NAME=PATH",
        help="Named mixed-learning report; repeat for multiple arms",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Mixed-learning science review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    named_paths: dict[str, Path] = {}
    for name, path in args.report:
        if name in named_paths:
            raise ValueError(f"duplicate report name: {name}")
        named_paths[name] = path
    reports = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in named_paths.items()
    }
    summary = summarize_reports(reports, {name: str(path) for name, path in named_paths.items()})
    outputs = write_bundle(summary, args.output_dir, args.title)
    print(json.dumps({"outputs": [str(path) for path in outputs], "warnings": summary["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
