"""Fixed protocol and provenance checks for the four-arm litter experiment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "litter_parallel_training_v1"
RESULT_SCHEMA_VERSION = "litter_parallel_training_result_v1"
ARMS = ("A", "B", "C", "D")
SELECTED_WEIGHT = 5.221716892123007e-5
QUARTER_WEIGHT = 1.3054292230307518e-5

SCIENCE_STOP = {
    "consecutive_monitors": 2,
    "rmse_ratio": 2.0,
    "minimum_absolute_increase_gC_m2": {
        "flux": 1.0,
        "biomass": 1.0,
        "canopy": 5.0,
        "nonliving": 10.0,
    },
}

EXPECTED_ARMS = {
    "A": {
        "litter_weight": SELECTED_WEIGHT,
        "learning_rate": 1e-5,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
    },
    "B": {
        "litter_weight": QUARTER_WEIGHT,
        "learning_rate": 1e-5,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
    },
    "C": {
        "litter_weight": SELECTED_WEIGHT,
        "learning_rate": 1e-5,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 1.0},
    },
    "D": {
        "litter_weight": SELECTED_WEIGHT,
        "learning_rate": 3e-6,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
    },
}


def validate_protocol(protocol: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("litter parallel-training protocol schema drift")
    if arm not in ARMS:
        raise ValueError("parallel arm must be one of A, B, C, D")
    expected = {
        "source_relative_update": 100,
        "source_absolute_update": 4100,
        "updates_per_arm": 500,
        "target_relative_update": 600,
        "milestones": [200, 350, 600],
        "recovery_interval": 25,
        "report_flush_interval": 10,
        "daily_batch": 64,
        "microbatch": 8,
        "scientific_stop": SCIENCE_STOP,
        "scientific_acceptance": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"litter parallel-training {key} drift")
    arms = protocol.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("litter parallel-training arm registry drift")
    for name in ARMS:
        supplied = arms[name]
        if not isinstance(supplied, Mapping) or set(supplied) != {
            "litter_weight", "learning_rate", "outer_weights"
        }:
            raise ValueError(f"litter parallel-training arm {name} keys drift")
        expected_arm = EXPECTED_ARMS[name]
        if (
            float(supplied["litter_weight"]) != expected_arm["litter_weight"]
            or float(supplied["learning_rate"]) != expected_arm["learning_rate"]
            or supplied["outer_weights"] != expected_arm["outer_weights"]
        ):
            raise ValueError(f"litter parallel-training arm {name} configuration drift")
    selected = EXPECTED_ARMS[arm]
    return {
        **expected,
        "milestones": tuple(expected["milestones"]),
        "update_indices": tuple(range(4100, 4600)),
        "selected_weight": selected["litter_weight"],
        "learning_rate": selected["learning_rate"],
        "outer_weights": dict(selected["outer_weights"]),
        "arm": arm,
    }


def source_continuation_method(
    report: Mapping[str, Any], *, checkpoint: Path, plan_identities: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the exact selected100 method, monitor, and daily panels."""

    if (
        report.get("schema_version") != "litter_mixed_continuation_result_v1"
        or report.get("status") != "completed"
        or report.get("technical_continuation_passed") is not True
        or report.get("bounded_continuation_completed") is not True
        or report.get("physical_monitoring_passed") is not True
        or report.get("scientific_acceptance") is not False
        or int(report.get("completed_relative_update", -1)) != 100
        or int(report.get("completed_absolute_update", -1)) != 4100
    ):
        raise ValueError("source continuation report is not a completed selected100 run")
    method = report.get("checkpoint_method")
    if not isinstance(method, Mapping) or method.get("implementation") != "litter_mixed_selected_continuation_v1":
        raise ValueError("source continuation checkpoint method is unavailable")
    rows = [
        row for row in report.get("milestone_checkpoints", ())
        if int(row.get("relative_update", -1)) == 100
    ]
    if (
        len(rows) != 1
        or int(rows[0].get("absolute_update", -1)) != 4100
        or Path(str(rows[0].get("path", ""))).resolve() != Path(checkpoint).resolve()
    ):
        raise ValueError("--checkpoint is not the completed selected100 checkpoint")
    monitors = [
        row for row in report.get("milestone_monitors", ())
        if int(row.get("relative_update", -1)) == 100
    ]
    if len(monitors) != 1 or int(monitors[0].get("absolute_update", -1)) != 4100:
        raise ValueError("source continuation selected100 monitor is unavailable")
    monitor = monitors[0]
    if (
        monitor.get("physical_gate_passed") is not True
        or monitor.get("metric_population_complete") is not True
        or set(monitor.get("daily_panels", {}))
        != {"fixed_training_daily", "excluded_train_daily"}
    ):
        raise ValueError("source continuation selected100 physical evidence is incomplete")
    if plan_identities is not None and report.get("plan_identities") != plan_identities:
        raise ValueError("source continuation and current plan identities differ")
    return dict(method), dict(monitor["development_monitor"]), dict(monitor["daily_panels"])


def effective_optimizer_protocol(
    model_protocol: Mapping[str, Any], *, arm: str, learning_rate: float,
) -> dict[str, Any]:
    if arm not in ARMS or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid parallel arm optimizer configuration")
    result = copy.deepcopy(dict(model_protocol))
    optimizer = result.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "AdamW":
        raise ValueError("parallel training requires the source AdamW optimizer")
    optimizer["learning_rate"] = float(learning_rate)
    return result


def checkpoint_method(
    *, source_checkpoint: Path, source_method: Mapping[str, Any], protocol: Mapping[str, Any],
    arm: str, effective_optimizer: Mapping[str, Any], runner_git_head: str,
) -> dict[str, Any]:
    return {
        "implementation": "litter_parallel_training_v1",
        "parent_checkpoint": str(Path(source_checkpoint).resolve()),
        "parent_method": dict(source_method),
        "protocol": dict(protocol),
        "arm": arm,
        "effective_optimizer": dict(effective_optimizer),
        "runner_git_head": str(runner_git_head),
        "claim": "bounded_parallel_experiment_not_final_model",
    }


def artifact_path(
    output: Path, *, arm: str, relative_update: int | None = None, recovery: bool = False,
) -> Path:
    if arm not in ARMS or recovery == (relative_update is not None):
        raise ValueError("choose exactly one valid parallel checkpoint identity")
    suffix = (
        f"arm-{arm}-recovery.checkpoint.pkl"
        if recovery
        else f"arm-{arm}-update-{int(relative_update):06d}.checkpoint.pkl"
    )
    return Path(output).with_name(f"{Path(output).stem}.{suffix}")
