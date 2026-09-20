"""Physical-unit progress and a bounded diagnostic regression ceiling.

These development comparisons neither define final scientific acceptance nor
choose a best checkpoint. No network computation or gradient code lives here.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

FLUXES = ("gpp", "npp", "autotrophic_respiration")
BIOMASS = ("aboveground_biomass_carbon", "belowground_biomass_carbon")
CANOPY = "canopy_interception_organic_carbon"
NONLIVING = "nonliving_ecosystem_carbon"


def _values(monitor: Mapping[str, Any]) -> dict[str, tuple[float, str]]:
    values = {}
    for horizon in (7, 30):
        panel = monitor["horizons"][str(horizon)]
        physical = panel["physical"]
        for key in FLUXES:
            values[f"T{horizon}/{key}"] = (
                float(physical["fluxes"][key]["cumulative_gC_m2"]["rmse"]), "flux"
            )
        for key in (*BIOMASS, NONLIVING):
            values[f"T{horizon}/{key}"] = (
                float(physical["stocks"][key]["change_gC_m2"]["rmse"]),
                "nonliving" if key == NONLIVING else "biomass",
            )
        values[f"T{horizon}/{CANOPY}"] = (
            float(panel["canopy_physical"]["change_gC_m2"]["rmse"]), "canopy"
        )
    if any(not isfinite(value) or value < 0 for value, _ in values.values()):
        raise ValueError("physical progress requires finite nonnegative RMSE")
    return values


def summarize_progress(
    initial_monitor: Mapping[str, Any], current_monitor: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the same panel with the fixed selected20 parent, never old20."""
    initial = _values(initial_monitor)
    current = _values(current_monitor)
    metrics = {}
    for key, (value, group) in current.items():
        base = initial[key][0]
        ratio = value / base if base > 0 else (1.0 if value == 0 else None)
        metrics[key] = {
            "group": group,
            "baseline_rmse": base,
            "current_rmse": value,
            "ratio": ratio,
            "relative_change": None if ratio is None else ratio - 1.0,
            "absolute_increase": value - base,
        }
    return {
        "baseline": "fixed_selected20_development_panel",
        "unit": "gC_m-2",
        "metrics": metrics,
        "scientific_acceptance": False,
    }


def should_stop(
    history: Sequence[Mapping[str, Any]], stopping_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Stop only for sustained, large degradation of the same physical metric.

    Require both a relative and an absolute increase to avoid treating tiny
    near-zero errors as catastrophic. A zero baseline uses the absolute floor.
    Two distinct consecutive milestone measurements are supplied by the runner.
    """
    count = int(stopping_config["consecutive_monitors"])
    ratio_limit = float(stopping_config["rmse_ratio"])
    floors = dict(stopping_config["minimum_absolute_increase_gC_m2"])
    if count < 2 or not isfinite(ratio_limit) or ratio_limit <= 1:
        raise ValueError("invalid sustained-regression policy")
    if set(floors) != {"flux", "biomass", "canopy", "nonliving"} or any(
        not isfinite(float(value)) or float(value) <= 0 for value in floors.values()
    ):
        raise ValueError("positive physical-unit floors required")
    if len(history) < count:
        return {"stop": False, "reason": "insufficient_consecutive_monitors", "triggered_metrics": []}
    failures = []
    for summary in history[-count:]:
        failed = set()
        for key, metric in summary["metrics"].items():
            relative_bad = metric["current_rmse"] > ratio_limit * metric["baseline_rmse"]
            absolute_bad = metric["absolute_increase"] > float(floors[metric["group"]])
            if relative_bad and absolute_bad:
                failed.add(key)
        failures.append(failed)
    triggered = sorted(set.intersection(*failures))
    return {
        "stop": bool(triggered),
        "reason": "sustained_large_physical_regression" if triggered else "within_bounded_exploration",
        "triggered_metrics": triggered,
    }
