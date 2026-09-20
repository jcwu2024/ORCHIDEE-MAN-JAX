"""Read-only decomposition of saved mixed-learning science trajectories.

No Teacher/checkpoint reads, inference, optimizer updates or loss modification.
Nonliving minus canopy is litter + particulate + dissolved organic carbon;
the saved aggregate cannot identify those three contributions separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stock_change_series(record, side):
    """Recover day-zero stock from endpoint and reported window change."""
    daily = np.asarray(record[f"{side}_daily"], dtype=float)
    final = np.asarray(record[f"{side}_final"], dtype=float)
    change = np.asarray(record[f"{side}_change"], dtype=float)
    return daily - (final - change)


def window_diagnostics(window):
    stocks = window["physical"]["stocks"]
    fluxes = window["physical"]["fluxes"]
    nonliving = stocks["nonliving_ecosystem_carbon"]
    canopy = window["canopy_physical"]
    mask = np.asarray(nonliving["defined"], dtype=bool) & np.asarray(canopy["defined"], dtype=bool)
    days = int(window["selection"]["length"])
    result = {"selection": window["selection"], "days": days}
    series = {}
    for side in ("predicted", "teacher"):
        nc = stock_change_series(nonliving, side)
        cc = stock_change_series(canopy, side)
        series[f"{side}_nonliving_change"] = nc
        series[f"{side}_canopy_change"] = cc
        series[f"{side}_noncanopy_change"] = nc - cc
    for key, values in series.items():
        valid = mask & np.isfinite(values)
        result[key] = np.where(valid, values, np.nan).tolist()
    for stock in ("nonliving", "canopy", "noncanopy"):
        values = series[f"predicted_{stock}_change"] - series[f"teacher_{stock}_change"]
        result[f"{stock}_error"] = np.where(mask & np.isfinite(values), values, np.nan).tolist()
    flux_errors = {}
    flux_mask = np.ones_like(mask)
    for key in ("gpp", "npp", "autotrophic_respiration", "maintenance_respiration", "growth_respiration"):
        row = fluxes[key]
        p, t = (np.asarray(row[f"{side}_daily"], dtype=float) for side in ("predicted", "teacher"))
        valid = np.asarray(row["defined"], dtype=bool) & np.isfinite(p) & np.isfinite(t)
        flux_mask &= valid
        flux_errors[key] = p - t
        # A missing day invalidates the subsequent cumulative comparison.
        cumulative_valid = np.logical_and.accumulate(valid, axis=0)
        result[f"{key}_cumulative_error"] = np.where(cumulative_valid, np.cumsum(p - t, axis=0), np.nan).tolist()
    closure = flux_errors["npp"] - flux_errors["gpp"] + flux_errors["autotrophic_respiration"]
    result["npp_error_identity_max_abs"] = float(np.max(np.abs(closure[flux_mask]))) if np.any(flux_mask) else None
    return result


def _summary(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    a = a[finite]
    return {"count": int(a.size), "bias": float(a.mean()) if a.size else None,
            "rmse": float(np.sqrt(np.mean(a * a))) if a.size else None}


def analyze(report):
    panels = []
    daily_components = []
    for monitor in report["monitors"]:
        for panel_name in ("fixed_training_daily", "excluded_train_daily"):
            fields = monitor[panel_name]["diagnostics"]["physical"]["stocks"]["fields"]
            components = {key: fields[key]["level"] for key in (
                "litter_carbon", "soil_particulate_organic_carbon", "dissolved_organic_carbon")}
            daily_components.append({"update": monitor["update"], "panel": panel_name,
                                     "components": components,
                                     "note": "Same Teacher initial state: endpoint error equals one-day change error; endpoint values are not changes."})
        for panel_name in ("development", "fixed_training_recurrent"):
            windows = [window_diagnostics(w) for w in monitor[panel_name]["windows"]]
            for horizon in sorted({w["days"] for w in windows}):
                selected = [w for w in windows if w["days"] == horizon]
                metrics = {}
                for key in ("nonliving_error", "canopy_error", "noncanopy_error", "gpp_cumulative_error", "npp_cumulative_error", "autotrophic_respiration_cumulative_error"):
                    a = np.concatenate([np.asarray(w[key]) for w in selected], axis=1)
                    metrics[key] = {"first_day": _summary(a[0]), "last_day": _summary(a[-1]),
                                    "mean_by_day": [_summary(row)["bias"] for row in a],
                                    "rmse_by_day": [_summary(row)["rmse"] for row in a]}
                nc = metrics["noncanopy_error"]
                first, last = nc["first_day"]["bias"], nc["last_day"]["bias"]
                panels.append({"update": monitor["update"], "panel": panel_name, "horizon": horizon,
                               "metrics": metrics, "windows": selected,
                               "noncanopy_mean_error_increment_after_day1": (
                                   (last - first) / (horizon - 1)
                                   if first is not None and last is not None and horizon > 1 else None),
                               "note": "This is observed error accumulation, not a causal feedback experiment."})
    return {"schema_version": "saved_carbon_drift_v1", "completed_updates": report["completed_updates"],
            "latest_science_update": max(m["update"] for m in report["monitors"]),
            "panels": panels, "daily_components": daily_components,
            "limits": ["Reused development and fixed training panels, not independent validation.",
                       "Noncanopy aggregate cannot separate litter, particulate carbon and DOC.",
                       "Error accumulation alone does not distinguish repeated one-day bias from state feedback.",
                       "No model execution, changed loss, or checkpoint selection."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Standard JSON nulls preserve missing values without emitting NaN tokens.
    def safe(value):
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (args.output_dir / "carbon-drift.json").write_text(json.dumps(safe(result), indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Saved carbon-drift decomposition", "", "All errors in gC/m2; bias is prediction minus Teacher.", "",
             "Noncanopy = nonliving minus canopy = litter + particulate carbon + DOC.", "",
             "| Panel | Update | Days | Nonliving bias | Canopy bias | Noncanopy bias | Noncanopy day-1 bias | NPP bias |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for p in result["panels"]:
        m = p["metrics"]
        values = [m[k]["last_day"]["bias"] for k in ("nonliving_error", "canopy_error", "noncanopy_error")]
        values += [m["noncanopy_error"]["first_day"]["bias"], m["npp_cumulative_error"]["last_day"]["bias"]]
        formatted = ["undefined" if v is None else f"{v:.6g}" for v in values]
        lines.append(f"| {p['panel']} | {p['update']} | {p['horizon']} | " + " | ".join(formatted) + " |")
    lines += ["", "## Single-day component errors", "",
              "Same Teacher initial state: endpoint error equals change error. These are different panels from the recurrent windows.", "",
              "| Panel | Update | Component | Count | Bias | RMSE | Teacher endpoint mean |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for p in result["daily_components"]:
        for name, m in p["components"].items():
            lines.append(f"| {p['panel']} | {p['update']} | {name} | {m['n']} | {m['bias']:.6g} | {m['rmse']:.6g} | {m['teacher_mean']:.6g} |")
    lines += ["", *[f"- {item}" for item in result["limits"]]]
    (args.output_dir / "carbon-drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for p in result["panels"]:
        if p["panel"] != "development" or p["horizon"] != 30 or p["update"] < 1000:
            continue
        for ax, key, title in zip(axes, ("canopy_error", "noncanopy_error", "npp_cumulative_error"),
                                  ("Canopy stock error", "Litter + particulate + DOC error", "Cumulative NPP error")):
            ax.plot(np.arange(1, 31), p["metrics"][key]["mean_by_day"], label=f"Update {p['update']}")
            ax.set(title=title, xlabel="Simulation day", ylabel="Mean signed error (gC / m2)")
            ax.axhline(0, color="black", linewidth=0.7)
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Saved T30 development windows: error accumulation (not causal attribution)")
    fig.savefig(args.output_dir / "carbon-drift.png", dpi=150)
    plt.close(fig)
    print(args.output_dir / "carbon-drift.md")


if __name__ == "__main__":
    main()
