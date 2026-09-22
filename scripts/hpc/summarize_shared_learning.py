"""Summarize fixed physical panels from an existing shared learning report."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(report):
    rows = []
    for monitor in report["monitors"]:
        panels = dict(monitor["panels"])
        panels["development_pooled"] = {
            "windows": [window for name, panel in monitor["panels"].items()
                        if name.startswith("development_") for window in panel["windows"]]
        }
        for name, panel in panels.items():
            for horizon in (7, 30):
                windows = [w for w in panel["windows"] if w["horizon"] == horizon]
                row = dict(update=monitor["update"], panel=name, horizon=horizon,
                           windows=len(windows), physical_pass=all(
                               w["physical_gate_passed"] for w in windows), metrics={})
                for kind in ("fluxes", "stocks"):
                    for key in windows[0]["physical_curves"][kind]:
                        predicted, teacher = [], []
                        for window in windows:
                            curve = window["physical_curves"][kind][key]
                            mask = np.asarray(curve["defined"], bool)
                            if kind == "fluxes":
                                p, t = (np.asarray(curve[k]) for k in ("predicted", "teacher"))
                                valid = np.all(mask, axis=0)
                                p, t = p.sum(axis=0), t.sum(axis=0)
                            else:
                                p, t = (np.asarray(curve[k])[-1] for k in (
                                    "predicted_change_from_common_initial",
                                    "teacher_change_from_common_initial"))
                                valid = mask[-1]
                            predicted.extend(p[valid].reshape(-1).tolist())
                            teacher.extend(t[valid].reshape(-1).tolist())
                        p, t = np.asarray(predicted), np.asarray(teacher)
                        delta = p-t
                        row["metrics"][key] = dict(
                            count=len(t), rmse=float(np.sqrt(np.mean(delta**2))),
                            bias=float(np.mean(delta)), mae=float(np.mean(abs(delta))),
                            teacher_rms=float(np.sqrt(np.mean(t**2))),
                            teacher_mean=float(np.mean(t)),
                            teacher_nonzero=int(np.sum(abs(t) > 1e-12)))
                rows.append(row)
    updates = report["updates"]
    return dict(status=report["status"], completed_updates=report["completed_updates"],
                total_minutes=report["elapsed_seconds"]/60,
                peak_gib=report["device_memory"]["peak_bytes_in_use"]/2**30,
                clip_count=sum(u["clip_scale"] < 1 for u in updates),
                median_clip_scale=float(np.median([u["clip_scale"] for u in updates])),
                warm_median_wall_seconds=float(np.median([u["wall_seconds"] for u in updates[1:]])),
                warm_median_compute_seconds=float(np.median([
                    u["timing_seconds"]["total"] for u in updates[1:]])),
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    keys = ("gpp", "npp", "autotrophic_respiration", "aboveground_biomass_carbon",
            "belowground_biomass_carbon", "canopy_interception_organic_carbon")
    print("panel T update: " + " / ".join(keys) + " (RMSE gC/m2)")
    for row in result["rows"]:
        print(row["panel"], row["horizon"], row["update"],
              " / ".join(f'{row["metrics"][k]["rmse"]:.3f}' for k in keys))


if __name__ == "__main__":
    main()
