"""Read recorded training timings; never load model weights or execute JAX."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

EVENT = re.compile(r"mixed_learning stage=(\S+) updates=(\d+) seconds=([\d.]+)")


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "sum": sum(values),
            "mean": statistics.mean(values), "median": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1],
            "minimum": values[0], "maximum": values[-1]}


def logged_cycles(report, text):
    """Separate update timers from adjacent wall cycles without monitor/save gaps.

    Log timestamps are captured BEFORE emit serializes the report. The residual
    includes report handling, reader statistics and Python bookkeeping; it is
    not a measurement of pure filesystem IO. Rounded log times add ~0.1s noise.
    """
    events = [(stage, int(update), float(seconds))
              for stage, update, seconds in EVENT.findall(text)]
    rows = {row["update"]: row for row in report.get("updates", [])}
    cycles = []
    for begin, done, following in zip(events, events[1:], events[2:]):
        if (begin[0] != "training" or done[0] != "optimizer_completed"
                or following[0] != "training" or done[1] != begin[1] + 1
                or following[1] != done[1] or done[1] not in rows
                or following[2] < begin[2]):
            continue
        duration = following[2] - begin[2]
        measured = rows[done[1]]["seconds"]
        cycles.append({"update": done[1], "wall_cycle_seconds": duration,
                       "update_seconds": measured,
                       "outside_update_seconds": duration - measured})
    return {"count": len(cycles), "cycles": cycles,
            "wall_cycle_seconds": distribution(r["wall_cycle_seconds"] for r in cycles),
            "outside_update_seconds": distribution(r["outside_update_seconds"] for r in cycles),
            "limitations": "Excludes cycles with checkpoint/monitor events; includes two emit calls and bookkeeping, not pure IO; rounded timestamps."}


def summarize_report(report, log=None):
    updates = report.get("updates", [])
    measured = sum(row["seconds"] for row in updates)
    elapsed = report.get("elapsed_seconds", 0)
    # Resume elapsed is process-local, while updates may include previous work.
    resumed = bool(report.get("resumed_from") or report.get("prior_elapsed_seconds"))
    monitors = sum(row.get("seconds", 0) for row in report.get("monitors", []))
    saves = sum(row.get("seconds", 0) for row in report.get("checkpoints", []))
    phase_names = sorted({row.get("objective", "mixed") for row in updates})
    phases = {name: distribution(row["seconds"] for row in updates
                                if row.get("objective", "mixed") == name)
              for name in phase_names}
    warm = [row for row in updates if row["update"] > 1 and "device_timing" in row]
    fields = ("daily_branch_seconds", "recurrent_branch_seconds", "branch_elapsed_seconds",
              "merge_checks_optimizer_seconds", "weight_synchronize_seconds")
    timing = {field: distribution(row["device_timing"][field] for row in warm)
              for field in fields}
    return {
        "status": report.get("status"), "stage": report.get("stage"),
        "source_git_head": report.get("source_git_head"),
        "completed_updates": report.get("completed_updates", 0),
        "elapsed_seconds_process_local": elapsed,
        "measured_update_seconds": measured,
        "outside_updates_seconds": None if resumed else elapsed - measured,
        "monitor_seconds": monitors,
        "retained_checkpoint_record_seconds_lower_bound": saves,
        "checkpoint_limit": "Rolling records overwrite previous save timings; this is NOT all checkpoint time.",
        "phases": phases, "warm_dual_update_seconds": distribution(row["seconds"] for row in warm),
        "warm_dual_components": timing,
        "daily_reader": report.get("daily_reader"),
        "reader_limit": "read_seconds includes gzip/unpickle and identity checks, across bootstrap/training/monitors; excludes stacking, transfers and recurrent reads; overlaps update/monitor timers.",
        "log_cycles": None if log is None else logged_cycles(report, log),
        "resumed_process_requires_separate_wall_accounting": resumed,
    }


def frequency_scenarios(profile, updates=5415):
    timing = profile["dual_device_profile"]["timing"]
    daily = timing["daily_only"]["median_seconds"]
    dual = timing["dual"]["median_seconds"]
    result = []
    for period in (1, 2, 4):
        mixed_updates = math.ceil(updates / period)
        seconds = mixed_updates * dual + (updates - mixed_updates) * daily
        result.append({"recurrent_every_n_updates": period, "daily_updates": updates,
                       "windows_per_horizon": mixed_updates,
                       "modeled_update_seconds": seconds / updates,
                       "update_hours_only": seconds / 3600,
                       "wall_reduction_fraction_vs_every_update": 1 - seconds / (updates * dual),
                       "dual_allocation_yuan_update_only": seconds / 3600 * 4.96})
    return {"basis": "Same-checkpoint fixed-input profile medians; not learning experiments or sustained throughput.",
            "daily_seconds": daily, "dual_seconds": dual,
            "all_recurrent_removed_optimistic_wall_reduction": 1 - daily / dual,
            "scenarios": result,
            "limitations": ["Daily-only profile omits peer sync: optimistic for continuous dual implementation.",
                            "Both GPUs remain allocated and billed; no startup/monitor/report/save overhead included.",
                            "Fewer windows changes training exposure/mean objective; no scientific equivalence assumed.",
                            "Holding recurrent-window count fixed instead increases daily updates and total runtime."]}


def markdown(result):
    lines = ["# Recorded training time audit", "",
             "All times come from saved reports/logs; no inference or new training.", "",
             "| Run | Updates | Python elapsed h | Timed updates h | Outside updates h | Monitors s | Daily artifact reads s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in result["reports"].items():
        outside = row["outside_updates_seconds"]
        reads = (row["daily_reader"] or {}).get("read_seconds")
        outside_text = "n/a" if outside is None else f"{outside / 3600:.3f}"
        reads_text = "n/a" if reads is None else f"{reads:.2f}"
        lines.append(f"| {name} | {row['completed_updates']} | {row['elapsed_seconds_process_local']/3600:.3f} | "
                     f"{row['measured_update_seconds']/3600:.3f} | {outside_text} | {row['monitor_seconds']:.2f} | {reads_text} |")
    lines.extend(["", "Daily artifact reads overlap other timers. Outside-update time includes initialization, compilation, monitors, saving, report serialization and other overhead; it is not all wasted time.",
                  "Rolling checkpoint records retain only the latest duration, so their sum is a lower bound.", "",
                  "## Adjacent cycles outside saves and monitors", "",
                  "Cycles below exclude saves and monitors. Their residual includes report handling and Python bookkeeping; it does not isolate disk writes.", "",
                  "| Run | Cycles | Mean wall cycle s | Mean outside-update s |",
                  "| --- | ---: | ---: | ---: |"])
    for name, row in result["reports"].items():
        cycle = row["log_cycles"]
        if cycle and cycle["count"]:
            lines.append(f"| {name} | {cycle['count']} | {cycle['wall_cycle_seconds']['mean']:.3f} | {cycle['outside_update_seconds']['mean']:.3f} |")
    lines.extend(["", "## Dual branch timings after update 1", "",
                  "All subsequent samples are retained, including slow outliers. Branches overlap and must not be added as wall time.", "",
                  "| Run | Samples | Update median / mean s | Daily mean s | Recurrent mean s | Merge mean s | Sync mean s |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, row in result["reports"].items():
        warm = row["warm_dual_update_seconds"]
        if warm["count"]:
            parts = row["warm_dual_components"]
            values = [parts[field]["mean"] for field in (
                "daily_branch_seconds", "recurrent_branch_seconds",
                "merge_checks_optimizer_seconds", "weight_synchronize_seconds")]
            lines.append(f"| {name} | {warm['count']} | {warm['median']:.3f} / {warm['mean']:.3f} | "
                         + " | ".join(f"{value:.3f}" for value in values) + " |")
    scenarios = result.get("frequency_scenarios")
    if scenarios:
        lines.extend(["", "## Lower recurrent frequency: arithmetic, not accuracy evidence", "",
                      "Same fixed-input profile: daily-only {:.3f}s, dual mixed {:.3f}s. Even removing every recurrent update gives only an optimistic {:.2%} update-wall reduction.".format(
                          scenarios["daily_seconds"], scenarios["dual_seconds"], scenarios["all_recurrent_removed_optimistic_wall_reduction"]), "",
                      "| Recurrent every N updates | Windows / horizon | Update hours only | Wall reduction | Dual cost, updates only CNY |",
                      "| ---: | ---: | ---: | ---: | ---: |"])
        for row in scenarios["scenarios"]:
            lines.append(f"| {row['recurrent_every_n_updates']} | {row['windows_per_horizon']} | {row['update_hours_only']:.3f} | {row['wall_reduction_fraction_vs_every_update']:.2%} | {row['dual_allocation_yuan_update_only']:.2f} |")
        lines.extend(["", *[f"- {text}" for text in scenarios["limitations"]]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--log", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    logs = {name: Path(path).read_text(encoding="utf-8")
            for name, path in (value.split("=", 1) for value in args.log)}
    reports, sources = {}, {}
    for value in args.report:
        name, filename = value.split("=", 1)
        path = Path(filename)
        reports[name] = summarize_report(json.loads(path.read_text(encoding="utf-8")), logs.get(name))
        sources[name] = str(path.resolve())
    result = {"schema_version": "mixed_learning_timing_audit_v1", "sources": sources, "reports": reports}
    if args.profile:
        result["frequency_scenarios"] = frequency_scenarios(json.loads(args.profile.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "timing.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "timing.md").write_text(markdown(result), encoding="utf-8")
    print(args.output_dir / "timing.md")


if __name__ == "__main__":
    main()
