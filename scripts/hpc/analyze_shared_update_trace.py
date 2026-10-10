"""Summarize one profiled full-update trace without guessing kernel ownership."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any

from scripts.hpc.analyze_shared_execution_trace import interval_union, kernel_category
from scripts.hpc.training_artifact_io import _atomic_json


def _union_in_window(events: list[dict[str, Any]], start: float, end: float):
    intervals = []
    for event in events:
        left, right = event["ts"], event["ts"] + event["dur"]
        left, right = max(left, start), min(right, end)
        if right > left:
            intervals.append((left, right))
    return intervals, interval_union(intervals)


def _merged(intervals):
    result = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


def _intersection_union(left_intervals, right_intervals):
    left_rows, right_rows = _merged(left_intervals), _merged(right_intervals)
    left_index = right_index = 0
    total = 0.0
    while left_index < len(left_rows) and right_index < len(right_rows):
        left, right = left_rows[left_index]
        gpu_left, gpu_right = right_rows[right_index]
        total += max(0.0, min(right, gpu_right) - max(left, gpu_left))
        if right <= gpu_right:
            left_index += 1
        else:
            right_index += 1
    return total


def summarize(trace: dict[str, Any], compile_counters: dict[str, Any] | None = None,
              run_log: str | None = None,
              update_report: dict[str, Any] | None = None) -> dict[str, Any]:
    events = trace.get("traceEvents", [])
    annotations = [event for event in events if event.get("ph") == "X"
                   and isinstance(event.get("name"), str)
                   and "ts" in event and "dur" in event]
    updates = [event for event in annotations if event["name"] == "full_update"]
    if len(updates) != 1:
        return {"status": "invalid_update_annotation_count", "full_update_annotation_count": len(updates)}
    update = updates[0]
    start, end = update["ts"], update["ts"] + update["dur"]
    gpu_pids = {event["pid"] for event in events if event.get("ph") == "M"
                and event.get("name") == "process_name"
                and "/device:GPU:" in event.get("args", {}).get("name", "")}
    gpu = [event for event in events if event.get("pid") in gpu_pids and event.get("ph") == "X"
           and "ts" in event and "dur" in event]
    clipped, active = _union_in_window(gpu, start, end)
    active_gpu_events = [event for event in gpu
                         if event["ts"] < end and event["ts"] + event["dur"] > start]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in annotations:
        name = event["name"]
        if name.startswith("call_"):
            by_name.setdefault(name, []).append(event)
    call_rows = []
    call_intervals = []
    call_duration_sum = 0.0
    for name, rows in sorted(by_name.items()):
        intervals = [(max(start, row["ts"]), min(end, row["ts"] + row["dur"]))
                     for row in rows if row["ts"] < end and row["ts"] + row["dur"] > start]
        call_union = interval_union(intervals)
        call_intervals.extend(intervals)
        call_duration_sum += call_union
        call_rows.append({
            "name": name,
            "annotation_count": len(intervals),
            "host_call_union_seconds": call_union * 1e-6,
            "gpu_active_overlap_union_seconds": _intersection_union(intervals, clipped) * 1e-6,
            "gpu_overlap_is_exclusive_attribution": False,
            "call_interval_is_full_blocking_branch_wall": False,
        })
    wall = (end - start) * 1e-6
    call_union = interval_union(call_intervals) * 1e-6
    active_seconds = active * 1e-6
    module_events = active_gpu_events
    mapped = sum("hlo_op" in event.get("args", {}) for event in module_events)
    gpu_first_last_span = (
        max(min(end, event["ts"] + event["dur"]) for event in active_gpu_events)
        - min(max(start, event["ts"]) for event in active_gpu_events)
        if active_gpu_events else None
    )
    categories: dict[str, dict[str, float | int]] = {}
    for event in active_gpu_events:
        category = kernel_category(event["name"])
        left, right = max(start, event["ts"]), min(end, event["ts"] + event["dur"])
        row = categories.setdefault(category, {"event_count": 0, "summed_clipped_seconds": 0.0})
        row["event_count"] += 1
        row["summed_clipped_seconds"] += (right-left) * 1e-6
    warnings = []
    if run_log:
        warnings = [line.strip() for line in run_log.splitlines()
                    if re.search(r"CUPTI|dropped|buffer unavailable|events dropped", line, re.I)]
    counters = compile_counters or {}
    required_counters = {"compile_or_get_cached_calls", "backend_compile_calls"}
    compile_free = (
        required_counters.issubset(counters)
        and all(counters[name] == 0 for name in required_counters)
    )
    coverage_status = (
        "missing_gpu_process_metadata" if not gpu_pids
        else "no_gpu_intervals_clipped_to_update" if not clipped
        else "gpu_intervals_observed"
    )
    gpu_coverage_passed = bool(clipped)
    quality_status = (
        "no_gpu_events" if not gpu_coverage_passed
        else "partial_profiler_warning" if warnings
        else "completed"
    )
    return {
        "status": quality_status,
        "gpu_trace_coverage_status": coverage_status,
        "gpu_trace_coverage_passed": gpu_coverage_passed,
        "update_wall_seconds_in_trace": wall,
        "gpu_event_count_clipped_to_update": len(clipped),
        "gpu_active_union_seconds": active_seconds,
        "gpu_events_summed_seconds": sum(right-left for left, right in clipped) * 1e-6,
        "gpu_first_last_event_span_fraction_of_root": (
            gpu_first_last_span/(end-start) if gpu_first_last_span is not None and end > start else None
        ),
        "gpu_kernel_categories_by_name_only": categories,
        "host_gap_seconds_relative_to_gpu_active_union": (
            max(0.0, wall-active_seconds) if gpu_coverage_passed else None
        ),
        "call_annotations": call_rows,
        "call_host_union_seconds": call_union,
        "call_annotation_overlap_seconds": max(0.0, call_duration_sum * 1e-6 - call_union),
        "unannotated_host_seconds": max(0.0, wall-call_union),
        "amdahl_style_optimistic_ceilings_from_untraced_runner": (
            (update_report or {}).get("amdahl_style_optimistic_ceilings")
        ),
        "amdahl_limitation": "Amdahl scenarios use disjoint per-execution branch and AdamW wall times from the untraced runner report; this trace does not define branch wall fractions.",
        "conditional_gpu_active_union_fraction": active_seconds/wall if gpu_coverage_passed and wall else None,
        "conditional_wall_over_gpu_active_union": wall/active_seconds if gpu_coverage_passed and active_seconds else None,
        "gpu_union_limitation": "conditional illustration only; not a hardware bound or an Amdahl speedup ceiling",
        "gpu_event_hlo_op_metadata_coverage": {
            "events_with_hlo_op": mapped,
            "events_total": len(module_events),
            "fraction": mapped/len(module_events) if module_events else None,
        },
        "forward_backward_or_neural_physics_split": "unknown; trace metadata does not establish these ownership boundaries",
        "compile_counters": counters,
        "compile_free_confirmed": compile_free,
        "profiler_warnings_from_run_log": warnings,
        "profiler_warning_evidence_complete": run_log is not None,
        "profiler_warning_makes_trace_partial": bool(warnings),
        "limitations": [
            "GPU intervals are clipped to the annotated full-update interval; overlapped kernel durations are reported separately from their union.",
            "Call overlap with GPU activity is temporal overlap only; CPU calls can be asynchronous and it does not establish exclusive process ownership.",
            "Kernel and XlaModule labels do not separate neural from physical or forward from backward work.",
            "Host gaps can include dispatch, synchronization, host computation, contention, and profiler effects.",
            "Trace warnings are only checked in the supplied run log; absence from this report is not proof that CUPTI dropped no events.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--compile-counters", type=Path)
    parser.add_argument("--update-report", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    update_report = json.loads(args.update_report.read_text(encoding="utf-8")) if args.update_report else None
    counters = json.loads(args.compile_counters.read_text(encoding="utf-8")) if args.compile_counters else (
        update_report.get("profiled_update", {}).get("compile_counters") if update_report else None
    )
    log = args.run_log.read_text(encoding="utf-8", errors="replace") if args.run_log else None
    report = summarize(trace, counters, log, update_report)
    report.update(trace_file=str(args.trace), run_log=str(args.run_log) if args.run_log else None)
    _atomic_json(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "gpu_event_count_clipped_to_update", "compile_free_confirmed")}))


if __name__ == "__main__":
    main()
