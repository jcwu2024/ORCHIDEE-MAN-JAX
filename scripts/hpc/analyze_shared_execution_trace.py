"""Summarize exported GPU trace events without counting nested host events."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json


def interval_union(intervals):
    end = None
    total = 0.0
    for left, right in sorted(intervals):
        total += max(0.0, right-max(left, end if end is not None else left))
        end = right if end is None else max(end, right)
    return total


def hlo_sources(text):
    """Keep ambiguous instruction names unassigned instead of guessing scope."""
    rows = defaultdict(set)
    for line in text.splitlines():
        instruction = re.match(r"\s*(?:ROOT )?%([^ ]+) =", line)
        source = re.search(r'source_file="([^"]+)" source_line=(\d+)', line)
        if instruction and source:
            rows[instruction[1]].add((source[1], int(source[2])))
    return {key: next(iter(values)) for key, values in rows.items() if len(values) == 1}


def kernel_category(name):
    name = name.lower()
    if "memcpy" in name or "memset" in name:
        return "memory_transfer_or_set"
    if any(token in name for token in ("gemm", "gemv", "matmul")):
        return "matrix_multiply"
    if "fusion" in name:
        return "fused_kernels"
    return "other_kernels"


def hlo_gemm_shapes(text):
    """Recover cuBLAS operand dimensions; do not infer FLOPs for fused kernels."""
    result = {}
    for line in text.splitlines():
        if 'custom_call_target="__cublas' not in line or "backend_config=" not in line:
            continue
        instruction = re.match(r"\s*(?:ROOT )?%([^ ]+) =", line)
        operands = line.split(" custom-call(", 1)
        if not instruction or len(operands) != 2:
            continue
        arrays = re.findall(r"\b(f\d+|bf16)\[([\d,]*)\]", operands[1].split(")", 1)[0])
        if len(arrays) < 2:
            continue
        try:
            backend = json.loads(line.split("backend_config=", 1)[1])
            dimensions = backend["gemm_backend_config"]["dot_dimension_numbers"]
            left, right = [tuple(int(v) for v in shape.split(",") if v) for _, shape in arrays[:2]]
            lc, rc, lb, rb = [tuple(map(int, dimensions[key])) for key in (
                "lhs_contracting_dimensions", "rhs_contracting_dimensions",
                "lhs_batch_dimensions", "rhs_batch_dimensions")]
            k = math.prod(left[i] for i in lc)
            batch = math.prod(left[i] for i in lb)
            if k != math.prod(right[i] for i in rc) or batch != math.prod(right[i] for i in rb):
                continue
            m = math.prod(size for i, size in enumerate(left) if i not in lc+lb)
            n = math.prod(size for i, size in enumerate(right) if i not in rc+rb)
        except (KeyError, ValueError, IndexError):
            continue
        result[instruction[1]] = {"operand_dtype": arrays[0][0], "batch": batch,
                                  "m": m, "n": n, "k": k, "mac_per_hlo_call": batch*m*n*k}
    return result


def summarize(trace, sources=None, gemm_shapes=None):
    events = trace["traceEvents"]
    sources = sources or {}
    gemm_shapes = gemm_shapes or {}
    gpu_pids = {e["pid"] for e in events if e.get("ph") == "M"
                and e.get("name") == "process_name"
                and "/device:GPU:" in e.get("args", {}).get("name", "")}
    gpu = [e for e in events if e.get("pid") in gpu_pids and e.get("ph") == "X"
           and "dur" in e and "ts" in e]
    if not gpu:
        return {"status": "no_gpu_events", "event_count": 0}
    totals = defaultdict(lambda: {"calls": 0, "summed_seconds": 0.0})
    names = defaultdict(lambda: {"calls": 0, "summed_seconds": 0.0})
    source_totals = defaultdict(lambda: {"calls": 0, "summed_seconds": 0.0})
    precision = defaultdict(lambda: {"calls": 0, "summed_seconds": 0.0})
    shape_totals = defaultdict(lambda: {"calls": 0, "summed_seconds": 0.0})
    for event in gpu:
        name = event["name"]
        args = event.get("args", {})
        source = sources.get(args.get("hlo_op"))
        source_key = f"{source[0]}:{source[1]}" if source else "unmapped"
        groups = [(totals, kernel_category(name)), (names, name), (source_totals, source_key)]
        shape = gemm_shapes.get(args.get("hlo_op"))
        if shape:
            shape_key = f"{shape['operand_dtype']} B={shape['batch']} M={shape['m']} N={shape['n']} K={shape['k']}"
            groups.append((shape_totals, shape_key))
        if kernel_category(name) == "matrix_multiply":
            label = ("explicit_fp64" if "dgemm" in name.lower() else
                     "explicit_fp32" if "sgemm" in name.lower() else "unknown_or_other")
            groups.append((precision, label))
        for table, key in groups:
            table[key]["calls"] += 1
            table[key]["summed_seconds"] += event["dur"] * 1e-6
    intervals = [(e["ts"], e["ts"]+e["dur"]) for e in gpu]
    span = (max(right for _, right in intervals)-min(left for left, _ in intervals))*1e-6
    active = interval_union(intervals)*1e-6
    summed = sum(e["dur"] for e in gpu)*1e-6

    def ranked(table, limit=None):
        result = [{"name": name, **value,
                   "summed_device_time_fraction": value["summed_seconds"]/summed if summed else 0.0}
                  for name, value in sorted(table.items(), key=lambda row: -row[1]["summed_seconds"])]
        return result if limit is None else result[:limit]

    annotations = [e for e in events if e.get("ph") == "X" and
                   e.get("name", "").startswith("shared_cached_gradient")]
    return {
        "status": "completed", "event_count": len(gpu), "gpu_process_ids": sorted(gpu_pids),
        "device_event_span_seconds": span, "device_active_union_seconds": active,
        "summed_device_event_seconds": summed,
        "gaps_within_device_span_seconds": max(0.0, span-active),
        "trace_annotation_seconds": [e["dur"]*1e-6 for e in annotations],
        "events_at_most_10us": sum(e["dur"] <= 10 for e in gpu),
        "events_at_most_10us_summed_seconds": sum(e["dur"] for e in gpu if e["dur"] <= 10) * 1e-6,
        "duration_buckets": [
            {"label": label,
             "events": sum(low < e["dur"] <= high for e in gpu),
             "summed_seconds": sum(e["dur"] for e in gpu if low < e["dur"] <= high) * 1e-6}
            for label, low, high in (("at_most_2us", -math.inf, 2),
                                     ("2_to_10us", 2, 10),
                                     ("10_to_100us", 10, 100),
                                     ("above_100us", 100, math.inf))
        ],
        "categories": ranked(totals), "gemm_precision_by_explicit_kernel_name": ranked(precision),
        "top_kernel_names": ranked(names, 30), "top_hlo_source_locations": ranked(source_totals, 30),
        "top_cublas_operand_shapes": ranked(shape_totals, 40),
        "limitations": [
            "Profiler instrumentation perturbs execution; warm unprofiled timing is the throughput evidence.",
            "Summed device duration can overlap; union/span/gaps are reported separately.",
            "Fused-operation source metadata is a source hint, not exclusive module attribution.",
            "Kernel names only identify explicit sgemm/dgemm precision; unknown is not FP32.",
            "One cuBLAS call may launch several kernels; shape call counts count device events, not unique HLO invocations.",
            "Trace gaps do not by themselves distinguish launch overhead, synchronization, or profiler overhead.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--hlo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    sources = {}
    gemm_shapes = {}
    if args.hlo:
        with gzip.open(args.hlo, "rt", encoding="utf-8") as handle:
            text = handle.read()
        sources, gemm_shapes = hlo_sources(text), hlo_gemm_shapes(text)
    report = summarize(trace, sources, gemm_shapes)
    report.update(trace_file=str(args.trace), hlo_file=str(args.hlo))
    _atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "event_count")}))


if __name__ == "__main__":
    main()
