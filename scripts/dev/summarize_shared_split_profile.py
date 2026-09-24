"""Summarize saved split-model HLO and GPU traces without loading JAX."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.hpc.analyze_shared_execution_trace import hlo_gemm_shapes, summarize

_INSTRUCTION = re.compile(rb"^\s*(?:ROOT )?%([^\s=]+)\s*=\s*.*?\b([\w-]+)\(")
_SOURCE = re.compile(rb'source_file="([^"]+)" source_line=(\d+)')
_PREFIX_LIMIT = 4096
_LONGEST_LINES = 5


def source_category(source: tuple[str, int] | None) -> str:
    """Use only source locations that identify one owner unambiguously."""
    if source is None:
        return "ambiguous_or_unmapped"
    path, line = source
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name == "shared_transformer.py":
        return "shared_transformer"
    if name == "strong_baseline_architecture.py":
        # _linear/_ffn and norms serve both tokenizer and shared decoder.
        return "ambiguous_or_unmapped" if line >= 730 else "tokenizer"
    if name in {"canonical_retained_tail.py", "conservative_update.py"} or "/jax_orchidee/" in normalized:
        return "retained_physics"
    if "/research/daily_coarse_graining/daily_operator/" in normalized:
        return "other_daily_operator"
    return "other_or_external"


def summarize_hlo_lines(lines, selected_op_names: set[str] | None = None) -> tuple[dict, dict, dict]:
    """Stream HLO once; retain detailed mappings only for traced operation IDs."""
    operations: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    files: Counter[str] = Counter()
    computations = 0
    line_count = text_bytes = instruction_bytes = constant_bytes = 0
    unparsed_assignment_count = unparsed_assignment_bytes = 0
    metadata_suffix_bytes = op_name_value_bytes = 0
    longest_lines: list[tuple[int, str]] = []
    candidate_sources: dict[str, set[tuple[str, int] | None]] = defaultdict(set)
    selected_counts: Counter[str] = Counter()
    shapes = {}
    for raw_line in lines:
        line = raw_line.encode("utf-8") if isinstance(raw_line, str) else raw_line
        line_count += 1
        size = len(line)
        text_bytes += size
        if len(longest_lines) < _LONGEST_LINES or size > longest_lines[-1][0]:
            preview = line[:120].decode("utf-8", errors="replace").strip()
            longest_lines.append((size, preview))
            longest_lines.sort(key=lambda item: item[0], reverse=True)
            del longest_lines[_LONGEST_LINES:]
        prefix = line[:_PREFIX_LIMIT]
        if prefix.lstrip().startswith((b"ENTRY %", b"%")) and line.rstrip().endswith(b"{"):
            computations += 1
        instruction = _INSTRUCTION.match(prefix)
        if instruction is None and b" = " in prefix[:256]:
            # Tuple shapes can exceed the preview limit; retry only likely
            # instruction lines against the full line.
            instruction = _INSTRUCTION.match(line)
        if instruction is None:
            if b" = " in prefix[:256] and prefix.lstrip().startswith((b"%", b"ROOT %")):
                unparsed_assignment_count += 1
                unparsed_assignment_bytes += size
            continue
        name = instruction.group(1).decode("ascii", errors="replace")
        operation = instruction.group(2).decode("ascii", errors="replace")
        operations[operation] += 1
        instruction_bytes += size
        if operation == "constant":
            constant_bytes += size
        metadata_start = line.find(b"metadata=")
        if metadata_start >= 0:
            metadata_suffix_bytes += size - metadata_start
        op_name_start = line.find(b'op_name="')
        if op_name_start >= 0:
            value_start = op_name_start + len(b'op_name="')
            value_end = line.find(b'"', value_start)
            if value_end >= 0:
                op_name_value_bytes += value_end - value_start
        origin = _SOURCE.search(line)
        source = (origin.group(1).decode("utf-8", errors="replace"), int(origin.group(2))) if origin else None
        categories[source_category(source)] += 1
        if source:
            files[source[0].replace("\\", "/").rsplit("/", 1)[-1]] += 1
        if selected_op_names is None or name in selected_op_names:
            selected_counts[name] += 1
            candidate_sources[name].add(source)
            if b'custom_call_target="__cublas' in line and b"backend_config=" in line:
                shapes.update(hlo_gemm_shapes(line.decode("utf-8", errors="replace")))
    sources = {
        name: next(iter(values))
        for name, values in candidate_sources.items()
        if len(values) == 1 and None not in values
    }
    summary = {
        "text_bytes": text_bytes,
        "line_count": line_count,
        "longest_lines": [{"bytes": size, "prefix": prefix} for size, prefix in longest_lines],
        "computation_count": computations,
        "instruction_count": sum(operations.values()),
        "instruction_line_bytes": instruction_bytes,
        "noninstruction_line_bytes": text_bytes - instruction_bytes,
        "unparsed_assignment_lines": unparsed_assignment_count,
        "unparsed_assignment_line_bytes": unparsed_assignment_bytes,
        "constant_instruction_line_bytes": constant_bytes,
        "metadata_suffix_bytes_upper_bound": metadata_suffix_bytes,
        "op_name_value_bytes": op_name_value_bytes,
        "operation_counts": dict(operations.most_common()),
        "source_category_instruction_counts": dict(categories),
        "source_file_instruction_counts": dict(files.most_common()),
        "selected_hlo_op_names": len(candidate_sources),
        "selected_duplicate_name_count": sum(count > 1 for count in selected_counts.values()),
        "selected_conflicting_source_count": sum(len(value) > 1 for value in candidate_sources.values()),
    }
    return summary, sources, shapes


def _hlo_instruction_sources(text: str) -> dict[str, tuple[str, int]]:
    return summarize_hlo_lines(text.splitlines(keepends=True))[1]


def summarize_hlo(text: str) -> dict:
    return summarize_hlo_lines(text.splitlines(keepends=True))[0]


def summarize_trace_sources(trace: dict, sources: dict) -> dict:
    gpu_pids = {
        event["pid"]
        for event in trace["traceEvents"]
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and "/device:GPU:" in event.get("args", {}).get("name", "")
    }
    totals = defaultdict(lambda: {"events": 0, "summed_seconds": 0.0})
    for event in trace["traceEvents"]:
        if event.get("pid") not in gpu_pids or event.get("ph") != "X" or "dur" not in event:
            continue
        category = source_category(sources.get(event.get("args", {}).get("hlo_op")))
        totals[category]["events"] += 1
        totals[category]["summed_seconds"] += event["dur"] * 1e-6
    return dict(totals)


def summarize_assets(assets: Path, on_progress=None) -> dict:
    report = {
        "status": "running",
        "assets": str(assets),
        "components": {},
        "limitations": [
            "Optimized HLO instruction counts include nested fusion computations and are not compile-time attribution.",
            "Source locations on shared helper functions cannot identify a unique neural stage.",
            "Profiled GPU durations are instrumented; unprofiled warm replays are throughput evidence.",
        ],
    }
    for name in ("daily", "T7", "T30"):
        hlo_path = assets / f"{name}.optimized-hlo.txt.gz"
        if not hlo_path.exists():
            report["components"][name] = {"status": "missing_hlo"}
            if on_progress is not None:
                on_progress(report, name)
            continue
        trace_name = "daily_micro4" if name == "daily" else name
        traces = sorted((assets / f"trace-{trace_name}").glob("plugins/profile/**/*.trace.json.gz"))
        trace = None
        selected_op_names = set()
        if traces:
            trace_path = traces[-1]
            with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
            selected_op_names = {
                event.get("args", {}).get("hlo_op")
                for event in trace["traceEvents"]
                if isinstance(event.get("args", {}).get("hlo_op"), str)
            }
        with gzip.open(hlo_path, "rb") as handle:
            hlo_summary, sources, shapes = summarize_hlo_lines(handle, selected_op_names)
        component = {"status": "completed", "hlo_file": str(hlo_path), "hlo": hlo_summary}
        if trace is not None:
            component.update(
                trace_file=str(trace_path),
                trace=summarize(trace, sources, shapes),
                source_category_gpu_events=summarize_trace_sources(trace, sources),
            )
        else:
            component["trace_status"] = "not_recorded"
        report["components"][name] = component
        if on_progress is not None:
            on_progress(report, name)
    report["status"] = "completed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def publish(report, component):
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"component": component, "status": report["components"][component]["status"]}), flush=True)

    report = summarize_assets(args.assets, on_progress=publish)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: row["status"] for name, row in report["components"].items()}))


if __name__ == "__main__":
    main()
