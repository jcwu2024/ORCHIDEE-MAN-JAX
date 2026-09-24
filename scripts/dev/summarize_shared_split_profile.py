"""Summarize saved split-model HLO and GPU traces without loading JAX."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.hpc.analyze_shared_execution_trace import hlo_gemm_shapes, summarize

_INSTRUCTION = re.compile(r"^\s*(?:ROOT )?%([^\s=]+)\s*=\s*.*?\b([\w-]+)\(")
_COMPUTATION = re.compile(r"^\s*(?:ENTRY\s+)?%?[^\s{}]+.*\{\s*$")
_SOURCE = re.compile(r'source_file="([^"]+)" source_line=(\d+)')


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


def _hlo_instruction_sources(text: str) -> dict[str, tuple[str, int]]:
    candidates = defaultdict(set)
    for line in text.splitlines():
        instruction = _INSTRUCTION.match(line)
        if instruction is None:
            continue
        origin = _SOURCE.search(line)
        source = (origin.group(1), int(origin.group(2))) if origin else None
        candidates[instruction.group(1)].add(source)
    return {name: next(iter(values)) for name, values in candidates.items() if len(values) == 1 and None not in values}


def summarize_hlo(text: str) -> dict:
    operations: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    files: Counter[str] = Counter()
    computations = 0
    candidate_sources = defaultdict(set)
    for line in text.splitlines():
        if _COMPUTATION.match(line):
            computations += 1
        instruction = _INSTRUCTION.match(line)
        if instruction is None:
            continue
        operations[instruction.group(2)] += 1
        origin = _SOURCE.search(line)
        source = (origin.group(1), int(origin.group(2))) if origin else None
        categories[source_category(source)] += 1
        candidate_sources[instruction.group(1)].add(source)
        if source:
            files[source[0].replace("\\", "/").rsplit("/", 1)[-1]] += 1
    return {
        "text_bytes": len(text.encode("utf-8")),
        "computation_count": computations,
        "instruction_count": sum(operations.values()),
        "operation_counts": dict(operations.most_common()),
        "source_category_instruction_counts": dict(categories),
        "source_file_instruction_counts": dict(files.most_common()),
        "unique_instruction_names": len(candidate_sources),
        "ambiguous_instruction_names": sum(len(value) > 1 for value in candidate_sources.values()),
    }


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


def summarize_assets(assets: Path) -> dict:
    report = {
        "status": "completed",
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
            continue
        with gzip.open(hlo_path, "rt", encoding="utf-8") as handle:
            hlo = handle.read()
        component = {"status": "completed", "hlo_file": str(hlo_path), "hlo": summarize_hlo(hlo)}
        trace_name = "daily_micro4" if name == "daily" else name
        traces = sorted((assets / f"trace-{trace_name}").glob("plugins/profile/**/*.trace.json.gz"))
        if traces:
            trace_path = traces[-1]
            with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
            sources = _hlo_instruction_sources(hlo)
            component.update(
                trace_file=str(trace_path),
                trace=summarize(trace, sources, hlo_gemm_shapes(hlo)),
                source_category_gpu_events=summarize_trace_sources(trace, sources),
            )
        else:
            component["trace_status"] = "not_recorded"
        report["components"][name] = component
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize_assets(args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: row["status"] for name, row in report["components"].items()}))


if __name__ == "__main__":
    main()
