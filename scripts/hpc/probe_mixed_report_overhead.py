"""Diagnostic CPU-only replay of report formatting, with no model execution."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from scripts.hpc.run_successor_process_head_canonical_gate import _json_value
    from scripts.hpc.trained_screen_common import json_safe

    report = json.loads(args.report.read_text(encoding="utf-8"))
    samples = []
    # Match the actual runner writer's second conversion and json.dumps.
    # No write_text/os.replace filesystem latency is measured here.
    for _ in range(args.repeats):
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        safe = json_safe(report)
        safe_wall = time.perf_counter() - start_wall
        safe_cpu = time.process_time() - start_cpu
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        encoded = json.dumps(_json_value(safe), indent=2, sort_keys=True) + "\n"
        dump_wall = time.perf_counter() - start_wall
        dump_cpu = time.process_time() - start_cpu
        samples.append({"json_safe_wall_seconds": safe_wall,
                        "json_safe_cpu_seconds": safe_cpu,
                        "json_dump_wall_seconds": dump_wall,
                        "json_dump_cpu_seconds": dump_cpu,
                        "total_wall_seconds": safe_wall + dump_wall,
                        "total_cpu_seconds": safe_cpu + dump_cpu,
                        "output_characters": len(encoded)})
        del safe, encoded
    output = {"schema_version": "mixed_report_cpu_overhead_v2",
              "input": str(args.report.resolve()), "input_bytes": args.report.stat().st_size,
              "source_root": str(args.source_root.resolve()),
              "completed_updates": report.get("completed_updates"),
              "monitor_count": len(report.get("monitors", [])),
              "host": platform.node(), "pid": os.getpid(),
              "python": sys.version, "samples": samples,
              "scope": "diagnostic formatting replay on saved JSON, no model/Teacher/gradient",
              "formatter": "json_safe + _json_value + json.dumps (actual runner writer)",
              "limitations": ["No disk write/replace or GPU synchronization measured; actual writer has no fsync.",
                              "Parsed JSON has plain types; live NumPy/JAX leaves may add conversion costs.",
                              "Local/shared CPU timing is diagnostic, not an allocated training speedup."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
