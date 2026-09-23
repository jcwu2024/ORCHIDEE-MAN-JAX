"""Inspect existing JAX GPU executables without tracing or model execution."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from collections import Counter
from pathlib import Path


def large_arrays(hlo: str, minimum_bytes: int = 8 * 2**20):
    """List large array instructions, not a simultaneous live-memory estimate."""
    sizes = {
        "pred": 1,
        "s8": 1,
        "u8": 1,
        "f16": 2,
        "bf16": 2,
        "s32": 4,
        "u32": 4,
        "f32": 4,
        "s64": 8,
        "u64": 8,
        "f64": 8,
        "c64": 8,
        "c128": 16,
    }
    pattern = re.compile(r"^\s*(?:ROOT )?([\w.%_-]+) = (\w+)\[([\d,]*)\](?:\{[^}]*\})? ([\w-]+)\(")
    rows = []
    for line in hlo.splitlines():
        match = pattern.match(line)
        if match is None or match[2] not in sizes:
            continue
        shape = tuple(int(value) for value in match[3].split(",") if value)
        size = math.prod(shape) * sizes[match[2]]
        if size >= minimum_bytes:
            rows.append(
                {
                    "bytes": size,
                    "shape": shape,
                    "dtype": match[2],
                    "operation": match[4],
                    "instruction": line.strip()[:2400],
                }
            )
    return sorted(rows, key=lambda row: row["bytes"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache-files", type=Path, nargs="+")
    source.add_argument("--hlo-files", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.hlo_files:
        rows = []
        for path in args.hlo_files:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                hlo = handle.read()
            arrays = large_arrays(hlo)
            bank_pattern = re.compile(r"\[(?:\d+,)*(?:339|1798)(?:,\d+)*\]")
            bank_lines = "\n".join(line for line in hlo.splitlines() if bank_pattern.search(line))
            bank_arrays = large_arrays(bank_lines, minimum_bytes=0)
            bank_counts = Counter(
                (tuple(row["shape"]), row["operation"])
                for row in bank_arrays
                if any(size in (339, 1798) for size in row["shape"])
            )
            rows.append(
                {
                    "hlo_file": str(path),
                    "large_array_count": len(arrays),
                    "largest_arrays": arrays[:100],
                    "global_bank_instruction_counts": [
                        {"shape": shape, "operation": operation, "instruction_count": count}
                        for (shape, operation), count in bank_counts.most_common(50)
                    ],
                }
            )
        (args.output_dir / "hlo-summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps([{key: row[key] for key in ("hlo_file", "large_array_count")} for row in rows]))
        return
    import jax
    from jax._src import compilation_cache, compiler, xla_bridge

    backend = xla_bridge.get_backend("gpu")
    options = compiler.get_compile_options(1, 1, backend=backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "jax_version": jax.__version__,
        "scope": "existing_executable_no_model_execution",
        "backend_compilations": 0,
        "executables": [],
    }
    for path in args.cache_files:
        start = time.perf_counter()
        print(f"loading {path.name}", flush=True)
        data = compilation_cache.decompress_executable(path.read_bytes())
        serialized, prior_seconds = compilation_cache.extract_executable_and_time(data)
        executable = backend.deserialize_executable(serialized, options)
        del data, serialized
        stats = executable.get_compiled_memory_stats()
        memory = {
            name: int(getattr(stats, name))
            for name in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
        }
        memory["estimated_live_bytes"] = (
            memory["argument_size_in_bytes"]
            + memory["output_size_in_bytes"]
            + memory["temp_size_in_bytes"]
            - memory["alias_size_in_bytes"]
        )
        hlo = "\n".join(module.to_string() for module in executable.hlo_modules())
        with gzip.open(args.output_dir / f"{path.stem}.hlo.txt.gz", "wt", encoding="utf-8") as handle:
            handle.write(hlo)
        buffers = getattr(stats, "serialized_buffer_assignment_proto", b"")
        if buffers:
            (args.output_dir / f"{path.stem}.buffers.pb").write_bytes(buffers)
        arrays = large_arrays(hlo)
        row = {
            "cache_file": str(path),
            "memory": memory,
            "saved_compile_seconds": prior_seconds,
            "inspect_seconds": time.perf_counter() - start,
            "large_array_count": len(arrays),
            "largest_arrays": arrays[:100],
            "buffer_assignment_bytes": len(buffers),
        }
        summary["executables"].append(row)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in row.items() if key != "largest_arrays"}), flush=True)
        del executable, hlo, buffers


if __name__ == "__main__":
    main()
