"""CPU-only compact signatures from bounded failed-GPU-pass HLO proto dumps."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import re
import time


def inspect_file(path):
    from jaxlib import xla_client
    blob = path.read_bytes()
    compressed = blob[:2] == b"\x1f\x8b"
    if compressed:
        blob = gzip.decompress(blob)
    uncompressed_bytes = len(blob)
    # XLA dumps HloProto containing HloModuleProto in length-delimited field1.
    # XlaComputation expects the inner module, whereas our tiny roundtrip
    # fixture directly serialized HloModuleProto.
    def varint(position):
        value = shift = 0
        while True:
            byte = blob[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, position
            shift += 7
    tag, position = varint(0)
    wrapped = False
    if tag == 10:
        length, position = varint(position)
        first = blob[position:position + length]
        if first[:1] == b"\x0a":
            blob = first
            wrapped = True
    module = xla_client.XlaComputation(blob).as_hlo_module()
    options = xla_client._xla.HloPrintOptions.short_parsable()
    options.print_metadata = False
    options.print_large_constants = False
    options.print_backend_config = False
    options.print_operand_shape = True
    options.print_operand_names = False
    options.print_program_shape = False
    compact = module.to_string(options)
    if len(compact) > 256 * 2**20:
        raise MemoryError("compact HLO exceeded 256 MiB bound")
    counts, signatures, examples = Counter(), Counter(), {}
    for line in io.StringIO(compact):
        rhs = line.partition(" = ")[2]
        if re.search(r"(?:^|\s)dot\(", rhs):
            kind = "dot"
        elif "custom_call_target=" in rhs and "cublas" in rhs:
            kind = "cublas_custom_call"
        elif re.search(r"(?:^|\s)fusion\(", rhs):
            counts["fusion"] += 1
            continue
        else:
            continue
        counts[kind] += 1
        algorithm = re.search(r"algorithm=([^,\s}]+)", rhs)
        if algorithm:
            counts["dot_with_algorithm"] += 1
        key = json.dumps({"kind": kind, "types": re.findall(r"\b(f16|f32|f64|bf16)\[[^]]*\]", rhs),
                          "algorithm": algorithm.group(1) if algorithm else None,
                          "shape_operation_prefix": rhs.split("(", 1)[0][:150]}, sort_keys=True)
        signatures[key] += 1
        if key not in examples:
            examples[key] = rhs[:1600].strip()
    return {"file": path.name, "file_bytes": path.stat().st_size, "gzip": compressed,
            "uncompressed_proto_bytes": uncompressed_bytes, "HloProto_envelope_unwrapped": wrapped,
            "compact_hlo_characters": len(compact),
            "instruction_counts": dict(counts),
            "signatures": [{**json.loads(key), "count": count, "example": examples[key]} for key, count in signatures.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior inspection")
    paths = sorted(path for path in args.dump_dir.iterdir() if path.is_file() and ".pb" in path.name)
    if not paths or len(paths) > 20 or sum(path.stat().st_size for path in paths) > 512 * 2**20:
        raise ValueError("expected at most20 proto files and512MiB total compressed data")
    started = time.perf_counter()
    rows = []
    for path in paths:
        rows.append(inspect_file(path))
        args.output.write_text(json.dumps({"status": "inspecting", "files": rows}, indent=2) + "\n")
        print(json.dumps({key: value for key, value in rows[-1].items() if key != "signatures"}), flush=True)
    report = {"status": "inspected", "mode": "CPU_only_proto_read_no_model_compile_or_execution",
              "files": rows, "elapsed_seconds": time.perf_counter() - started,
              "full_HLO_text_exported": False, "metadata_printed": False,
              "scope": "failed public native-AD GPU passes; residual dot signatures are evidence, not automatic error attribution"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
