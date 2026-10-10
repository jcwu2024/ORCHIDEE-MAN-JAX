"""Census saved optimized-HLO metadata by historical decoder source line and loop phase."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


SOURCE_SUFFIX = "/research/daily_coarse_graining/daily_operator/strong_baseline_architecture.py"
NORMALIZATION_LINES = set(range(901, 912))
SOURCE_LINE = re.compile(r"source_line=(\d+)")
SOURCE_FILE = re.compile(r'source_file="([^"]+)"')
OP_NAME = re.compile(r'op_name="([^"]+)"')
RESULT_TYPE = re.compile(r"(?:ROOT )?%[^=]+ = ([a-z0-9]+\[[^]]*\])")
MAX_HLO_BYTES = 20_000_000


def phase_for(op_name: str) -> str:
    if "transpose(" in op_name and "while" in op_name and "/body" in op_name:
        return "transpose_while_body"
    if "while" in op_name and "/body" in op_name:
        return "while_body"
    if "while" in op_name and "/condition" in op_name:
        return "while_condition"
    return "outside_while_or_unclassified"


def census(path: Path) -> dict:
    input_bytes = path.stat().st_size
    if input_bytes > MAX_HLO_BYTES:
        raise ValueError(f"saved HLO exceeds the {MAX_HLO_BYTES}-byte offline input limit")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().startswith("HloModule "):
        raise ValueError("saved input is not a recognized textual optimized-HLO module")
    counts: Counter[tuple[int, str, str]] = Counter()
    types: Counter[tuple[int, str]] = Counter()
    samples: dict[tuple[int, str], str] = {}
    matched_lines = 0
    for line in text.splitlines():
        if SOURCE_SUFFIX not in line:
            continue
        file_match = SOURCE_FILE.search(line)
        line_match = SOURCE_LINE.search(line)
        if file_match is None or line_match is None or not file_match.group(1).replace("\\", "/").endswith(SOURCE_SUFFIX):
            continue
        source_line = int(line_match.group(1))
        if source_line not in NORMALIZATION_LINES:
            continue
        matched_lines += 1
        op_match = OP_NAME.search(line)
        op_name = op_match.group(1) if op_match is not None else ""
        phase = phase_for(op_name)
        result_match = RESULT_TYPE.search(line)
        result_type = result_match.group(1) if result_match is not None else "unknown"
        counts[(source_line, phase, result_type)] += 1
        types[(source_line, phase)] += 1
        samples.setdefault((source_line, phase), op_name)

    rows = [
        {"source_line": source_line, "phase": phase, "result_type": result_type, "instructions": count}
        for (source_line, phase, result_type), count in sorted(counts.items())
    ]
    phase_rows = [
        {"source_line": source_line, "phase": phase, "instructions": count,
         "sample_op_name": samples[(source_line, phase)]}
        for (source_line, phase), count in sorted(types.items())
    ]
    return {
        "schema": "shared_decoder_saved_hlo_source_census_v1",
        "status": "completed",
        "input": str(path.resolve()),
        "input_bytes": input_bytes,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_file_suffix": SOURCE_SUFFIX,
        "source_line_scope": [901, 911],
        "matched_instruction_lines": matched_lines,
        "instructions_by_source_line_phase_and_result_type": rows,
        "instructions_by_source_line_phase": phase_rows,
        "interpretation_limit": (
            "Optimized-HLO metadata occurrence counts only; not execution counts, "
            "recompute counts, timing, memory, or a proof that loop-invariant "
            "instructions execute once per mapped row."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hlo",
        type=Path,
        default=Path("outputs/diagnostics/shared-command-buffer-loops-20261001/"
                     "shared-command-buffer-loops-threshold-component-default-20261001.default.native_vjp.hlo.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diagnostics/shared-decoder-recompute-feasibility-20261004/"
                     "original-native-vjp-normalization-census.json"),
    )
    args = parser.parse_args()
    if not args.hlo.is_file():
        parser.error(f"saved HLO does not exist: {args.hlo}")
    try:
        report = census(args.hlo)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    if report["matched_instruction_lines"] == 0:
        parser.error("saved HLO has no instruction metadata attributed to historical normalization lines 901-911")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "matched_instruction_lines": report["matched_instruction_lines"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
