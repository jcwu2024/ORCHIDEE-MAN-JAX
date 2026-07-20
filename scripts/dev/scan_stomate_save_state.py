from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "fortran_source" / "ORCHIDEE" / "src_stomate"

MODULE_RE = re.compile(r"^\s*MODULE\s+(\w+)\b", re.IGNORECASE)
END_MODULE_RE = re.compile(r"^\s*END\s+MODULE\b", re.IGNORECASE)
SUBROUTINE_RE = re.compile(r"^\s*SUBROUTINE\s+(\w+)\b", re.IGNORECASE)
FUNCTION_RE = re.compile(r"^\s*(?:[\w(),\s]+\s+)?FUNCTION\s+(\w+)\b", re.IGNORECASE)
END_PROCEDURE_RE = re.compile(r"^\s*END\s+(?:SUBROUTINE|FUNCTION)\b", re.IGNORECASE)
THREADPRIVATE_RE = re.compile(r"!\$OMP\s+THREADPRIVATE\s*\(([^)]*)\)", re.IGNORECASE)
DECL_RE = re.compile(r"\bSAVE\b", re.IGNORECASE)
NAME_RE = re.compile(r"^\s*([A-Za-z]\w*)")


def _strip_comment(line: str) -> str:
    return line.split("!!", 1)[0].split("!", 1)[0]


def _split_var_list(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _record_threadprivate(threadprivate: dict[str, list[int]], line: str, lineno: int) -> None:
    match = THREADPRIVATE_RE.search(line)
    if not match:
        return
    for name in match.group(1).split(","):
        cleaned = name.strip()
        if cleaned:
            threadprivate.setdefault(cleaned.lower(), []).append(lineno)


def scan_fortran_save_state(source_dir: Path) -> dict[str, object]:
    """Scan Fortran ``SAVE`` declarations in ``src_stomate``.

    This is a source-ledger helper, not a full Fortran parser. It targets the
    declaration style used by ORCHIDEE STOMATE: a declaration line containing
    ``SAVE`` and ``::`` followed by one or more variable names.
    """

    records: list[dict[str, object]] = []
    threadprivate: dict[str, list[int]] = {}
    for source in sorted(source_dir.glob("*.f90")):
        module: str | None = None
        procedure: str | None = None
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, raw in enumerate(lines, start=1):
            _record_threadprivate(threadprivate, raw, lineno)
            code = _strip_comment(raw)
            if not code.strip():
                continue
            if END_PROCEDURE_RE.search(code):
                procedure = None
                continue
            if END_MODULE_RE.search(code):
                module = None
                continue
            module_match = MODULE_RE.search(code)
            if module_match and not code.strip().upper().startswith("MODULE PROCEDURE"):
                module = module_match.group(1)
                continue
            subroutine_match = SUBROUTINE_RE.search(code)
            if subroutine_match:
                procedure = subroutine_match.group(1)
            else:
                function_match = FUNCTION_RE.search(code)
                if function_match:
                    procedure = function_match.group(1)

            if "::" not in code or not DECL_RE.search(code):
                continue
            declaration, variables = code.split("::", 1)
            if not DECL_RE.search(declaration):
                continue
            for item in _split_var_list(variables):
                name_match = NAME_RE.match(item)
                if not name_match:
                    continue
                name = name_match.group(1)
                records.append(
                    {
                        "file": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
                        "line": lineno,
                        "module": module,
                        "procedure": procedure,
                        "name": name,
                        "declaration": declaration.strip(),
                        "initializer": "=" in item,
                        "text": raw.strip(),
                    }
                )

    for record in records:
        lines = threadprivate.get(str(record["name"]).lower(), [])
        record["threadprivate"] = bool(lines)
        if lines:
            record["threadprivate_lines"] = lines

    unique_names = sorted({str(record["name"]) for record in records})
    return {
        "source_dir": str(source_dir.relative_to(ROOT) if source_dir.is_relative_to(ROOT) else source_dir),
        "records": records,
        "summary": {
            "records": len(records),
            "unique_names": len(unique_names),
            "files": len({record["file"] for record in records}),
            "threadprivate_records": sum(1 for record in records if record["threadprivate"]),
        },
        "unique_names": unique_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scan = scan_fortran_save_state(args.source_dir)
    text = json.dumps(scan, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
