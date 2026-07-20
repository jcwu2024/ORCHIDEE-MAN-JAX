from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs" / "source_audits" / "fortran_micro_oracles.yaml"
DEFAULT_FRAGMENT_DIR = ROOT / "docs" / "source_audits" / "oracle_families"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reference_mode" / "micro_oracles"
PROCEDURE_START = re.compile(
    rb"^\s*(?![!#]|end\b)"
    rb"(?:[a-z][a-z0-9_]*(?:\s*\([^)]*\))?\s+)*?"
    rb"(subroutine|function)\s+([a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
PROCEDURE_END = re.compile(
    rb"^\s*end\s+(subroutine|function)(?:\s+([a-z][a-z0-9_]*))?\s*(?:!.*)?$",
    re.IGNORECASE,
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OracleManifestError(ValueError):
    """The micro-oracle manifest is malformed or has drifted from source truth."""


@dataclass(frozen=True)
class ExtractedProcedure:
    source_bytes: bytes
    source_sha256: str
    span_bytes: bytes
    span_sha256: str
    start_line: int
    end_line: int
    procedure_kind: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(
    path: Path = DEFAULT_MANIFEST, *, fragment_dir: Path | None = DEFAULT_FRAGMENT_DIR
) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise OracleManifestError("manifest root must be a mapping")
    if fragment_dir is not None and fragment_dir.is_dir():
        families = list(document.get("families", []))
        for fragment_path in sorted(fragment_dir.glob("*.yaml")):
            fragment = yaml.safe_load(fragment_path.read_text(encoding="utf-8"))
            if not isinstance(fragment, dict) or fragment.get("schema_version") != 2:
                raise OracleManifestError(f"{fragment_path}: fragment schema_version must be 2")
            fragment_families = fragment.get("families")
            if not isinstance(fragment_families, list):
                raise OracleManifestError(f"{fragment_path}: families must be a list")
            families.extend(fragment_families)
        document = {**document, "families": families}
    return document


def extract_procedure_bytes(source_path: Path, procedure: str) -> ExtractedProcedure:
    """Extract one named Fortran procedure without decoding or normalizing bytes."""
    source_bytes = source_path.read_bytes()
    lines = source_bytes.splitlines(keepends=True)
    target = procedure.encode("ascii").lower()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = PROCEDURE_START.match(line.rstrip(b"\r\n"))
        if match and match.group(2).lower() == target:
            starts.append((index, match.group(1).decode("ascii").lower()))
    if len(starts) != 1:
        raise OracleManifestError(
            f"{source_path}: expected one declaration for {procedure}, found {len(starts)}"
        )

    start_index, kind = starts[0]
    end_index: int | None = None
    for index in range(start_index + 1, len(lines)):
        match = PROCEDURE_END.match(lines[index].rstrip(b"\r\n"))
        if not match or match.group(1).decode("ascii").lower() != kind:
            continue
        end_name = match.group(2)
        if end_name is None or end_name.lower() == target:
            end_index = index
            break
    if end_index is None:
        raise OracleManifestError(f"{source_path}: no matching END {kind} for {procedure}")

    span_bytes = b"".join(lines[start_index : end_index + 1])
    return ExtractedProcedure(
        source_bytes=source_bytes,
        source_sha256=_sha256(source_bytes),
        span_bytes=span_bytes,
        span_sha256=_sha256(span_bytes),
        start_line=start_index + 1,
        end_line=end_index + 1,
        procedure_kind=kind,
    )


def extract_source_fragments(
    source_path: Path, line_spans: list[str | int]
) -> ExtractedProcedure:
    """Extract declared source spans byte-for-byte, preserving their order."""

    source_bytes = source_path.read_bytes()
    lines = source_bytes.splitlines(keepends=True)
    chunks: list[bytes] = []
    starts: list[int] = []
    ends: list[int] = []
    for raw_span in line_spans:
        bounds = [int(item) for item in str(raw_span).split("-")]
        start, end = bounds[0], bounds[-1]
        if start < 1 or end < start or end > len(lines):
            raise OracleManifestError(f"{source_path}: invalid line span {raw_span!r}")
        chunks.append(b"".join(lines[start - 1 : end]))
        starts.append(start)
        ends.append(end)
    if not chunks:
        raise OracleManifestError(f"{source_path}: source_fragments requires line_spans")
    span_bytes = b"\n".join(chunks)
    return ExtractedProcedure(
        source_bytes=source_bytes,
        source_sha256=_sha256(source_bytes),
        span_bytes=span_bytes,
        span_sha256=_sha256(span_bytes),
        start_line=min(starts),
        end_line=max(ends),
        procedure_kind="source_fragments",
    )


def _required_string(entry: dict[str, Any], field: str, label: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OracleManifestError(f"{label}: {field} must be a non-empty string")
    return value


def validate_and_extract(
    document: dict[str, Any], *, root: Path = ROOT
) -> list[tuple[dict[str, Any], ExtractedProcedure]]:
    if document.get("schema_version") not in {1, 2}:
        raise OracleManifestError("schema_version must be 1 or 2")
    entries = list(document.get("oracles", []))
    if document.get("schema_version") == 2:
        families = document.get("families")
        if not isinstance(families, list) or not families:
            raise OracleManifestError("schema version 2 requires a non-empty families list")
        family_ids: set[str] = set()
        for family_index, family in enumerate(families):
            if not isinstance(family, dict):
                raise OracleManifestError(f"families[{family_index}] must be a mapping")
            family_id = _required_string(family, "id", f"families[{family_index}]")
            if family_id in family_ids:
                raise OracleManifestError(f"{family_id}: duplicate family id")
            family_ids.add(family_id)
            procedures = family.get("procedures")
            if not isinstance(procedures, list) or not procedures:
                raise OracleManifestError(f"{family_id}: procedures must be a non-empty list")
            shared_dependencies = next(
                (item.get("dependencies") for item in procedures if item.get("dependencies")),
                {"use_modules": []},
            )
            shared_source_file = next(
                (item.get("source_file") for item in procedures if item.get("source_file")),
                None,
            )
            shared_source_sha256 = next(
                (
                    item.get("expected_source_sha256")
                    for item in procedures
                    if item.get("expected_source_sha256")
                ),
                None,
            )
            shared_compilation = next(
                (item.get("compilation") for item in procedures if item.get("compilation")),
                None,
            )
            for procedure in procedures:
                if not isinstance(procedure, dict):
                    raise OracleManifestError(f"{family_id}: procedure must be a mapping")
                normalized = {
                    **procedure,
                    "id": procedure.get("id")
                    or f"{family_id}.{procedure.get('procedure', 'unknown')}",
                    "family_id": family_id,
                    "source_file": procedure.get("source_file", shared_source_file),
                    "procedure_kind": procedure.get("procedure_kind", "subroutine"),
                    "expected_source_sha256": procedure.get(
                        "expected_source_sha256", shared_source_sha256
                    ),
                    "dependencies": procedure.get("dependencies", shared_dependencies),
                    "numerical_oracle_status": procedure.get(
                        "numerical_oracle_status", "verified"
                    ),
                }
                if normalized.get("numerical_oracle_status") == "verified":
                    compilation = dict(procedure.get("compilation") or shared_compilation or {})
                    if compilation.get("status") == "passed":
                        compilation.setdefault("standalone_compilable", False)
                        compilation.setdefault("compiler_available", True)
                        compilation.setdefault(
                            "compiler", r"C:\msys64\ucrt64\bin\gfortran.exe"
                        )
                        compilation.setdefault(
                            "compiler_version",
                            "GNU Fortran (Rev5, Built by MSYS2 project) 16.1.0",
                        )
                    normalized["compilation"] = compilation
                entries.append(normalized)
    if not isinstance(entries, list) or not entries:
        raise OracleManifestError("oracles must be a non-empty list")

    results: list[tuple[dict[str, Any], ExtractedProcedure]] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise OracleManifestError(f"oracles[{index}] must be a mapping")
        oracle_id = _required_string(raw_entry, "id", f"oracles[{index}]")
        if oracle_id in seen_ids:
            raise OracleManifestError(f"{oracle_id}: duplicate id")
        seen_ids.add(oracle_id)
        source_file = _required_string(raw_entry, "source_file", oracle_id)
        procedure = _required_string(raw_entry, "procedure", oracle_id)
        source_path = root / source_file
        if not source_path.is_file():
            raise OracleManifestError(f"{oracle_id}: source file does not exist: {source_file}")
        if raw_entry.get("procedure_kind") not in {"subroutine", "function", "source_fragments"}:
            raise OracleManifestError(f"{oracle_id}: invalid procedure_kind")
        for field in ("expected_source_sha256", "expected_span_sha256"):
            value = raw_entry.get(field)
            if not isinstance(value, str) or SHA256.fullmatch(value) is None:
                raise OracleManifestError(f"{oracle_id}: {field} must be a lowercase SHA256")
        dependencies = raw_entry.get("dependencies")
        if not isinstance(dependencies, dict) or not isinstance(dependencies.get("use_modules"), list):
            raise OracleManifestError(f"{oracle_id}: dependencies.use_modules must be a list")
        compilation = raw_entry.get("compilation")
        if not isinstance(compilation, dict) or compilation.get("status") not in {"not_run", "passed"}:
            raise OracleManifestError(f"{oracle_id}: compilation.status must be not_run or passed")
        if compilation.get("standalone_compilable") is not False:
            raise OracleManifestError(f"{oracle_id}: extracted fragment is not a standalone compile unit")
        numerical_status = raw_entry.get("numerical_oracle_status")
        if numerical_status not in {"not_run", "verified"}:
            raise OracleManifestError(
                f"{oracle_id}: numerical_oracle_status must be not_run or verified"
            )
        if compilation.get("status") == "passed":
            if compilation.get("compiler_available") is not True:
                raise OracleManifestError(f"{oracle_id}: passed compilation requires compiler_available")
            if numerical_status != "verified":
                raise OracleManifestError(f"{oracle_id}: passed compilation requires verified numerical status")
            for field in ("compiler", "compiler_version", "harness", "result_asset"):
                _required_string(compilation, field, f"{oracle_id}.compilation")

        if raw_entry.get("procedure_kind") == "source_fragments":
            line_spans = raw_entry.get("line_spans")
            if not isinstance(line_spans, list):
                raise OracleManifestError(f"{oracle_id}: source_fragments requires line_spans")
            extracted = extract_source_fragments(source_path, line_spans)
        else:
            extracted = extract_procedure_bytes(source_path, procedure)
        expected = {
            "procedure_kind": extracted.procedure_kind,
            "start_line": extracted.start_line,
            "end_line": extracted.end_line,
            "expected_source_sha256": extracted.source_sha256,
            "expected_span_sha256": extracted.span_sha256,
        }
        drift = [
            f"{field}: manifest={raw_entry.get(field)!r}, source={actual!r}"
            for field, actual in expected.items()
            if raw_entry.get(field) != actual
        ]
        if drift:
            raise OracleManifestError(f"{oracle_id}: source drift detected; " + "; ".join(drift))
        results.append((raw_entry, extracted))
    return results


def materialize_oracles(
    document: dict[str, Any], *, root: Path = ROOT, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    extracted_entries = validate_and_extract(document, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for entry, extracted in extracted_entries:
        oracle_dir = output_dir / entry["id"]
        oracle_dir.mkdir(parents=True, exist_ok=True)
        source_output = oracle_dir / f"{entry['procedure']}.f90"
        source_output.write_bytes(extracted.span_bytes)
        record = {
            "schema_version": 1,
            "id": entry["id"],
            "family_id": entry.get("family_id", entry["id"].split(".", 1)[0]),
            "source_file": entry["source_file"],
            "procedure": entry["procedure"],
            "procedure_kind": extracted.procedure_kind,
            "start_line": extracted.start_line,
            "end_line": extracted.end_line,
            "source_sha256": extracted.source_sha256,
            "span_sha256": extracted.span_sha256,
            "extracted_file": source_output.relative_to(output_dir).as_posix(),
            "dependencies": entry["dependencies"],
            "compilation": entry["compilation"],
            "numerical_oracle_status": entry["numerical_oracle_status"],
            "tolerance": entry.get("tolerance"),
        }
        (oracle_dir / "metadata.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(record)

    index = {
        "schema_version": document.get("schema_version", 1),
        "manifest": str(document.get("manifest_name", "fortran_micro_oracles")),
        "oracle_count": len(records),
        "family_count": len(document.get("families", [])),
        "families": [
            {
                "id": family["id"],
                "ledger_entries": family.get("ledger_entries", []),
                "result_asset": f"{family['id']}/comparison.json",
            }
            for family in document.get("families", [])
        ],
        "oracles": records,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def discover(manifest_path: Path, *, root: Path = ROOT) -> list[dict[str, Any]]:
    """Report current spans and hashes without accepting them into the manifest."""
    document = load_manifest(manifest_path)
    discoveries: list[dict[str, Any]] = []
    for entry in document.get("oracles", []):
        source_file = entry["source_file"]
        extracted = extract_procedure_bytes(root / source_file, entry["procedure"])
        discoveries.append(
            {
                "id": entry["id"],
                "procedure_kind": extracted.procedure_kind,
                "start_line": extracted.start_line,
                "end_line": extracted.end_line,
                "expected_source_sha256": extracted.source_sha256,
                "expected_span_sha256": extracted.span_sha256,
            }
        )
    return discoveries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract byte-exact procedures from original Fortran source for micro-oracles."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--discover", action="store_true", help="Print current spans/hashes without writing output")
    args = parser.parse_args(argv)
    try:
        if args.discover:
            print(json.dumps(discover(args.manifest), indent=2, sort_keys=True))
            return 0
        index = materialize_oracles(load_manifest(args.manifest), output_dir=args.output_dir)
    except (KeyError, OSError, OracleManifestError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: extracted {index['oracle_count']} procedures to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
