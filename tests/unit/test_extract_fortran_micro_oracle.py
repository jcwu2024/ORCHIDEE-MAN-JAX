from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "extract_fortran_micro_oracle.py"
SPEC = importlib.util.spec_from_file_location("extract_fortran_micro_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXTRACTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXTRACTOR
SPEC.loader.exec_module(EXTRACTOR)


def test_manifest_extracts_original_procedure_bytes_with_exact_lines_and_hashes(tmp_path):
    manifest = EXTRACTOR.load_manifest()
    index = EXTRACTOR.materialize_oracles(manifest, root=ROOT, output_dir=tmp_path)

    extracted_records = EXTRACTOR.validate_and_extract(manifest, root=ROOT)
    assert index["oracle_count"] == len(extracted_records)
    assert index["family_count"] == len(manifest["families"])
    for entry, extracted in extracted_records:
        output = tmp_path / entry["id"] / f"{entry['procedure']}.f90"
        metadata = json.loads((tmp_path / entry["id"] / "metadata.json").read_text(encoding="utf-8"))
        assert output.read_bytes() == extracted.span_bytes
        assert (extracted.start_line, extracted.end_line) == (entry["start_line"], entry["end_line"])
        assert metadata["span_sha256"] == entry["expected_span_sha256"]
        assert metadata["compilation"]["status"] == "passed"
        assert metadata["numerical_oracle_status"] == "verified"


def test_source_hash_prevents_silent_fortran_drift(tmp_path):
    manifest = EXTRACTOR.load_manifest()
    source_rel = Path(manifest["oracles"][0]["source_file"])
    copied_source = tmp_path / source_rel
    copied_source.parent.mkdir(parents=True)
    copied_source.write_bytes((ROOT / source_rel).read_bytes() + b"\n! drift\n")

    with pytest.raises(EXTRACTOR.OracleManifestError, match="source drift detected"):
        EXTRACTOR.validate_and_extract(manifest, root=tmp_path)


def test_line_or_span_hash_change_requires_explicit_manifest_update():
    manifest = EXTRACTOR.load_manifest()
    line_drift = deepcopy(manifest)
    line_drift["oracles"][0]["end_line"] -= 1
    with pytest.raises(EXTRACTOR.OracleManifestError, match="end_line"):
        EXTRACTOR.validate_and_extract(line_drift, root=ROOT)

    hash_drift = deepcopy(manifest)
    hash_drift["oracles"][1]["expected_span_sha256"] = "0" * 64
    with pytest.raises(EXTRACTOR.OracleManifestError, match="expected_span_sha256"):
        EXTRACTOR.validate_and_extract(hash_drift, root=ROOT)


def test_manifest_declares_dependencies_and_verified_compilation_state():
    manifest = EXTRACTOR.load_manifest()
    for entry in manifest["oracles"]:
        assert entry["dependencies"]["enclosing_module"] == "qsat_moisture"
        assert entry["dependencies"]["use_modules"]
        assert entry["compilation"]["status"] == "passed"
        assert entry["compilation"]["standalone_compilable"] is False
        assert entry["compilation"]["compiler_available"] is True
        assert entry["compilation"]["compiler"] == r"C:\msys64\ucrt64\bin\gfortran.exe"
        assert entry["numerical_oracle_status"] == "verified"


def test_default_generated_assets_stay_under_reference_mode_outputs():
    assert EXTRACTOR.DEFAULT_OUTPUT_DIR == (
        ROOT / "outputs" / "reference_mode" / "micro_oracles"
    )
