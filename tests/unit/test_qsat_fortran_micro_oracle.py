from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEV_SCRIPTS = ROOT / "scripts" / "dev"
if str(DEV_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DEV_SCRIPTS))

BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_qsat_micro_oracle", DEV_SCRIPTS / "build_qsat_micro_oracle.py"
)
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
BUILD = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = BUILD
BUILD_SPEC.loader.exec_module(BUILD)

RUN_SPEC = importlib.util.spec_from_file_location(
    "run_qsat_micro_oracle", DEV_SCRIPTS / "run_qsat_micro_oracle.py"
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
RUN = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = RUN
RUN_SPEC.loader.exec_module(RUN)


def test_compiled_harness_embeds_each_original_procedure_byte_exactly(tmp_path):
    source_path, hashes = BUILD.compose_harness(tmp_path)
    compile_unit = source_path.read_bytes()
    extractor = BUILD._load_extractor()
    manifest = extractor.load_manifest()
    extracted = extractor.validate_and_extract(manifest, root=ROOT)

    for entry, procedure in extracted:
        if entry.get("family_id") is not None:
            continue
        assert compile_unit.count(procedure.span_bytes.rstrip(b"\r\n")) == 1
        assert hashes[f"{entry['procedure']}_span_sha256"] == entry["expected_span_sha256"]
    assert hashes["source_sha256"] == manifest["oracles"][0]["expected_source_sha256"]


def test_committed_qsat_oracle_assets_record_boundaries_phases_and_passing_errors():
    output_dir = RUN.DEFAULT_OUTPUT_DIR
    result = json.loads((output_dir / "comparison.json").read_text(encoding="ascii"))
    inputs = json.loads((output_dir / "inputs.json").read_text(encoding="ascii"))

    assert result["status"] == "passed"
    assert result["build"]["compiler"] == r"C:\msys64\ucrt64\bin\gfortran.exe"
    assert len(result["build"]["hashes"]["source_sha256"]) == 64
    assert all(comparison["passed"] for comparison in result["comparisons"])
    coverage = {row["coverage"] for name in ("qsatcalc", "dev_qsatcalc") for row in inputs[name]}
    assert {"ice", "liquid", "phase_rounding_boundary"} <= coverage
    assert any("lower_valid" in item for item in coverage)
    assert any("upper_valid" in item for item in coverage)

    with (output_dir / "point_comparisons.csv").open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert all(row["passed"] == "true" for row in rows)


def test_qsat_micro_oracle_rebuilds_runs_and_matches_jax_owner(tmp_path):
    result = RUN.run_oracle(tmp_path)
    assert result["status"] == "passed"
    assert {item["name"] for item in result["comparisons"]} == {
        "qsfrict_init",
        "qsatcalc",
        "dev_qsatcalc",
    }
