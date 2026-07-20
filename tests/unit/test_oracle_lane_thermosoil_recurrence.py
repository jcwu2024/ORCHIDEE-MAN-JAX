from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _main_tail() -> bytes:
    return _span(990, 993) + b"\n" + _span(1011, 1033)


def test_undefined_ok_zimov_is_not_promoted_without_policy_decision():
    manifest = yaml.safe_load((ROOT / "docs/source_audits/oracle_families/thermosoil_recurrence.yaml").read_text())
    assert len(manifest["families"]) == 1
    family = manifest["families"][0]
    assert family["ledger_entries"] == ["thermosoil.active.cwrr_recurrence"]
    assert family["fortran_undefined_contract"][0]["id"] == "thermosoil_main.local_ok_zimov"
    pending = manifest["pending_families"][0]
    assert pending["numerical_oracle_status"] == "all_defined_behavior_verified"
    assert pending["component_status"] == "verified"
    assert pending["resolution_status"] == "candidate_resolved_requires_policy_decision"
    assert pending["fortran_undefined_contract"][0]["id"] == "thermosoil_main.local_ok_zimov"


def test_source_spans_are_exact_and_current():
    manifest = yaml.safe_load((ROOT / "docs/source_audits/oracle_families/thermosoil_recurrence.yaml").read_text())
    procedures = manifest["pending_families"][0]["procedures"]
    assert procedures[0]["expected_source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert procedures[0]["expected_span_sha256"] == hashlib.sha256(_span(3426, 3463)).hexdigest()
    assert procedures[1]["expected_span_sha256"] == hashlib.sha256(_main_tail()).hexdigest()


def test_cached_component_result_is_partial_not_ledger_evidence():
    result = json.loads((ROOT / "outputs/reference_mode/micro_oracles/thermosoil_cwrr_recurrence_defined_tail/comparison.json").read_text())
    assert result["status"] == "passed"
    assert result["component_status"] == "verified"
    assert result["ledger_status"] == "partial"
    assert result["verified_ledger_entries"] == []


def test_cached_coefficient_call_graph_covers_all_defined_outputs():
    result = json.loads((ROOT / "outputs/reference_mode/micro_oracles/thermosoil_cwrr_recurrence_coef/comparison.json").read_text())
    assert result["status"] == "passed"
    assert result["defined_output_status"] == "verified"
    assert result["verified_ledger_entries"] == []
    fields = {comparison["name"] for comparison in result["comparisons"]}
    explicit = {name.removeprefix("explicit.") for name in fields if name.startswith("explicit.")}
    no_explicit = {name.removeprefix("no_explicit.") for name in fields if name.startswith("no_explicit.")}
    assert {"pcapa", "pcapa_en", "pkappa", "profil", "supp", "cgrnd", "dgrnd", "soilcap", "soilcap_pft", "soilflx", "soilflx_pft", "pcapa_snow", "pkappa_snow", "lambda_snow", "cgrnd_snow", "dgrnd_snow"} == explicit
    assert {"pcapa", "pcapa_en", "pkappa", "cgrnd", "dgrnd", "soilcap", "soilcap_pft", "soilflx", "soilflx_pft", "lambda_snow", "cgrnd_snow", "dgrnd_snow"} == no_explicit
