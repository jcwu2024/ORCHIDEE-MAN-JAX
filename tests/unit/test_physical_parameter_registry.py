from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.daily_coarse_graining.physical_parameter_registry import (
    DEFAULT_REGISTRY_PATH,
    load_physical_parameter_registry,
    registry_summary,
)


def _write_mutation(tmp_path: Path, mutate) -> Path:
    payload = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registry_has_bounded_red_mangrove_shortlist_and_complete_counts() -> None:
    registry = load_physical_parameter_registry()
    summary = registry_summary(registry)

    assert summary["status"] == "accepted_candidate_inventory"
    assert summary["parameter_families"] == 34
    assert summary["scalar_components"] == 91
    assert summary["classification_counts"] == {
        "discrete_control": 2,
        "inversion_candidate": 9,
        "not_identifiable_now": 6,
        "sensitivity_only": 17,
    }
    assert registry.first_wave == (
        "vcmax25",
        "maintenance_temperature_intercept",
        "alloc_min",
        "residence_time",
    )
    assert len(registry.second_wave) == 5


def test_registry_separates_pft_capability_and_soil_owned_parameters() -> None:
    registry = load_physical_parameter_registry()
    by_id = registry.by_id

    assert by_id["vcmax25"].axis_owner == "pft"
    assert by_id["salinity_response_floor"].axis_owner == "mangrove_capability_global"
    assert by_id["soil_hydraulic_conductivity_table"].axis_owner == "soil_class"
    assert by_id["phenology_threshold_family"].classification == "not_identifiable_now"
    assert by_id["tide_forcing_length"].classification == "discrete_control"


def test_registry_rejects_source_hash_drift(tmp_path: Path) -> None:
    path = _write_mutation(
        tmp_path,
        lambda payload: payload["source_files"][0].__setitem__("sha256", "0" * 64),
    )

    with pytest.raises(ValueError, match="source hash drift"):
        load_physical_parameter_registry(path)


def test_registry_rejects_duplicate_parameter_ownership(tmp_path: Path) -> None:
    def mutate(payload):
        payload["parameters"][1]["run_def_keys"] = ["VCMAX25"]

    path = _write_mutation(tmp_path, mutate)
    with pytest.raises(ValueError, match="one registry owner"):
        load_physical_parameter_registry(path)


def test_registry_rejects_promoting_a_discrete_control_to_the_shortlist(tmp_path: Path) -> None:
    def mutate(payload):
        payload["shortlist"]["second_wave_after_priors_and_controlled_perturbations"].append(
            "ok_laidev"
        )

    path = _write_mutation(tmp_path, mutate)
    with pytest.raises(ValueError, match="only inversion candidates"):
        load_physical_parameter_registry(path)


def test_registry_rejects_count_drift(tmp_path: Path) -> None:
    path = _write_mutation(
        tmp_path,
        lambda payload: payload["expected_counts"].__setitem__("parameter_families", 999),
    )

    with pytest.raises(ValueError, match="parameter family count drift"):
        load_physical_parameter_registry(path)
