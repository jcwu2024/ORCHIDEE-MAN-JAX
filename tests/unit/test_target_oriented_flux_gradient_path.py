from __future__ import annotations

from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_flux_gradient_path import (
    FIELD_BASIS,
    TARGET_PARAMETER_FAMILIES,
    WINDOW_NAMES,
    classify_gradient_path,
    fine_parameter_family_trees,
    gram_cosine_summary,
    load_target_oriented_flux_gradient_path_contract,
)
from scripts.hpc.audit_target_oriented_flux_gradient_path import (
    _aggregate_alignment,
)


def _field_record(
    *,
    free_norm: float,
    teacher_norm: float,
    alignment: float,
    dominant_family: str,
):
    return {
        "free_gradient": {"l2_norm": free_norm},
        "teacher_state_gradient": {"l2_norm": teacher_norm},
        "free_alignment_to_aggregate": alignment,
        "dominant_fine_parameter_family": dominant_family,
        "dominant_fine_parameter_family_squared_norm_fraction": 0.9,
    }


def _records(*, free_to_teacher: float, dominant_family: str):
    records = {}
    for name in WINDOW_NAMES:
        extreme = name.endswith("extreme")
        maintenance_norm = 100.0 if extreme else 1.0
        records[name] = {
            "fields": {
                "gpp": _field_record(
                    free_norm=1.0,
                    teacher_norm=1.0,
                    alignment=0.1,
                    dominant_family="science_cross_blocks",
                ),
                "maintenance_respiration": _field_record(
                    free_norm=maintenance_norm,
                    teacher_norm=maintenance_norm / free_to_teacher,
                    alignment=0.99,
                    dominant_family=dominant_family,
                ),
                "growth_respiration": _field_record(
                    free_norm=0.5,
                    teacher_norm=0.5,
                    alignment=0.05,
                    dominant_family="primary_heads",
                ),
            }
        }
    return records


def test_frozen_contract_and_window_registry() -> None:
    contract = load_target_oriented_flux_gradient_path_contract()
    assert contract.sha256 == contract.payload["contract_sha256"]
    assert tuple(record["name"] for record in contract.windows) == WINDOW_NAMES
    assert tuple(contract.payload["measurement"]["field_basis"]) == FIELD_BASIS


def test_fine_parameter_partition_is_complete() -> None:
    gradient = {
        "source": np.zeros((2,), dtype=np.float32),
        "target_oriented": {
            name: np.zeros((index + 1,), dtype=np.float32)
            for index, name in enumerate(TARGET_PARAMETER_FAMILIES)
        },
    }
    families = fine_parameter_family_trees(gradient)
    assert set(families) == {"base_source_encoder", *TARGET_PARAMETER_FAMILIES}
    assert sum(np.asarray(value).size for value in families["base_source_encoder"].values()) == 2
    assert sum(np.asarray(value).size for value in families.values() if isinstance(value, np.ndarray)) == sum(
        range(1, len(TARGET_PARAMETER_FAMILIES) + 1)
    )


def test_gram_summary_reports_cross_path_cosines() -> None:
    vectors = np.asarray(((1.0, 0.0), (0.0, 2.0), (1.0, 1.0)))
    report = gram_cosine_summary(vectors @ vectors.T, ("a", "b", "c"))
    assert report["gradient_norms"] == [1.0, 2.0, np.sqrt(2.0)]
    assert report["cosines"][0][1] == 0.0
    assert report["cosines"][0][2] == 1.0 / np.sqrt(2.0)


def test_free_field_order_reconstructs_the_aggregate_gradient() -> None:
    vectors = np.asarray(
        (
            (8.0, 0.0),
            (0.0, 8.0),
            (8.0, 8.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        )
    )
    norm, alignments = _aggregate_alignment(vectors @ vectors.T)
    expected = 0.125 * np.sum(vectors[:3], axis=0)
    assert norm == np.linalg.norm(expected)
    assert np.isclose(alignments[2], 1.0)


def test_classifier_separates_recurrence_and_output_path() -> None:
    contract = load_target_oriented_flux_gradient_path_contract()
    recurrence = classify_gradient_path(
        _records(free_to_teacher=20.0, dominant_family="primary_heads"),
        contract,
    )
    assert recurrence["classification"] == "rollout_recurrence_dominant"

    output = classify_gradient_path(
        _records(free_to_teacher=1.0, dominant_family="primary_heads"),
        contract,
    )
    assert output["classification"] == "primary_or_state_output_path_dominant"


def test_launcher_is_free_gpu_zero_training_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts" / "hpc" / "run_target_oriented_flux_gradient_path_attribution.sh"
    ).read_text(encoding="utf-8")
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_flux_gradient_path" in launcher
    assert "stage-update-1000.pkl" in launcher
    assert "residual-tail-b80494f/report.json" in launcher


def test_step_factory_exposes_only_diagnostic_field_gradients() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(
        encoding="utf-8"
    )
    assert '"seven_flux_field_gradient"' in source
    assert '"thirty_flux_field_gradient"' in source
    assert '"seven_teacher_state_flux_field_gradient"' in source
    assert '"thirty_teacher_state_flux_field_gradient"' in source
