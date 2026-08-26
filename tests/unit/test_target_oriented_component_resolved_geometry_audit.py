from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.hpc.audit_target_oriented_component_resolved_geometry import (
    PROTECTION_SOURCE_NAMES,
    SOURCE_NAMES,
    _cosine_report,
    _group_variance_report,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import AUDIT_UPDATES

ROOT = Path(__file__).resolve().parents[2]


def test_component_resolved_geometry_source_registry_matches_the_frozen_contract():
    payload = json.loads(
        (
            ROOT
            / "manifests"
            / "coarse_graining"
            / "target_oriented_flux_basis_amendment_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert tuple(payload["zero_update_gate"]["gradient_sources"]) == SOURCE_NAMES
    assert tuple(payload["zero_update_gate"]["audit_updates"]) == AUDIT_UPDATES
    assert PROTECTION_SOURCE_NAMES == (
        "one_day_state_and_process",
        "seven_day_independent_stock",
        "thirty_day_independent_stock",
    )


def test_component_resolved_geometry_cosine_and_variance_reports_are_exact():
    gram = np.asarray([[4.0, -2.0], [-2.0, 1.0]], dtype=np.float64)
    report = _cosine_report(gram, ("left", "right"))

    assert report["gradient_norms"] == [2.0, 1.0]
    assert report["cosine_matrix"] == [[1.0, -1.0], [-1.0, 1.0]]

    group_gram = np.eye(len(AUDIT_UPDATES), dtype=np.float64)
    variance = _group_variance_report(group_gram)
    assert variance == {
        "gradient_norm_minimum": 1.0,
        "gradient_norm_median": 1.0,
        "gradient_norm_maximum": 1.0,
        "pairwise_cosine_minimum": 0.0,
        "pairwise_cosine_median": 0.0,
        "pairwise_cosine_negative_fraction": 0.0,
    }


def test_component_resolved_geometry_launcher_is_free_node_and_zero_update_only():
    runner = (
        ROOT / "scripts" / "hpc" / "audit_target_oriented_component_resolved_geometry.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts" / "hpc" / "run_target_oriented_component_resolved_geometry_audit.sh"
    ).read_text(encoding="utf-8")

    assert "target_oriented_adamw_update" not in runner
    assert "save_target_oriented_stage_checkpoint" not in runner
    assert '"optimizer_updates_retained": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "gnall" not in launcher
