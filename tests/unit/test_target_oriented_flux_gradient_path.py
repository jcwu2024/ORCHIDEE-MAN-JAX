from __future__ import annotations

from pathlib import Path

import numpy as np



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
















def test_step_factory_exposes_only_diagnostic_field_gradients() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(
        encoding="utf-8"
    )
    assert '"seven_flux_field_gradient"' in source
    assert '"thirty_flux_field_gradient"' in source
    assert '"seven_teacher_state_flux_field_gradient"' in source
    assert '"thirty_teacher_state_flux_field_gradient"' in source
