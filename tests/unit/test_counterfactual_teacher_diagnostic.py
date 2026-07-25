from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining import counterfactual_teacher_diagnostic as diagnostic
from research.daily_coarse_graining.counterfactual_teacher_diagnostic import (
    _diagnostic_classification,
    _normalized_metrics,
    _teacher_reentry_packet,
    _validate_oracle_offsets,
)


def test_normalized_metrics_reports_scale_aware_error_and_defined_status():
    metrics = _normalized_metrics(
        np.asarray([3.0, 1.0e20, 8.0]),
        np.asarray([1.0, 7.0, 4.0]),
        np.asarray([2.0, 5.0, 4.0]),
    )

    assert metrics["normalized_rmse"] == pytest.approx(1.0)
    assert metrics["normalized_mae"] == pytest.approx(1.0)
    assert metrics["max_absolute_error"] == pytest.approx(4.0)
    assert metrics["defined_status_mismatches"] == 1
    assert metrics["evaluated_values"] == 2


@pytest.mark.parametrize(
    ("offsets", "days"),
    [([], 7), ([0], 7), ([1, 1], 7), ([3, 1], 7), ([7], 7), ([1], 1)],
)
def test_oracle_offsets_reject_invalid_windows(offsets, days):
    with pytest.raises(ValueError):
        _validate_oracle_offsets(offsets, days)


def test_oracle_offsets_accept_selected_model_fed_days():
    assert _validate_oracle_offsets([1, 3, 6], 7) == (1, 3, 6)


def test_teacher_reentry_rebuilds_complete_finalize_packet(monkeypatch):
    sentinel = SimpleNamespace()
    captured = {}
    overwrite_names = (
        diagnostic.SECHIBA_FINALIZE_SOURCE_FIELDS
        - diagnostic.teacher._SECHIBA_HALF_HOUR_CARRY_FIELDS
    )
    template = {name: np.asarray([0.0]) for name in overwrite_names}
    daily_template = {"counter": np.asarray([5.0])}
    sentinel.fields_by_component = {
        "sechiba_finalize_state": {"fwet_new": np.asarray([3.0])},
        "hydrol_previous_step_state": {"fwet_new": np.asarray([3.0])},
        "slowproc_stomate_previous_step_state": {
            "daily_accumulators": {"counter": np.asarray([0.0])}
        },
    }

    def fake_packet(continuous, discrete, contract, **kwargs):
        captured.update(kwargs)
        captured["continuous"] = continuous
        captured["discrete"] = discrete
        captured["contract"] = contract
        return sentinel

    monkeypatch.setattr(diagnostic, "_packet_from_canonical_state", fake_packet)
    continuous = np.asarray([1.0])
    discrete = {"flag": np.asarray([True])}
    contract = object()

    result = _teacher_reentry_packet(
        continuous,
        discrete,
        contract,
        tstep=95,
        overwritten_finalize_template=template,
        daily_accumulator_template=daily_template,
    )

    assert result is sentinel
    assert captured == {
        "continuous": continuous,
        "discrete": discrete,
        "contract": contract,
        "tstep": 95,
        "template_fields": {
            "sechiba_finalize_state": template,
            "slowproc_stomate_previous_step_state": {
                "daily_accumulators": daily_template
            },
        },
        "require_complete_finalize": True,
    }


def _record(*, operator, sensitivity, completed=True, mask=0, discrete=0):
    return {
        "teacher_reentry": {"completed": completed},
        "teacher_next_vs_clean_next": {
            "normalized_rmse": sensitivity,
            "defined_status_mismatches": mask,
        },
        "teacher_discrete_vs_clean_next": {"mismatches": discrete},
        "neural_next_vs_teacher_next": {"normalized_rmse": operator},
    }


def test_classification_identifies_matched_operator_error():
    result = _diagnostic_classification(
        [_record(operator=0.4, sensitivity=0.1), _record(operator=0.2, sensitivity=0.1)]
    )

    assert result["teacher_reentry_valid"]
    assert (
        result["classification"]
        == "matched_operator_error_at_least_as_large_as_state_sensitivity"
    )
    assert result["operator_to_state_sensitivity_ratio"] == pytest.approx(3.0)


def test_classification_stops_on_reentry_validity_failure():
    result = _diagnostic_classification(
        [_record(operator=0.1, sensitivity=0.2, discrete=1)]
    )

    assert not result["teacher_reentry_valid"]
    assert result["classification"] == "teacher_reentry_or_state_validity_failure"
    assert result["operator_to_state_sensitivity_ratio"] is None


def test_classification_accepts_structured_teacher_execution_failure():
    result = _diagnostic_classification(
        [{"teacher_reentry": {"completed": False, "error_type": "ValueError"}}]
    )

    assert not result["teacher_reentry_valid"]
    assert result["mean_operator_normalized_rmse"] is None
