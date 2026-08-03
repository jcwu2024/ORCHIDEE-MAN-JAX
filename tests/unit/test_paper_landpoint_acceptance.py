from __future__ import annotations

from argparse import Namespace

from jax_orchidee.runners.acceptance import (
    KEY_MODEL_OUTPUTS,
    _evaluate,
    _multiyear_command,
    _payload_matches_request,
)
from jax_orchidee.runners.multiyear import (
    PRODUCTION_COMPILED_SECHIBA_DAY_DEFAULT,
    PRODUCTION_NUMPY_ACCUMULATOR_DEFAULT,
    _latest_year_checkpoint,
    _paper_reference_pft_index,
)
from scripts.dev.aggregate_paper_landpoint_acceptance import aggregate_summaries


def _delta(reference: float, error: float) -> dict[str, float]:
    return {
        "jax": reference + error,
        "reference": reference,
        "abs_error": abs(error),
        "rel_error": abs(error) / max(abs(reference), 1.0),
    }


def _payload(errors: tuple[float, ...], *, diagnostic_error: float = 0.0) -> dict[str, object]:
    years = []
    for offset, error in enumerate(errors):
        modelout = {name: _delta(float(index + 1), error) for index, name in enumerate(KEY_MODEL_OUTPUTS)}
        years.append(
            {
                "year": 1961 + offset,
                "annual_reference_delta": modelout,
                "annual_reference_field_delta": {"GPP": _delta(1.0, diagnostic_error)},
                "annual_reference_diagnostic_delta": {"LAI": _delta(1.0, diagnostic_error)},
            }
        )
    return {"ready_for_requested_years": True, "years": years}


def _evaluate_payload(payload: dict[str, object], *, years: int) -> dict[str, object]:
    return _evaluate(
        payload,
        requested_years=years,
        science_atol=1.0e-5,
        science_rtol=1.0e-3,
        diagnostic_atol=2.0e-6,
        diagnostic_rtol=1.0e-7,
        max_normalized_bias=1.0e-3,
        max_normalized_rmse=1.0e-3,
    )


def test_scientific_acceptance_is_independent_from_strict_internal_diagnostic_gate():
    result = _evaluate_payload(_payload((1.0e-4, -1.0e-4), diagnostic_error=5.0e-5), years=2)

    assert result["scientific_acceptance_pass"] is True
    assert result["numerical_diagnostic_pass"] is False


def test_scientific_acceptance_rejects_persistent_normalized_drift():
    result = _evaluate_payload(_payload((5.0e-3, 5.0e-3)), years=2)

    assert result["annual_scientific_tolerance_pass"] is False
    assert result["series_statistics_pass"] is False
    assert result["scientific_acceptance_pass"] is False


def test_scientific_acceptance_requires_every_requested_reference_year():
    result = _evaluate_payload(_payload((0.0,)), years=2)

    assert result["ready"] is False
    assert result["annual_truth_complete"] is False
    assert result["scientific_acceptance_pass"] is False


def test_latest_year_checkpoint_ignores_unrelated_files(tmp_path):
    (tmp_path / "paper_driver_1961_year_end_state.pkl").touch()
    expected = tmp_path / "paper_driver_1964_year_end_state.pkl"
    expected.touch()
    (tmp_path / "paper_driver_bad_year_end_state.pkl").touch()
    (tmp_path / "summary.json").touch()

    assert _latest_year_checkpoint(tmp_path) == expected


def test_production_defaults_enable_compiled_day_and_reject_numpy_accumulator(tmp_path):
    assert PRODUCTION_COMPILED_SECHIBA_DAY_DEFAULT == "on"
    assert PRODUCTION_NUMPY_ACCUMULATOR_DEFAULT == "off"

    args = Namespace(
        start_year=1961,
        years=50,
        days_per_year="365",
        initial_state="cold-start",
    )
    compiled = _multiyear_command(
        "001.0-071.0",
        args=args,
        compiled=True,
        output=tmp_path / "compiled.json",
        checkpoint_dir=tmp_path / "compiled",
    )
    strict = _multiyear_command(
        "001.0-071.0",
        args=args,
        compiled=False,
        output=tmp_path / "strict.json",
        checkpoint_dir=tmp_path / "strict",
    )

    compiled_options = dict(zip(compiled[::2], compiled[1::2], strict=True))
    strict_options = dict(zip(strict[::2], strict[1::2], strict=True))
    assert compiled_options["--compiled-sechiba-day"] == "on"
    assert compiled_options["--numpy-accumulator"] == "off"
    assert strict_options["--compiled-sechiba-day"] == "off"
    assert strict_options["--numpy-accumulator"] == "off"


def test_archived_fortran_history_pft_slot_resolves_from_named_stable_layout():
    assert _paper_reference_pft_index("configs/orchidee_man_250919.yaml") == 13


def test_cached_acceptance_requires_identical_run_protocol():
    args = Namespace(
        start_year=1961,
        years=1,
        days_per_year="365",
        initial_state="cold-start",
    )
    payload = {
        "landpoint_id": "319.0-057.0",
        "start_year": 1961,
        "requested_years": 1,
        "days_per_year": [365],
        "initial_state": "cold-start",
        "ready_for_requested_years": True,
        "years": [{"year": 1961}],
    }
    assert _payload_matches_request(payload, landpoint_id="319.0-057.0", args=args)

    payload["initial_state"] = "restart-backed"
    assert not _payload_matches_request(payload, landpoint_id="319.0-057.0", args=args)

    payload["initial_state"] = "cold-start"
    assert not _payload_matches_request(payload, landpoint_id="069.0-119.0", args=args)


def test_acceptance_aggregate_tracks_completeness_and_worst_landpoint():
    summaries = (
        {
            "tolerance_policy": {"scientific_rtol": 1.0e-3},
            "results": [
                {
                    "landpoint_id": "001.0-071.0",
                    "scientific_acceptance_pass": True,
                    "classification": "accepted_compiled",
                    "compiled_evaluation": {
                        "series": {"GPP_model": {"max_abs_error": 1.0e-5, "normalized_bias": 1.0e-5, "normalized_rmse": 1.0e-5}}
                    },
                },
                {
                    "landpoint_id": "003.0-077.0",
                    "scientific_acceptance_pass": False,
                    "classification": "shared_semantic_input_or_reference_mismatch",
                    "compiled_evaluation": {
                        "series": {"GPP_model": {"max_abs_error": 2.0e-3, "normalized_bias": -2.0e-3, "normalized_rmse": 2.0e-3}}
                    },
                },
            ],
        },
    )

    result = aggregate_summaries(summaries, expected_landpoints=2)

    assert result["complete_landpoint_set"] is True
    assert result["scientific_acceptance_count"] == 1
    assert result["all_scientifically_accepted"] is False
    assert result["failed_landpoint_ids"] == ["003.0-077.0"]
    assert result["worst_compiled_scientific_metrics"]["GPP_model.normalized_bias"]["landpoint_id"] == "003.0-077.0"
