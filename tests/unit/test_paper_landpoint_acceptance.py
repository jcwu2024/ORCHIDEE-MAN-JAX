from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from jax_orchidee.cli.main import main as cli_main
from jax_orchidee.driver.source_clock import SourceClockBinding, SourceClockRuntime
from jax_orchidee.runners import acceptance, multiyear
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
from jax_orchidee.runners.run_identity import RESULT_PROTOCOL, write_atomic
from scripts.dev.aggregate_paper_landpoint_acceptance import aggregate_summaries
from scripts.dev.aggregate_paper_landpoint_acceptance import main as aggregate_main


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


def _request(args):
    days = multiyear._days_for_years(args.start_year, args.years, int(args.days_per_year))
    clocks = [
        {"calendar": "noleap", "date0": 715765.0, "dt": 1800.0,
         "first_istp": offset * 17520 + 1, "last_istp": offset * 17520 + ndays * 48}
        for offset, ndays in enumerate(days)
    ]
    return multiyear.RunRequest(
        args=args, days=days, fields=(), reference_run_dir=None, histories={},
        identity={
            "version": 1, "runtime": {"source_sha256": "source"}, "batch": "batch",
            "initial_source": "cold-start", "common": f"{args.landpoint_id}:{args.compiled_sechiba_day}",
            "years": [f"year:{offset}" for offset in range(args.years)],
        },
        clocks=clocks, steps_per_day=48, predecessor=None,
    )


def _finished_result(request, error=0.0):
    args = request.args
    payload = _payload((error,) * args.years)
    for index, row in enumerate(payload["years"]):
        row.update(
            requested_days=request.days[index], closed_modelout_days=request.days[index],
            ready_for_requested_days=True, last_tstep=request.days[index] * request.steps_per_day - 1,
            source_clock=request.clocks[index],
        )
    return payload | {
        "landpoint_id": args.landpoint_id, "start_year": args.start_year, "requested_years": args.years,
        "days_per_year": request.days, "initial_state": args.initial_state, "run_identity": request.identity,
    } | {key: getattr(args, key) for key in multiyear.EXECUTION_OPTIONS}


def test_cached_acceptance_requires_identical_run_protocol():
    request = _request(multiyear.parse_run_args(["--years", "2"]))
    payload = _finished_result(request)
    assert _payload_matches_request(payload, request)
    assert not _payload_matches_request({**payload, "run_identity": None}, request)
    assert not _payload_matches_request({**payload, "landpoint_id": "069.0-119.0"}, request)
    assert not _payload_matches_request({**payload, "compiled_sechiba_day": "off"}, request)
    assert not _payload_matches_request({**payload, "years": list(reversed(payload["years"]))}, request)
    payload["years"][1]["source_clock"] = {"last_istp": 17520}
    assert not _payload_matches_request(payload, request)


@pytest.mark.parametrize(
    ("outcome", "exit_code"),
    [("exception", 1), ("nonzero", 1), ("missing", 1), ("partial", 1), ("corrupt", 1), ("interrupt", 130)],
)
def test_current_failure_never_accepts_old_same_path_output(monkeypatch, tmp_path, outcome, exit_code):
    site = "319.0-057.0"
    output = tmp_path / site / "compiled_1961_1961.json"
    output.parent.mkdir()
    old_request = _request(multiyear.parse_run_args(["--landpoint-id", site, "--years", "1", "--initial-state", "cold-start"]))
    old_data = json.dumps(_finished_result(old_request)).encode()
    output.write_bytes(old_data)
    monkeypatch.setattr(acceptance, "prepare_run", _request)

    def run(command, *, request):
        assert request.args.resume_checkpoints == "off"
        assert request.args.output != output
        if outcome == "exception":
            raise RuntimeError("asserted execution failure")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "corrupt":
            request.args.output.write_text("{", encoding="utf-8")
        if outcome in {"nonzero", "partial"}:
            current = _finished_result(request)
            if outcome == "partial":
                current["ready_for_requested_years"] = False
            request.args.output.write_text(json.dumps(current), encoding="utf-8")
        return 1 if outcome == "nonzero" else 0

    monkeypatch.setattr(acceptance, "run_multiyear_main", run)
    assert cli_main(["validate-landpoints", "--landpoint-id", site, "--force", "--output-dir", str(tmp_path)]) == exit_code
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["scientific_acceptance_count"] == 0
    expected = "compiled_interrupted" if outcome == "interrupt" else "compiled_execution_failure"
    assert summary["results"][0]["classification"] == expected
    assert output.read_bytes() == old_data
    assert not aggregate_summaries((summary,), expected_landpoints=1)["all_scientifically_accepted"]


def test_matching_finished_result_is_reused_and_conflicting_mode_is_recomputed(monkeypatch, tmp_path):
    argv = ["validate-landpoints", "--landpoint-id", "319.0-057.0", "--output-dir", str(tmp_path)]
    calls = []

    def run(command, *, request):
        calls.append(request.args.compiled_sechiba_day)
        request.args.output.write_text(json.dumps(_finished_result(request)), encoding="utf-8")
        return 0

    monkeypatch.setattr(acceptance, "prepare_run", _request)
    monkeypatch.setattr(acceptance, "run_multiyear_main", run)
    assert cli_main(argv) == cli_main(argv) == 0
    assert calls == ["on"]
    output = tmp_path / "319.0-057.0" / "compiled_1961_1961.json"
    wrong = json.loads(output.read_text(encoding="utf-8"))
    wrong["compiled_sechiba_day"] = "off"
    output.write_text(json.dumps(wrong), encoding="utf-8")
    assert cli_main(argv) == 0
    assert calls == ["on", "on"]


def test_checkpoint_resume_falls_back_from_torn_or_wrong_clock_pairs(monkeypatch, tmp_path):
    request = _request(multiyear.parse_run_args(["--years", "2", "--year-checkpoint-dir", str(tmp_path)]))
    rows = _finished_result(request)["years"]
    pft_layout = {"layout_id": "paper_250919_legacy14"}
    monkeypatch.setattr(multiyear, "driver_year_handoff_state_gaps", lambda state: ())
    binding = SourceClockBinding("control", "noleap", 715765.0, 1800.0, 715765, 0.0, 0)

    def save(count, *, bad_clock=False):
        last = count * 17520 + int(bad_clock)
        state = SimpleNamespace(
            source_clock_runtime=SourceClockRuntime(binding, last, 0, last - 17519), tstep=17519,
        )
        return multiyear._write_year_checkpoint(
            checkpoint_dir=tmp_path, request=request, year=1960 + count, ndays=365,
            state=state, summaries=rows[:count], started_at=0,
            previous_state_cache_metadata=None, pft_layout_metadata=pft_layout,
        )

    save(1)
    save(2, bad_clock=True)
    assert multiyear._resume_checkpoint(request, pft_layout)[0] == 1
    second = save(2)
    assert multiyear._resume_checkpoint(request, pft_layout)[0] == 2
    Path(second["state_cache"]).write_bytes(b"interrupted")
    assert multiyear._resume_checkpoint(request, pft_layout)[0] == 1
    for different in (
        request.identity | {"common": "strict"},
        request.identity | {"runtime": {"source_sha256": "other"}},
        request.identity | {"years": ["changed-input", "year:1"]},
    ):
        assert multiyear._resume_checkpoint(replace(request, identity=different), pft_layout) is None
    (tmp_path / "multiyear_partial_through_1961.json").write_text('{"end_year": 1961}', encoding="utf-8")
    assert multiyear._resume_checkpoint(request, pft_layout) is None


def test_days_per_year_accepts_an_exact_contiguous_schedule():
    assert multiyear._days_for_years(1961, 2, [1, 2]) == [1, 2]
    with pytest.raises(ValueError, match="list length"):
        multiyear._days_for_years(1961, 2, [1])


def test_scientific_compiled_failure_still_runs_strict_diagnostic(monkeypatch, tmp_path):
    calls = []

    def run(command, *, request):
        compiled = request.args.compiled_sechiba_day == "on"
        calls.append(compiled)
        request.args.output.write_text(json.dumps(_finished_result(request, error=0.02 if compiled else 0.0)), encoding="utf-8")
        return 0

    monkeypatch.setattr(acceptance, "prepare_run", _request)
    monkeypatch.setattr(acceptance, "run_multiyear_main", run)
    result = cli_main(["validate-landpoints", "--landpoint-id", "319.0-057.0", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert result == 1
    assert calls == [True, False]
    assert summary["results"][0]["classification"] == "compiled_path_regression"
    assert summary["scientific_acceptance_count"] == 0


def test_unbound_restart_backed_request_cannot_be_accepted(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("unbound restart must fail before dispatch")

    monkeypatch.setattr(acceptance, "prepare_run", _request)
    monkeypatch.setattr(acceptance, "run_multiyear_main", unexpected)
    assert cli_main([
        "validate-landpoints", "--landpoint-id", "319.0-057.0",
        "--initial-state", "restart-backed", "--output-dir", str(tmp_path),
    ]) == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "requires a declared driver/SECHIBA/STOMATE restart triplet" in summary["results"][0]["compiled_run"]["error"]


def test_aggregate_requires_current_identity_and_every_declared_shard():
    historical = {
        "tolerance_policy": {}, "results": [
            {"landpoint_id": "001", "classification": "accepted_compiled", "scientific_acceptance_pass": True},
        ],
    }
    old = aggregate_summaries((historical,), expected_landpoints=1)
    assert old["scientific_acceptance_count"] == 1
    assert not old["all_scientifically_accepted"]
    with pytest.raises(ValueError, match="different scientific requests"):
        aggregate_summaries((historical, {**historical, "start_year": 1962}), expected_landpoints=2)
    row = {
        **historical["results"][0],
        "compiled_run": {"returncode": 0, "batch_identity": "batch"},
    }
    expected = ("001", "319")
    shard = {
        **historical, "result_protocol": RESULT_PROTOCOL, "selection_ids": expected, "max_points": None,
        "shard_count": 2, "shard_index": 0, "results": [row],
    }
    assert not aggregate_summaries((shard,), expected_landpoint_ids=expected)["complete_landpoint_set"]
    second = {**shard, "shard_index": 1, "results": [{**row, "landpoint_id": "319"}]}
    assert aggregate_summaries((second, shard), expected_landpoint_ids=expected)["all_scientifically_accepted"]
    reported = aggregate_summaries((shard, second), expected_landpoints=2)
    assert reported["scientific_acceptance_count"] == 2
    assert reported["complete_landpoint_set"]
    assert not reported["all_scientifically_accepted"]
    legacy = tuple({**summary, "result_protocol": 1} for summary in (shard, second))
    assert not aggregate_summaries(legacy, expected_landpoint_ids=expected)["all_scientifically_accepted"]
    with pytest.raises(ValueError, match="different production environments or shared inputs"):
        aggregate_summaries((shard, second | {
            "results": [{**second["results"][0], "compiled_run": {"returncode": 0, "batch_identity": "other"}}],
        }), expected_landpoint_ids=expected)
    assert not aggregate_summaries(({**shard, "results": []},), expected_landpoint_ids=expected)["all_scientifically_accepted"]


def test_public_aggregate_rejects_same_count_shards_from_a_different_selection(monkeypatch, tmp_path):
    target = ("001.0-071.0", "069.0-119.0")
    other = (target[0], "319.0-057.0")
    batch = tmp_path / "batch"
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"selected_landpoint_ids": target}), encoding="utf-8")

    def run(command, *, request):
        request.args.output.write_text(json.dumps(_finished_result(request)), encoding="utf-8")
        return 0

    monkeypatch.setattr(acceptance, "prepare_run", _request)
    monkeypatch.setattr(acceptance, "run_multiyear_main", run)

    def shard(selection, index):
        path = tmp_path / "selection.json"
        path.write_text(json.dumps({"selected_landpoint_ids": selection}), encoding="utf-8")
        return cli_main([
            "validate-landpoints", "--selection", str(path), "--shard-count", "2",
            "--shard-index", str(index), "--output-dir", str(batch), "--force",
        ])

    def aggregate():
        code = aggregate_main(["--input-dir", str(batch), "--expected-selection", str(expected)])
        data = json.loads((batch / "aggregate.json").read_text(encoding="utf-8"))
        return code, data

    assert shard(target, 0) == shard(target, 1) == 0
    code, data = aggregate()
    assert code == 0 and data["all_scientifically_accepted"]
    assert shard(other, 1) == 0
    code, data = aggregate()
    assert code == 1 and not data["all_scientifically_accepted"]
    assert data["scientific_acceptance_count"] == 2
    assert data["missing_landpoint_ids"] == [target[1]]
    assert data["unexpected_landpoint_ids"] == [other[1]]
    assert data["selection_errors"]


def test_shard_order_and_per_shard_limit_are_bound_to_the_expected_roster():
    expected = ("001", "069", "319", "359", "777")
    def summary(index, sites, *, selection=expected, count=2, limit=None):
        return {
            "result_protocol": RESULT_PROTOCOL, "tolerance_policy": {},
            "selection_ids": selection, "shard_count": count, "shard_index": index, "max_points": limit,
            "results": [
                {
                    "landpoint_id": site, "classification": "accepted_compiled", "scientific_acceptance_pass": True,
                    "compiled_run": {"returncode": 0, "batch_identity": "batch"},
                }
                for site in sites
            ],
        }

    first = summary(0, expected[::2])
    second = summary(1, expected[1::2])
    assert aggregate_summaries((second, first), expected_landpoint_ids=expected)["all_scientifically_accepted"]
    assert aggregate_summaries(
        (summary(0, expected[::2], limit=3), summary(1, expected[1::2], limit=3)),
        expected_landpoint_ids=expected,
    )["all_scientifically_accepted"]
    cap_result = aggregate_summaries(
        (summary(0, expected[::2][:2], limit=2), summary(1, expected[1::2][:2], limit=2)),
        expected_landpoint_ids=expected,
    )
    assert not cap_result["selection_errors"]
    assert cap_result["missing_landpoint_ids"] == ["777"]
    assert not cap_result["all_scientifically_accepted"]
    assert aggregate_summaries(
        (summary(0, expected[::2][:2], limit=2), summary(1, expected[1::2][:2], limit=2)),
        expected_landpoint_ids=expected[:4],
    )["all_scientifically_accepted"]
    different_rule = aggregate_summaries((first, summary(1, expected[1::2], limit=2)), expected_landpoint_ids=expected)
    assert different_rule["selection_errors"] and not different_rule["all_scientifically_accepted"]
    wrong_order = aggregate_summaries((summary(0, reversed(expected[::2])), second), expected_landpoint_ids=expected)
    assert wrong_order["selection_errors"] and not wrong_order["all_scientifically_accepted"]
    unsharded = tuple(summary(0, [site], selection=[site], count=1) for site in reversed(expected))
    assert aggregate_summaries(unsharded, expected_landpoint_ids=expected)["all_scientifically_accepted"]
    wrong_single = (*unsharded[:-1], summary(0, ["999"], selection=["999"], count=1))
    assert not aggregate_summaries(wrong_single, expected_landpoint_ids=expected)["all_scientifically_accepted"]


def test_atomic_publication_uses_short_same_directory_names_and_cleans_failed_replace(monkeypatch, tmp_path):
    target = tmp_path / f"used_run.{'0' * 64}.def"
    target.write_bytes(b"old")
    replace_file = Path.replace
    observed = []

    def replace_short_temp(path, destination):
        assert path.parent == target.parent and destination == target
        assert len(path.name) < 16 and path.name != target.name
        assert target.read_bytes() == b"old"
        observed.append(path)
        raise OSError("injected publication failure")

    monkeypatch.setattr(Path, "replace", replace_short_temp)
    with pytest.raises(OSError, match="injected"):
        write_atomic(target, b"new")
    assert observed and not observed[0].exists() and target.read_bytes() == b"old"
    monkeypatch.setattr(Path, "replace", replace_file)
    write_atomic(target, b"new")
    assert target.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [target]


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
