from copy import deepcopy

from scripts.hpc.probe_saved_mixed_forward import (
    _checkpoint_path,
    _compare_forward_records,
    parser,
)


def _forward():
    return {
        "input_metadata_sha256": "same-input",
        "combined_loss": 2.0,
        "per_sample_terms": [[1.0, 2.0, 3.0]] * 8,
        "physical_counts": {
            "per_sample": [{"nonfinite": 0}] * 8,
            "totals": {"nonfinite": 0},
            "all_zero": True,
        },
    }


def test_parser_matches_metadata_wrapper_and_infers_recovery_checkpoints(tmp_path):
    cli = parser()
    values = []
    for action in cli._actions:
        if action.required:
            values.extend((action.option_strings[0], str(tmp_path / (action.dest + ".json"))))
    args = cli.parse_args(values)
    assert args.training_checkpoint is None
    assert args.reference_training_checkpoint is None
    assert _checkpoint_path(args.training_report, None) == args.training_report.with_suffix(
        ".recovery.checkpoint.pkl").resolve()


def test_forward_parity_uses_declared_tolerance_and_exact_counts():
    reference = _forward()
    candidate = deepcopy(reference)
    candidate["combined_loss"] += 1.5e-5
    candidate["per_sample_terms"][0][0] += 1.0e-5
    assert _compare_forward_records(candidate, reference)["passed"]

    candidate["physical_counts"]["totals"]["nonfinite"] = 1
    comparison = _compare_forward_records(candidate, reference)
    assert not comparison["passed"]
    assert not comparison["physical_counts_equal"]

    candidate = deepcopy(reference)
    candidate["physical_counts"]["totals"]["nonfinite"] = 1
    candidate["physical_counts"]["all_zero"] = False
    reference = deepcopy(candidate)
    comparison = _compare_forward_records(candidate, reference)
    assert comparison["physical_counts_equal"]
    assert not comparison["candidate_physical_counts_all_zero"]
    assert not comparison["passed"]


def test_forward_parity_rejects_different_input_or_out_of_tolerance_value():
    reference = _forward()
    candidate = deepcopy(reference)
    candidate["input_metadata_sha256"] = "different-input"
    candidate["per_sample_terms"][3][2] += 1.0e-3
    comparison = _compare_forward_records(candidate, reference)
    assert not comparison["same_input"]
    assert not comparison["per_sample_terms_close"]
    assert not comparison["passed"]
