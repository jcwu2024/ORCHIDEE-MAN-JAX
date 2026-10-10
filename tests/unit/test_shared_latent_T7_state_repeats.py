import json

import numpy as np

from scripts.hpc import analyze_shared_latent_T7_state_repeats as analysis


def test_observed_range_overlap_and_direction_are_explicit():
    assert analysis.compare_ranges([2.0, 3.0], [1.0, 2.0])["observed_ranges_overlap"]
    above = analysis.compare_ranges([4.0, 5.0], [1.0, 2.0])
    assert above["direction"] == "actual_above"
    assert above["observed_range_gap"] == 2.0
    assert above["range_gap_over_reference_self_width"] == 2.0
    below = analysis.compare_ranges([1.0, 2.0], [4.0, 6.0])
    assert below["direction"] == "actual_below"
    assert below["observed_range_gap"] == 2.0
    assert below["range_gap_over_reference_self_width"] == 1.0


def test_metrics_keep_strict_failures_despite_high_cosine_and_small_relative_l2():
    reference = np.asarray([1e200, 1.0, 0.0])
    actual = np.asarray([1e200, 1.0001, 0.0])
    result = analysis.state_metrics(actual, reference)
    assert result["strict_failed"] == 1
    assert result["failed_positions"] == [[1]]
    assert result["cosine"] == 1.0
    assert result["relative_L2_to_reference"] < 1e-190
    assert result["descriptive_metrics_do_not_replace_strict_gate"]
    assert analysis.state_metrics(np.zeros(3), np.zeros(3))["relative_L2_to_reference"] == 0


def test_coordinate_statistics_preserve_values_and_population_dispersion():
    row = analysis.scalar_statistics([1.0, 2.0, 3.0])
    assert row["values"] == [1.0, 2.0, 3.0]
    assert row["minimum"] == 1.0 and row["maximum"] == 3.0
    assert row["mean"] == 2.0 and row["range_width"] == 2.0
    np.testing.assert_allclose(row["population_std"], np.sqrt(2 / 3))


def test_analysis_keeps_paid_session_envelope_separate_from_other_session(tmp_path):
    arrays, records = {}, []
    for session, modes in (("paid_B4", ("serial", "batch")),
                           ("dispatch_B1", ("block_each", "dispatch_four"))):
        for mode in modes:
            for i, label in enumerate(("cold-first" if session == "paid_B4" else "warmup", "pair1", "pair2", "pair3")):
                state = np.ones(analysis.EXPECTED_SHAPE)
                state[3, 434] = (1.0 + i / 100) if mode == "serial" else (1.3 + i / 100) if mode == "batch" else (0.5 + i / 100)
                key = f"state{len(records):02d}"
                arrays[key] = state
                records.append({"key": key, "session": session, "mode": mode, "label": label})
    prefix = tmp_path / "states"
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    prefix.with_suffix(".metadata.json").write_text(json.dumps({"records": records, "reported_coordinate_bindings": {}}))
    result = analysis.analyze(prefix, tmp_path / "result.json")
    row = next(row for row in result["coordinates"] if row["coordinate"] == 434)
    assert row["same_session_paid_cold_and_pairs"]["reference"]["minimum"] == 1.0
    assert row["same_session_paid_cold_and_pairs"]["actual"]["minimum"] == 1.3
    assert not row["same_session_paid_cold_and_pairs"]["observed_ranges_overlap"]
    assert row["B4_paired_values_vs_cross_session_dispatch_B1"]["reference"]["minimum"] == 0.5
    assert all(item["absolute_difference_exceeds_observed_B1_self_range"] for item in row["paired_signed_differences"])
    assert len([item for item in result["full_state_comparisons"] if item["kind"] == "same_session_paired_cross"]) == 6
