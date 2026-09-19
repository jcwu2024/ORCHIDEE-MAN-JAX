"""Regression checks for attribution and limits of saved timing evidence."""
from scripts.hpc.summarize_mixed_timing import (
    frequency_scenarios,
    logged_cycles,
    summarize_report,
)


def test_cycle_excludes_monitor_and_checkpoint_gaps():
    report = {"updates": [{"update": 1, "seconds": 7}, {"update": 2, "seconds": 8}]}
    log = "\n".join([
        "mixed_learning stage=training updates=0 seconds=10.0",
        "mixed_learning stage=optimizer_completed updates=1 seconds=18.0",
        "mixed_learning stage=training updates=1 seconds=19.0",
        "mixed_learning stage=optimizer_completed updates=2 seconds=28.0",
        "mixed_learning stage=checkpoint_saved updates=2 seconds=31.0",
        "mixed_learning stage=training updates=2 seconds=32.0",
    ])
    result = logged_cycles(report, log)
    assert result["count"] == 1
    assert result["cycles"][0]["outside_update_seconds"] == 2


def test_resumed_report_does_not_subtract_cumulative_updates_from_process_wall():
    result = summarize_report({"updates": [{"update": 2, "seconds": 20}],
                               "elapsed_seconds": 10, "resumed_from": "prior.pkl"})
    assert result["outside_updates_seconds"] is None
    assert summarize_report({"updates": [], "monitors": []})["completed_updates"] == 0


def test_frequency_model_does_not_halve_overlapping_dual_compute():
    result = frequency_scenarios({"dual_device_profile": {"timing": {
        "daily_only": {"median_seconds": 6}, "dual": {"median_seconds": 8},
    }}}, updates=8)
    half = result["scenarios"][1]
    assert half["windows_per_horizon"] == 4
    assert half["modeled_update_seconds"] == 7
    assert half["wall_reduction_fraction_vs_every_update"] == .125
