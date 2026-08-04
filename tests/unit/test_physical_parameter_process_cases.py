from __future__ import annotations

from research.daily_coarse_graining.physical_parameter_process_cases import (
    run_process_gradient_cases,
)


def test_process_gradient_matrix_covers_active_inactive_and_threshold_pairs():
    results = run_process_gradient_cases()

    assert len(results) == 7
    assert all(result.passed for result in results)
    assert {result.pair_class for result in results} == {
        "smooth_active",
        "inactive",
        "threshold_adjacent",
    }
    assert {result.pair_id for result in results} >= {
        "process.vmax__vcmax25",
        "process.maintenance__maint_resp_slope_c",
        "process.allocation__alloc_min",
        "process.gap__residence_time",
    }
