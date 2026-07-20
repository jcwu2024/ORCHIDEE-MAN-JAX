from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_runtime_day_result,
)
from jax_orchidee.stomate.carbon_kernels import NPARTS  # noqa: E402
from jax_orchidee.trace.server_1961 import read_server_records  # noqa: E402


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DAY2_START_ITIME = 49
STEPS_PER_DAY = 48
PFT14_INDEX = 13
ATOL = 1.0e-12


@pytest.mark.skipif(
    os.environ.get("ORCJAX_RUN_SLOW_ORACLES") != "1",
    reason="set ORCJAX_RUN_SLOW_ORACLES=1 for the 65-second integrated trace oracle",
)
def test_cold_start_day2_48_steps_match_point_001_active_path_trace_oracle():
    """Validate only point 001's cold-start Day2 active path, not a general oracle."""

    day = paper_1961_driver_cold_start_runtime_day_result(
        CONFIG,
        day_index=2,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
        single_pass_daily_fold=False,
        retain_stomate_step_results=True,
    )

    assert day.daily_process_fold is not None, day.missing_components
    step_results = day.daily_process_fold.maintenance.step_results
    assert len(step_results) == STEPS_PER_DAY

    end_itime = DAY2_START_ITIME + STEPS_PER_DAY - 1
    scan_limit = end_itime * 14 * NPARTS
    groups: dict[int, list[dict[str, object]]] = {}
    for row in read_server_records("stomate_maint", tags="maint_after", limit=scan_limit):
        itime = int(row["itime"])
        if DAY2_START_ITIME <= itime <= end_itime and int(row["ji"]) == 1 and int(row["jv"]) == 14:
            groups.setdefault(itime, []).append(row)

    assert sorted(groups) == list(range(DAY2_START_ITIME, end_itime + 1))

    accumulated = np.zeros(NPARTS, dtype=np.float64)
    max_abs_error = 0.0
    for itime, step_result in zip(range(DAY2_START_ITIME, end_itime + 1), step_results, strict=True):
        group = sorted(groups[itime], key=lambda row: int(row["part"]))
        assert [int(row["part"]) for row in group] == list(range(1, NPARTS + 1))

        actual_parts = np.asarray(step_result.resp_maint_part, dtype=np.float64)[0, PFT14_INDEX, :NPARTS]
        accumulated += actual_parts
        expected_parts = np.asarray([row["resp_maint_part_radia"] for row in group], dtype=np.float64)
        expected_accumulated = np.asarray(
            [row["resp_maint_part_after_accum"] for row in group], dtype=np.float64
        )
        expected_sums = {float(row["resp_maint_radia_after_sum"]) for row in group}
        assert len(expected_sums) == 1

        actual_sum = float(np.sum(actual_parts))
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(actual_parts - expected_parts))),
            float(np.max(np.abs(accumulated - expected_accumulated))),
            abs(actual_sum - expected_sums.pop()),
        )

    assert max_abs_error <= ATOL
