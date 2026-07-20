from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import hydrol_trace_backed_alt_residual, hydrol_trace_backed_full_step
from jax_orchidee.sechiba.hydrol_trace import (
    ALT_RESIDUAL_REQUIRES_TRACE,
    DEFAULT_ALT_RESIDUAL_TRACE_SAMPLES,
    DEFAULT_FULL_STEP_TRACE_SAMPLES,
    FULL_STEP_TRACE_REQUIRES_TRACE,
    HYDROL_LIST_DIRECTED_TRACE_SCHEMAS,
    discover_hydrol_full_step_sample_keys,
    first_matching_records,
    load_first_hydrol_full_step_slice,
    load_hydrol_alt_residual_sample_slices,
    load_hydrol_full_step_sample_slices,
)


TRACE_DIR = ROOT / "outputs" / "server_1961_trace_full_20260623" / "traces"


def test_hydrol_list_directed_reader_streams_first_active_tile4_slice():
    rows = first_matching_records(
        TRACE_DIR / "orchjax_hydrol_main_trace.txt",
        "pre",
        match={"kjit": 1, "ji": 1, "jst": 4},
        count=11,
    )

    assert len(rows) == 11
    assert tuple(row.values["jsl"] for row in rows) == tuple(range(1, 12))
    assert rows[0].values["resolv"] is True
    assert rows[0].values["mask_soiltile"] == 1.0
    assert np.isclose(rows[0].values["dt_days"], 1.0 / 48.0)
    assert "dz_mm" not in HYDROL_LIST_DIRECTED_TRACE_SCHEMAS["pre"]


def test_hydrol_full_step_sampler_loads_configured_first_match_slices():
    sample_keys = discover_hydrol_full_step_sample_keys(TRACE_DIR)
    slices = load_hydrol_full_step_sample_slices(TRACE_DIR)

    assert sample_keys == DEFAULT_FULL_STEP_TRACE_SAMPLES
    assert len(slices) == len(DEFAULT_FULL_STEP_TRACE_SAMPLES)
    assert [
        {
            "kjit": sample.post_tile.values["kjit"],
            "ji": sample.post_tile.values["ji"],
            "jst": sample.post_tile.values["jst"],
        }
        for sample in slices
    ] == list(DEFAULT_FULL_STEP_TRACE_SAMPLES)


def test_hydrol_alt_residual_sampler_loads_branch_samples():
    slices = load_hydrol_alt_residual_sample_slices(TRACE_DIR)
    results = [hydrol_trace_backed_alt_residual(sample) for sample in slices]

    assert len(slices) == len(DEFAULT_ALT_RESIDUAL_TRACE_SAMPLES)
    assert [sample.alt_first_solve.values["kjit"] for sample in slices] == [1, 541]
    assert [bool(result.resolv_expected) for result in results] == [False, True]
    assert all(max(result.max_abs_errors.values()) == 0.0 for result in results)


@pytest.mark.parametrize(
    ("kjit", "rhs_atol"),
    (
        (1, 5.0e-13),
        (2, 8.0e-13),
    ),
)
def test_hydrol_trace_backed_full_step_closes_tile4_samples(kjit, rhs_atol):
    trace_slice = load_first_hydrol_full_step_slice(TRACE_DIR, kjit=kjit, ji=1, jst=4)

    result = hydrol_trace_backed_full_step(trace_slice)

    assert result.requires_trace == FULL_STEP_TRACE_REQUIRES_TRACE
    assert max(result.max_abs_errors.values()) < 1.0e-12
    assert result.max_abs_errors["rhs"] < rhs_atol
    assert result.max_abs_errors["mcl_after_tridiag"] < 1.0e-13
    assert result.max_abs_errors["mc_after_update"] < 1.0e-13
    assert result.max_abs_errors["tmci"] < 1.0e-12
    assert result.max_abs_errors["tmcf"] < 1.0e-12
    assert result.max_abs_errors["check_tr_ns"] < 1.0e-22
    assert result.max_abs_errors["dr_before_corr"] < 1.0e-15
    assert result.max_abs_errors["dr_corrnum"] < 1.0e-18
    assert result.max_abs_errors["dr_after_corr"] < 1.0e-18

    post = trace_slice.post_tile.values
    dr_before = post["dr_ns"] - post["dr_corrnum_ns"]
    assert np.isclose(
        dr_before,
        post["k_bottom"] * post["free_drain_coef"] * post["dt_days"],
        rtol=0.0,
        atol=1.0e-20,
    )
    assert np.isclose(post["dr_ns"], dr_before + post["dr_corrnum_ns"], rtol=0.0, atol=1.0e-24)


@pytest.mark.parametrize(
    ("resolv", "expected_kjit"),
    (
        (False, 1),
        (True, 541),
    ),
)
def test_hydrol_alt_residual_trace_closes_trigger_and_top_boundary(resolv, expected_kjit):
    trace_slice = load_hydrol_alt_residual_sample_slices(TRACE_DIR, samples=({"jst": 4, "resolv": resolv},))[0]
    result = hydrol_trace_backed_alt_residual(trace_slice)

    assert trace_slice.alt_first_solve.values["kjit"] == expected_kjit
    assert len(DEFAULT_ALT_RESIDUAL_TRACE_SAMPLES) == 2
    assert result.requires_trace == ALT_RESIDUAL_REQUIRES_TRACE
    assert max(result.max_abs_errors.values()) < 1.0e-12
    assert result.max_abs_errors["resolv"] == 0.0
    assert result.max_abs_errors["top_rhs"] == 0.0
    assert result.max_abs_errors["top_f"] == 0.0
    assert result.max_abs_errors["top_g1"] == 0.0
