from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import NPARTS
from jax_orchidee.stomate.daily import (
    accumulate_resp_maint_part,
    prepare_daily_carbon_inputs,
    stomate_accumulate_daily,
    sum_resp_maint_radia,
)
from jax_orchidee.stomate.trace import (
    read_first_pft14_after_alloc_group,
    read_first_pft14_maint_after_group,
)
from jax_orchidee.trace.server_1961 import read_server_records


PFT14 = 13


def test_trace_backed_gpp_accumulation_first_pft14_record_closes_exactly():
    before = read_server_records("stomate_daily", tags="gpp_before_accu", limit=1)[0]
    after = read_server_records("stomate_daily", tags="gpp_after_accu", limit=1)[0]

    assert before["itime"] == after["itime"] == 1
    assert before["ik"] == after["ik"] == 1
    assert before["pft"] == after["pft"] == 14

    result = stomate_accumulate_daily(
        np.asarray([[before["gpp_daily"]]], dtype=np.float64),
        np.asarray([[before["gpp_d"]]], dtype=np.float64),
        bool(before["do_slow"]),
        dt_sechiba=before["dt_sechiba"],
        dt_stomate=before["dt_stomate"],
    )

    assert np.asarray(result)[0, 0] == pytest.approx(after["gpp_daily"], abs=0.0)
    assert before["gpp_d"] == after["gpp_d"]


def test_trace_backed_maintenance_sum_and_accumulation_nonzero_pft14_group_close():
    group = read_first_pft14_maint_after_group(require_nonzero=True)
    radia = np.zeros((1, 14, NPARTS), dtype=np.float64)
    radia[0, PFT14, :] = group.resp_maint_part_radia

    total, flood_root = sum_resp_maint_radia(radia)
    accumulated = accumulate_resp_maint_part(np.zeros_like(radia), radia)

    np.testing.assert_array_equal(np.asarray(flood_root), np.zeros((1, 14)))
    assert group.itime == 49
    assert np.asarray(total)[0, PFT14] == pytest.approx(group.resp_maint_radia_after_sum)
    assert np.allclose(
        np.asarray(accumulated)[0, PFT14, :],
        group.resp_maint_part_after_accum,
        rtol=0.0,
        atol=0.0,
    )
    assert group.resp_maint_radia_after_sum > 0.0


def test_after_alloc_trace_exposes_explicit_f_alloc_but_not_full_alloc_closure():
    group = read_first_pft14_after_alloc_group()
    f_alloc = np.zeros((1, 14, NPARTS), dtype=np.float64)
    f_alloc[0, PFT14, :] = group.f_alloc

    boundary = prepare_daily_carbon_inputs(
        gpp_daily=np.zeros((1, 14), dtype=np.float64),
        biomass=np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        resp_maint_part=np.zeros((1, 14, NPARTS), dtype=np.float64),
        f_alloc=f_alloc,
    )

    assert group.ip == 1
    assert np.sum(group.f_alloc) == pytest.approx(1.0)
    assert "f_alloc" not in boundary.requires_trace
    assert "biomass_before_alloc" in boundary.requires_trace
    assert "bm_alloc" in boundary.requires_trace
