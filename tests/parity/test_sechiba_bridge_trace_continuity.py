from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.bridge import (
    HYDROL_BRIDGE_TRACE_PROVENANCE,
    hydrol_bridge_trace_pairs,
    read_hydrol_bridge_trace_slice,
)


def test_pft14_hydrol_exports_match_slowproc_and_stomate_entry_trace_fields():
    trace_slice = read_hydrol_bridge_trace_slice(kjit=1, ji=1, jv=14, jst=4, jsl=1)

    assert trace_slice.pft["jst_pref"] == 4
    assert trace_slice.before_slowproc["jst_pref"] == 4
    assert trace_slice.provenance == HYDROL_BRIDGE_TRACE_PROVENANCE

    for pair in hydrol_bridge_trace_pairs(trace_slice, target="slowproc"):
        assert pair.source_value == pytest.approx(pair.target_value), pair.mapping.name

    for pair in hydrol_bridge_trace_pairs(trace_slice, target="stomate"):
        assert pair.source_value == pytest.approx(pair.target_value), pair.mapping.name


def test_tile4_top_layer_exports_match_stomate_top_tile_fields():
    trace_slice = read_hydrol_bridge_trace_slice(kjit=1, ji=1, jv=14, jst=4, jsl=1)

    assert trace_slice.tile_layer["jst"] == 4
    assert trace_slice.tile_layer["jsl"] == 1
    assert trace_slice.tile["jst"] == 4

    assert trace_slice.tile_layer["soil_mc"] == pytest.approx(trace_slice.before_stomate["soil_mc_top_tile"])
    assert trace_slice.tile_layer["wat_flux"] == pytest.approx(trace_slice.before_stomate["wat_flux_top_tile"])
    assert trace_slice.tile["drainage_per_soil"] == pytest.approx(
        trace_slice.before_stomate["drainage_per_soil_tile"]
    )
    assert trace_slice.tile["runoff_per_soil"] == pytest.approx(
        trace_slice.before_stomate["runoff_per_soil_tile"]
    )
    assert trace_slice.tile["runoff2peat"] == pytest.approx(trace_slice.before_stomate["runoff2peat_tile"])


@pytest.mark.parametrize("jsl", (1, 2, 10))
def test_layer_records_are_read_by_jsl_for_pft14_preferred_tile(jsl):
    trace_slice = read_hydrol_bridge_trace_slice(kjit=1, ji=1, jv=14, jst=4, jsl=jsl)

    assert trace_slice.layer["jv"] == 14
    assert trace_slice.layer["jst_pref"] == 4
    assert trace_slice.layer["jsl"] == jsl
    assert trace_slice.tile_layer["jv"] == 14
    assert trace_slice.tile_layer["jst"] == 4
    assert trace_slice.tile_layer["jsl"] == jsl

    assert trace_slice.layer["mc_layh"] == pytest.approx(trace_slice.tile_layer["mc_layh_s"])
    assert trace_slice.layer["mcl_layh"] == pytest.approx(trace_slice.tile_layer["mcl_layh_s"])
