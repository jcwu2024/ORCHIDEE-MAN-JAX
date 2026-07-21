from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.persistence_baseline import (  # noqa: E402
    _persistence_daily_interface,
    _persistence_ok_leak_interface,
)


def _state():
    slow = {
        "daily_accumulators": {
            "resp_maint_radia": np.ones((1, 14)),
            "flood_root_radia": np.zeros((1, 14)),
        },
        "npp_daily": np.full((1, 14), 2.0),
        "resp_maint": np.full((1, 14), 1.0),
        "resp_growth": np.full((1, 14), 0.5),
        "resp_hetero": np.zeros((1, 14)),
        "litter_above": np.ones((1, 2, 14, 1)),
        "litter_below": np.ones((1, 2, 14, 32, 1)),
        "lignin_struc_above": np.ones((1, 14)),
        "lignin_struc_below": np.ones((1, 14, 32)),
        "litterpart": np.ones((1, 14, 2)),
        "dead_leaves": np.ones((1, 14, 2)),
        "fuel_1hr": np.ones((1, 14, 2, 1)),
        "fuel_10hr": np.ones((1, 14, 2, 1)),
        "fuel_100hr": np.ones((1, 14, 2, 1)),
        "fuel_1000hr": np.ones((1, 14, 2, 1)),
        "carbon_32l": np.ones((1, 3, 14, 32)),
        "DOC": np.ones((1, 14, 32, 2, 7, 1)),
        "deepC_peat": np.ones((1, 32, 14)),
        "interception_storage": np.ones((1, 14, 1)),
    }
    return SimpleNamespace(
        fields_by_component={
            "slowproc_stomate_previous_step_state": slow,
            "hydrol_previous_step_state": {
                "humrel": np.full((1, 14), 0.5),
                "soil_mc": np.full((1, 11, 6), 0.4),
                "snow": np.asarray([3.0]),
            },
            "enerbil_previous_step_state": {"temp_sol": np.asarray([290.0])},
            "thermosoil_previous_step_state": {
                "stempdiag": np.full((1, 11), 288.0),
            },
        }
    )


def test_persistence_daily_interface_uses_state_and_current_forcing():
    forcing = SimpleNamespace(
        temp_air=np.linspace(280.0, 290.0, 48)[:, None],
        precip_rain=np.full((48, 1), 0.1),
        precip_snow=np.zeros((48, 1)),
        u=np.full((48, 1), 3.0),
        v=np.full((48, 1), 4.0),
    )

    fields = _persistence_daily_interface(_state(), forcing, min_wind=0.1)

    assert len(fields) == 17
    np.testing.assert_allclose(fields["t2m_daily"], [285.0])
    np.testing.assert_allclose(fields["precip_daily"], [4.8])
    np.testing.assert_allclose(fields["wspeed_daily"], [5.0])
    np.testing.assert_allclose(fields["gpp_daily"], np.full((1, 14), 3.5))
    assert fields["resp_maint_part"].shape == (1, 14, 12)


def test_persistence_ok_leak_interface_carries_common_and_deep_peat_state():
    state = _state()

    result = _persistence_ok_leak_interface(state)

    slow = state.fields_by_component["slowproc_stomate_previous_step_state"]
    assert result.soilcarbon.carbon_32l is slow["carbon_32l"]
    assert result.soilcarbon.doc is slow["DOC"]
    assert result.soilcarbon.perma_peat.deepc_peat is slow["deepC_peat"]
    assert result.soilcarbon.resp_hetero_soil is slow["resp_hetero"]
