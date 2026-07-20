from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jax_orchidee.driver.interpolation_core12 import AggregateRequest
from jax_orchidee.driver.interpolation_routing import AggregateRouter


def _target():
    return {
        "lalo": np.array([[0.0, 0.0]]),
        "resolution": np.array([[111_317.0, 111_317.0]]),
        "neighbours": np.full((1, 8), -1),
        "contfrac": np.array([1.0]),
    }


def test_rank_one_request_routes_to_vector_aggregate_core():
    request = AggregateRequest(
        source_rank=1,
        **_target(),
        longitude=np.array([0.0]),
        latitude=np.array([0.0]),
        mask=None,
        max_resolution_lon=100_000.0,
        max_resolution_lat=100_000.0,
        callsign="vector map",
        nbvmax=4,
    )
    packet = AggregateRouter()(request)
    assert packet.ok
    assert packet.sub_index.shape == (1, 4)
    assert packet.sub_index[0, 0] == 1
    assert packet.sub_area[0, 0] > 0.0


def test_gridded_request_routes_to_2d_aggregate_and_preserves_mask():
    axis = np.array([-1.0, 0.0, 1.0])
    lon = np.repeat(axis[:, None], 3, axis=1)
    lat = np.repeat(axis[None, :], 3, axis=0)
    mask = np.zeros((3, 3), dtype=np.int32)
    mask[1, 1] = 1
    request = AggregateRequest(
        source_rank=4,
        **_target(),
        longitude=lon,
        latitude=lat,
        mask=mask,
        max_resolution_lon=-1.0,
        max_resolution_lat=-1.0,
        callsign="grid map",
        nbvmax=4,
    )
    packet = AggregateRouter()(request)
    assert packet.ok
    np.testing.assert_array_equal(packet.sub_index[0, 0], [2, 2])
    assert packet.sub_area[0, 0] > 0.0
    assert np.count_nonzero(packet.sub_area[0] > 0.0) == 1


def test_router_rejects_missing_gridded_mask_and_unconfigured_regxy():
    request = AggregateRequest(
        source_rank=2,
        **_target(),
        longitude=np.zeros((2, 2)),
        latitude=np.zeros((2, 2)),
        mask=None,
        max_resolution_lon=-1.0,
        max_resolution_lat=-1.0,
        callsign="grid map",
        nbvmax=2,
    )
    with pytest.raises(ValueError, match="shared source mask"):
        AggregateRouter()(request)
    with pytest.raises(ValueError, match="RegXY requires explicit"):
        AggregateRouter(grid_type="RegXY")(replace(request, mask=np.ones((2, 2))))
