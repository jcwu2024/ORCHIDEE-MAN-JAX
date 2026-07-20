from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol_reference import (
    HYDROL_RESTART_FIELDS,
    hydrol_reference_restart_path,
    inventory_hydrol_restart_fields,
    read_hydrol_restart_anchors,
    read_hydrol_restart_fields,
)


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def test_hydrol_restart_inventory_exposes_local_anchor_fields():
    path = hydrol_reference_restart_path(CONFIG, "sechiba_start.nc")

    inventory = inventory_hydrol_restart_fields(path, (*HYDROL_RESTART_FIELDS, "rhs", "tmat_e"))

    assert set(HYDROL_RESTART_FIELDS).issubset(inventory.present)
    assert "rhs" in inventory.missing
    assert "tmat_e" in inventory.missing


def test_read_hydrol_restart_anchors_normalizes_shapes_and_known_values():
    anchors = read_hydrol_restart_anchors(hydrol_reference_restart_path(CONFIG, "sechiba_start.nc"))

    assert anchors.njsc.shape == (1,)
    assert anchors.moistc.shape == (1, 6, 11)
    assert anchors.moistcl.shape == (1, 6, 11)
    assert anchors.free_drain_coef.shape == (1, 6)
    assert anchors.veget.shape == (1, 14)
    assert anchors.veget_max.shape == (1, 14)
    assert anchors.humrel.shape == (1, 14)
    assert anchors.vegstress.shape == (1, 14)
    assert anchors.wtp.shape == (1,)
    assert anchors.wt_ab.shape == (1,)
    assert anchors.wt_ab_tide.shape == (1,)
    assert anchors.zwt_force.shape == (1, 6)
    assert anchors.water2infilt.shape == (1, 6)
    assert anchors.ae_ns.shape == (1, 6)
    assert anchors.resdist.shape == (1, 6)

    assert np.array_equal(anchors.njsc, np.asarray([2], dtype=np.int32))
    assert np.allclose(anchors.free_drain_coef, np.ones((1, 6)))
    assert np.allclose(anchors.resdist, [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    assert np.allclose(anchors.ae_ns[0, 3], 0.00011632716997991419)
    assert np.allclose(anchors.zwt_force, np.full((1, 6), 1.0e20))


def test_hydrol_restart_reader_raises_on_missing_variable():
    path = hydrol_reference_restart_path(CONFIG, "sechiba_start.nc")

    try:
        read_hydrol_restart_fields(path, ["moistc", "rhs"])
    except KeyError as exc:
        assert "rhs" in str(exc)
    else:
        raise AssertionError("missing HYDROL restart field did not raise")
