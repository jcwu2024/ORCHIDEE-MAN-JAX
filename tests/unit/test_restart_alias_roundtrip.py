from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import DriverFirstDayStomateStatePacket  # noqa: E402
from jax_orchidee.stomate.reference import read_restart_carbon_32l, read_restart_doc  # noqa: E402
from scripts.dev.audit_restart_alias_roundtrip import (  # noqa: E402
    build_audit,
    many_to_one_alias_groups,
    project_packet_aliases,
)


def _write_alias_sentinel_restart(path: Path) -> dict[str, np.ndarray]:
    dims = {
        "time": 1,
        "element": 2,
        "pool": 3,
        "deep": 4,
        "pft": 2,
        "y": 2,
        "x": 3,
    }
    raw: dict[str, np.ndarray] = {}
    with Dataset(path, "w") as dataset:
        for name, length in dims.items():
            dataset.createDimension(name, length)

        carbon_shape = tuple(dims[name] for name in ("time", "deep", "pft", "y", "x"))
        for source_index, name in enumerate(("carbon_32l_a", "carbon_32l_s", "carbon_32l_p"), start=1):
            sentinel = source_index * 1_000_000.0 + np.arange(np.prod(carbon_shape)).reshape(carbon_shape)
            dataset.createVariable(name, "f8", ("time", "deep", "pft", "y", "x"))[:] = sentinel
            raw[name] = sentinel

        doc_shape = tuple(dims[name] for name in ("time", "element", "pool", "deep", "pft", "y", "x"))
        for source_index, name in enumerate(("freedoc", "adsdoc"), start=1):
            sentinel = source_index * 10_000_000.0 + np.arange(np.prod(doc_shape)).reshape(doc_shape)
            dataset.createVariable(
                name,
                "f8",
                ("time", "element", "pool", "deep", "pft", "y", "x"),
            )[:] = sentinel
            raw[name] = sentinel
    return raw


def _normalized_carbon(raw: np.ndarray) -> np.ndarray:
    selected = raw[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(6, 2, 4)


def _normalized_doc(raw: np.ndarray) -> np.ndarray:
    selected = raw[0]
    return np.transpose(selected, (4, 5, 3, 2, 1, 0)).reshape(6, 2, 4, 3, 2)


def test_declared_alias_axes_roundtrip_reader_packet_with_shape_sentinels(tmp_path: Path):
    restart = tmp_path / "sentinel_restart.nc"
    raw = _write_alias_sentinel_restart(restart)

    packet = DriverFirstDayStomateStatePacket(
        day_index=1,
        fields={
            "carbon_32l": read_restart_carbon_32l(restart),
            "DOC": read_restart_doc(restart),
        },
        provenance_by_field={},
    )
    projected = project_packet_aliases(packet)

    assert packet.fields["carbon_32l"].shape == (6, 3, 2, 4)
    assert packet.fields["DOC"].shape == (6, 2, 4, 2, 3, 2)
    for name in ("carbon_32l_a", "carbon_32l_s", "carbon_32l_p"):
        assert projected[name].shape == (6, 2, 4)
        np.testing.assert_array_equal(projected[name], _normalized_carbon(raw[name]))
    for name in ("freedoc", "adsdoc"):
        assert projected[name].shape == (6, 2, 4, 3, 2)
        np.testing.assert_array_equal(projected[name], _normalized_doc(raw[name]))


def test_alias_audit_uses_real_production_writer_and_closes():
    groups = many_to_one_alias_groups()
    audit = build_audit()
    records = {record["canonical"]: record for record in audit["records"]}

    assert groups == {
        "DOC": ("adsdoc", "freedoc"),
        "carbon_32l": ("carbon_32l_a", "carbon_32l_p", "carbon_32l_s"),
    }
    assert audit["closed"] is True
    assert audit["summary"] == {
        "many_to_one_groups": 2,
        "declared_pack_axes": 2,
        "undeclared_pack_axes": 0,
        "real_restart_writer_roundtrips": 1,
        "alias_variables_compared": 5,
    }
    assert records["carbon_32l"]["pack_axis"] == 1
    assert records["DOC"]["pack_axis"] == 3
    assert records["carbon_32l"]["pack_members"] == [
        {"alias": "carbon_32l_a", "index": 0},
        {"alias": "carbon_32l_s", "index": 1},
        {"alias": "carbon_32l_p", "index": 2},
    ]
    assert records["DOC"]["pack_members"] == [
        {"alias": "freedoc", "index": 0},
        {"alias": "adsdoc", "index": 1},
    ]
    roundtrip = audit["production_roundtrip"]
    assert roundtrip["passed"] is True
    assert all(roundtrip["checks"].values())
    assert all(item["exact"] for item in roundtrip["canonical_comparisons"].values())
    assert all(item["exact"] for item in roundtrip["alias_comparisons"].values())
    assert all(item["passed"] for item in roundtrip["variable_metadata"].values())
    assert audit["limitations"] == []
