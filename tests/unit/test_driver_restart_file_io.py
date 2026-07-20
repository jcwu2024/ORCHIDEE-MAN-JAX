from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.dim2 import Dim2DriverRestart  # noqa: E402
from jax_orchidee.driver.restart_file_io import (  # noqa: E402
    DEFAULT_DRIVER_RESTART_SCHEMA,
    DRIVER_RESTART_VARIABLES,
    read_dim2_driver_restart,
    write_dim2_driver_restart,
)
from jax_orchidee.stomate.restart_io import StomateRestartPhysicalState  # noqa: E402
from scripts.dev.extract_stomate_restart_netcdf_schema import extract_schema  # noqa: E402


def _template() -> Path:
    root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    paths = sorted(root.rglob("driver_start.nc"))
    if not paths:
        pytest.skip("paper driver_start.nc is unavailable")
    return paths[0]


def _physical(path: Path) -> StomateRestartPhysicalState:
    with Dataset(path) as dataset:
        return StomateRestartPhysicalState(
            **{
                name: np.asarray(dataset.variables[name][:])
                for name in ("nav_lon", "nav_lat", "nav_lev", "time", "time_steps")
            }
        )


def test_driver_schema_matches_fortran_paper_file() -> None:
    actual = json.loads(DEFAULT_DRIVER_RESTART_SCHEMA.read_text(encoding="utf-8"))
    expected = extract_schema(
        _template(),
        scope="ORCHIDEE-MAN PFT14 independent single-landpoint DIM2 driver restart",
        fortran_file="fortran_source/ORCHIDEE/src_driver/dim2_driver.f90",
        subroutine="dim2_driver main restart block",
        lines="722-796, 1139-1157, 1411-1429",
    )
    assert actual == expected
    assert len(actual["dimensions"]) == 4
    assert len(actual["variables"]) == 18


def test_driver_standalone_restart_roundtrips_every_prognostic_field(
    tmp_path: Path,
) -> None:
    template = _template()
    original = read_dim2_driver_restart(template)
    changed = Dim2DriverRestart(
        **{
            field: np.asarray(getattr(original, field)) + (index + 1) / 16.0
            for index, field in enumerate(DRIVER_RESTART_VARIABLES)
        }
    )
    output = tmp_path / "driver_restart.nc"

    report = write_dim2_driver_restart(
        output,
        state=changed,
        physical_state=_physical(template),
    )
    restored = read_dim2_driver_restart(output)

    for field in DRIVER_RESTART_VARIABLES:
        np.testing.assert_array_equal(getattr(restored, field), getattr(changed, field))
    assert report.written_fields == tuple(DRIVER_RESTART_VARIABLES)
    assert report.unsupported_fields == ()


def test_driver_writer_rejects_undefined_or_wrong_shape(tmp_path: Path) -> None:
    template = _template()
    original = read_dim2_driver_restart(template)
    with pytest.raises(ValueError, match="undefined"):
        write_dim2_driver_restart(
            tmp_path / "undefined.nc",
            state=original.__class__(**{**original.__dict__, "z0": None}),
            physical_state=_physical(template),
        )
    with pytest.raises(ValueError, match="normalized shape"):
        write_dim2_driver_restart(
            tmp_path / "bad_shape.nc",
            state=original.__class__(
                **{**original.__dict__, "z0": np.zeros((2,), dtype=np.float64)}
            ),
            physical_state=_physical(template),
        )
