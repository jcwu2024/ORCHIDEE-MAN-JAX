from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.reference import (  # noqa: E402
    read_stomate_daily_accumulator_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
)
from jax_orchidee.driver.restart_export import (  # noqa: E402
    write_stomate_restart_from_day_end_packet,
)
from jax_orchidee.parameters.pft_catalog import (  # noqa: E402
    build_pft_run_layout,
    load_pft_catalog,
)
from jax_orchidee.stomate.restart_io import (  # noqa: E402
    DEFAULT_STOMATE_RESTART_SCHEMA,
    StomateRestartPhysicalState,
    create_stomate_restart_skeleton_from_schema,
    read_stomate_readstart_states_from_template,
    write_stomate_full_writerestart_states,
)
from scripts.dev.extract_stomate_restart_netcdf_schema import (  # noqa: E402
    extract_schema,
)


def _template() -> Path:
    case_root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    paths = sorted(case_root.rglob("stomate_start.nc"))
    if not paths:
        pytest.skip("no real paper stomate_start.nc under reference/OUT")
    return paths[0]


def _physical(path: Path) -> StomateRestartPhysicalState:
    with Dataset(path) as dataset:
        return StomateRestartPhysicalState(
            nav_lon=np.asarray(dataset.variables["nav_lon"][:]),
            nav_lat=np.asarray(dataset.variables["nav_lat"][:]),
            nav_lev=np.asarray(dataset.variables["nav_lev"][:]),
            time=np.asarray(dataset.variables["time"][:]),
            time_steps=np.asarray(dataset.variables["time_steps"][:]),
        )


def _states(path: Path):
    entry = read_stomate_restart_entry_state(path)
    season = read_stomate_restart_season_state(path)
    daily = read_stomate_daily_accumulator_state(path)
    with Dataset(path) as dataset:
        states = read_stomate_readstart_states_from_template(
            path,
            t2m=np.asarray(daily.t2m_daily),
            nvm=entry.age.shape[1],
            nslm=season.tsoil_month.shape[1],
            ndeep=entry.carbon_32l.shape[3],
            nsnow=int(dataset.variables["O2_snow"].shape[2]),
            nvert=int(dataset.variables["uo_0"].shape[1]),
            months_num=int(dataset.variables["fwet_series"].shape[1]),
            ncarb=entry.carbon.shape[1],
            nlitt=entry.litter.shape[1],
            nbpools=int(dataset.variables["MatrixV"].shape[1]),
        )
    return states


def _assert_state_equal(expected: object, actual: object) -> None:
    for field in expected._fields:
        if field == "provenance" or field.startswith("read_input_"):
            continue
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if np.isscalar(expected_value):
            assert actual_value == expected_value, field
        else:
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=field)


def test_audited_schema_is_exact_extraction_of_paper_template() -> None:
    expected = extract_schema(_template())
    actual = json.loads(DEFAULT_STOMATE_RESTART_SCHEMA.read_text(encoding="utf-8"))
    assert actual == expected
    assert len(actual["dimensions"]) == 26
    assert len(actual["variables"]) == 169
    assert actual["single_landpoint_scatter"] == {
        "index_g": [1],
        "grid_shape": [1, 1],
        "fortran_provenance": {
            "file": "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90",
            "subroutine": "writerestart",
            "lines": "1751-2944 restput_p scatter calls",
        },
    }


def test_standalone_writer_roundtrips_all_writerestart_state(tmp_path: Path) -> None:
    template = _template()
    original = _states(template)
    output = tmp_path / "standalone_stomate_restart.nc"

    report = write_stomate_full_writerestart_states(
        output,
        physical_state=_physical(template),
        entry_state=original.entry_state,
        season_state=original.season_state,
        daily_state=original.daily_state,
        gas_state=original.gas_state,
        remainder_state=original.remainder_state,
    )
    restored = _states(output)

    _assert_state_equal(original.entry_state, restored.entry_state)
    _assert_state_equal(original.season_state, restored.season_state)
    _assert_state_equal(original.daily_state, restored.daily_state)
    _assert_state_equal(original.gas_state, restored.gas_state)
    _assert_state_equal(original.remainder_state, restored.remainder_state)
    assert len(report.written_fields) == 161
    assert len(report.validated_derived_fields) == 6
    assert report.unsupported_fields == ()

    schema = json.loads(DEFAULT_STOMATE_RESTART_SCHEMA.read_text(encoding="utf-8"))
    with Dataset(output) as dataset:
        assert dataset.file_format == schema["file_format"]
        assert list(dataset.dimensions) == list(schema["dimensions"])
        assert list(dataset.variables) == list(schema["variables"])
        for name, declaration in schema["variables"].items():
            variable = dataset.variables[name]
            assert str(variable.dtype) == declaration["dtype"]
            assert list(variable.dimensions) == declaration["dimensions"]
            assert variable.chunking() == declaration["chunking"]
            assert variable.endian() == declaration["endian"]
            assert {
                attribute: variable.getncattr(attribute)
                for attribute in variable.ncattrs()
                if attribute != "_FillValue"
            } == declaration["attributes"]


def test_standalone_skeleton_rejects_nonpaper_scatter_and_wrong_axes(
    tmp_path: Path,
) -> None:
    physical = _physical(_template())
    with pytest.raises(ValueError, match="index_g"):
        create_stomate_restart_skeleton_from_schema(
            tmp_path / "bad_index.nc",
            physical.__class__(**{**physical.__dict__, "index_g": (2,)}),
        )
    with pytest.raises(ValueError, match="nav_lon normalized shape"):
        create_stomate_restart_skeleton_from_schema(
            tmp_path / "bad_shape.nc",
            physical.__class__(
                **{**physical.__dict__, "nav_lon": np.zeros((2, 1))}
            ),
        )


def test_standalone_restart_records_stable_pft_layout_metadata(tmp_path: Path) -> None:
    catalog = load_pft_catalog(
        ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"
    )
    layout = build_pft_run_layout(
        catalog,
        layout_id="paper_250919_legacy14",
        fractions=[0.0] * 13 + [1.0],
    )
    output = tmp_path / "stable_pft_layout.nc"

    create_stomate_restart_skeleton_from_schema(
        output,
        _physical(_template()),
        pft_layout=layout,
    )

    with Dataset(output) as dataset:
        assert dataset.orchidee_jax_pft_catalog_id == "orchidee_man_paper_250919"
        assert dataset.orchidee_jax_pft_layout_id == "paper_250919_legacy14"
        assert json.loads(dataset.orchidee_jax_pft_ids_json)[-1] == "mangrove_pft14"
        assert json.loads(dataset.orchidee_jax_fortran_pft_ids_json) == list(range(1, 15))
        assert json.loads(dataset.orchidee_jax_mtc_ids_json)[-1] == 2


def test_day_end_packet_exports_directly_to_standalone_restart(tmp_path: Path) -> None:
    template = _template()
    base = _states(template)
    changed_biomass = np.asarray(base.entry_state.biomass) + 0.125
    changed_t2m_month = np.asarray(base.season_state.t2m_month) + 0.5
    changed_gpp = np.asarray(base.daily_state.gpp_daily) + 0.75
    daily = base.daily_state._replace(gpp_daily=changed_gpp)
    packet = {
        "biomass": changed_biomass,
        "t2m_month": changed_t2m_month,
        "gpp_daily": changed_gpp,
        "daily_accumulators": daily._asdict(),
        "height": np.ones_like(base.entry_state.age),
    }
    output = tmp_path / "packet_stomate_restart.nc"

    write_report, merge_report = write_stomate_restart_from_day_end_packet(
        str(output),
        packet,
        base_states=base,
        physical_state=_physical(template),
    )
    restored = _states(output)

    np.testing.assert_array_equal(restored.entry_state.biomass, changed_biomass)
    np.testing.assert_array_equal(restored.season_state.t2m_month, changed_t2m_month)
    np.testing.assert_array_equal(restored.daily_state.gpp_daily, changed_gpp)
    _assert_state_equal(base.gas_state, restored.gas_state)
    _assert_state_equal(base.remainder_state, restored.remainder_state)
    assert "height" in merge_report.packet_only_fields
    assert len(write_report.written_fields) == 161
