from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from jax_orchidee.driver.forcing_metadata import (
    ForcingMetadataError,
    forcing_info,
    forcing_vertical_ioipsl,
)


def _forcing_dataset(
    *,
    times: tuple[float, ...] = (0.0, 6.0),
    units: str = "hours since 1961-01-01 00:00:00",
    calendar: str | None = "noleap",
    vertical: dict[str, tuple[tuple[str, ...], np.ndarray]] | None = None,
    include_contfrac: bool = True,
) -> xr.Dataset:
    attrs = {"units": units}
    if calendar is not None:
        attrs["calendar"] = calendar
    lon = np.array([10.0, 20.0, 30.0])
    lat = np.array([5.0, -5.0])
    qair = np.full((len(times), lat.size, lon.size), 0.01)
    data: dict[str, object] = {
        "time_counter": (("record",), np.asarray(times), attrs),
        "Qair": (("record", "lat", "lon"), qair),
    }
    for index, name in enumerate(
        ("Tair", "PSurf", "Rainf", "Snowf", "SWdown", "LWdown", "Wind_N", "Wind_E")
    ):
        data[name] = (("record", "lat", "lon"), np.full_like(qair, index + 1.0))
    if include_contfrac:
        data["contfrac"] = (
            ("lat", "lon"),
            np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
        )
    if vertical is None:
        data.update({"Height_Lev1": ((), 2.0), "Height_Levuv": ((), 10.0)})
    else:
        data.update(vertical)
    return xr.Dataset(data, coords={"lon": lon, "lat": lat})


def test_forcing_info_preserves_file_and_fortran_order_and_routes_interpolation():
    result = forcing_info(
        _forcing_dataset(vertical={"lev": (("lev",), np.array([2.0]))}),
        date0=715875.0,
        limit_west=15.0,
        limit_east=30.0,
        limit_south=-10.0,
        limit_north=10.0,
    )

    assert (result.iim_full, result.jjm_full, result.llm_full, result.tm) == (3, 2, 1, 2)
    assert (result.iim, result.jjm, result.llm, result.nbpoint) == (2, 2, 1, 3)
    assert result.dt_force == 21600.0
    assert result.interpol and not result.daily_interpol and not result.weathergen
    np.testing.assert_array_equal(result.i_index_fortran, [2, 3])
    np.testing.assert_array_equal(result.j_index_fortran, [1, 2])
    # The zero-contfrac cell is first in the zoom; j-outer/i-inner leaves 2,3,4.
    np.testing.assert_array_equal(result.land_index_fortran, [2, 3, 4])
    qair_meta = result.variables["Qair"]
    assert qair_meta.file_dimensions == ("record", "lat", "lon")
    assert qair_meta.fortran_dimensions == ("lon", "lat", "record")
    assert qair_meta.file_shape == (2, 2, 3)
    assert qair_meta.fortran_shape == (3, 2, 2)
    assert result.vertical.zheight and result.vertical.zlev_fixed == 2.0


def test_absent_contfrac_takes_explicit_fortran_all_land_branch():
    result = forcing_info(_forcing_dataset(include_contfrac=False), date0=0.0)
    assert result.nbpoint == 6
    np.testing.assert_array_equal(result.land_index_fortran, np.arange(1, 7))


def test_daily_and_monthly_temporal_routes_and_missing_calendar_policy():
    daily = forcing_info(_forcing_dataset(times=(0.0, 1.0), units="days since 2001-01-01"), date0=0.0)
    assert daily.interpol and daily.daily_interpol
    assert daily.calendar == "noleap"

    missing_calendar = forcing_info(_forcing_dataset(calendar=None), date0=0.0)
    assert missing_calendar.calendar == "gregorian"

    monthly_source = _forcing_dataset(
        times=(0.0, 30.0), units="days since 2001-01-01"
    )
    with pytest.raises(ForcingMetadataError, match="ALLOW_WEATHERGEN"):
        forcing_info(monthly_source, date0=0.0)
    monthly = forcing_info(
        monthly_source,
        date0=0.0,
        allow_weathergen=True,
        dt_weathgen=3600.0,
        limit_west=0.0,
        limit_east=4.0,
        limit_south=0.0,
        limit_north=4.0,
        zonal_res=2.0,
        merid_res=2.0,
    )
    assert monthly.weathergen and not monthly.interpol
    assert monthly.calendar == "noleap"
    assert monthly.dt_force == 3600.0
    assert (monthly.iim, monthly.jjm, monthly.tm) == (2, 2, 8760)


@pytest.mark.parametrize("delta_hours", [7.0, 48.0])
def test_unsuitable_time_cadence_is_fatal(delta_hours: float):
    with pytest.raises(ForcingMetadataError, match="not suitable"):
        forcing_info(
            _forcing_dataset(times=(0.0, delta_hours)), date0=0.0
        )


def test_missing_qair_and_parallel_overcommit_are_not_silently_repaired():
    source = _forcing_dataset().drop_vars("Qair")
    with pytest.raises(ForcingMetadataError, match="missing.*Qair"):
        forcing_info(source, date0=0.0)
    with pytest.raises(ForcingMetadataError, match="less than mpi_size"):
        forcing_info(_forcing_dataset(), date0=0.0, mpi_size=6)


def test_forcing_info_leaves_reader_validation_downstream_and_uses_height_fallback():
    assert forcing_info(_forcing_dataset().drop_vars("Wind_E"), date0=0.0).nbpoint == 5
    source = _forcing_dataset().drop_vars(("Height_Lev1", "Height_Levuv"))
    fallback = forcing_info(source, date0=0.0)
    assert fallback.vertical.used_run_def
    assert (fallback.vertical.zlev_fixed, fallback.vertical.zlevuv_fixed) == (2.0, 10.0)


def test_vertical_priority_sigma_and_separate_uv_levels():
    source = _forcing_dataset(
        vertical={
            "Sigma": ((), np.array(0.91)),
            "Sigma_uv": ((), np.array(0.82)),
            "HybSigA": ((), np.array(12.0)),
            "HybSigB": ((), np.array(0.5)),
        }
    )
    result = forcing_vertical_ioipsl(source)
    assert result.zsigma and not result.zhybrid
    assert not result.zsamelev_uv
    assert (result.zhybrid_a, result.zhybrid_b) == (0.0, 0.91)
    assert (result.zhybriduv_a, result.zhybriduv_b) == (0.0, 0.82)


def test_vertical_hybrid_pairs_are_mandatory_and_values_are_read():
    missing_tq_b = _forcing_dataset(
        vertical={"HybSigA": ((), np.array(10.0))}
    )
    with pytest.raises(ForcingMetadataError, match="T and Q"):
        forcing_vertical_ioipsl(missing_tq_b)

    missing_uv_b = _forcing_dataset(
        vertical={
            "HybSigA": ((), np.array(10.0)),
            "HybSigB": ((), np.array(0.8)),
            "HybSigA_uv": ((), np.array(20.0)),
        }
    )
    with pytest.raises(ForcingMetadataError, match="U and V"):
        forcing_vertical_ioipsl(missing_uv_b)

    source = missing_uv_b.assign(HybSigB_uv=xr.DataArray(0.7))
    result = forcing_vertical_ioipsl(source)
    assert result.zhybrid and not result.zsamelev_uv
    assert (result.zhybrid_a, result.zhybrid_b) == (10.0, 0.8)
    assert (result.zhybriduv_a, result.zhybriduv_b) == (20.0, 0.7)


def test_levels_height_legacy_and_run_def_fallback_are_distinct():
    levels = forcing_vertical_ioipsl(
        _forcing_dataset(
            vertical={
                "Levels": (("lat", "lon"), np.ones((2, 3))),
                "Levels_uv": (("lat", "lon"), np.ones((2, 3)) * 2),
            }
        )
    )
    assert levels.zlevels and not levels.zsamelev_uv
    assert levels.zlev_fixed is None

    height = forcing_vertical_ioipsl(
        _forcing_dataset(
            vertical={
                "Height_Lev1": ((), np.array(3.0)),
                "Height_Levuv": ((), np.array(12.0)),
            }
        )
    )
    assert height.zheight and (height.zlev_fixed, height.zlevuv_fixed) == (3.0, 12.0)

    legacy = forcing_vertical_ioipsl(
        _forcing_dataset(vertical={"lev": (("lev",), np.array([4.0]))})
    )
    assert legacy.source_variables == ("lev",) and legacy.zsamelev_uv

    fallback = forcing_vertical_ioipsl(
        None, force_id=-1, height_lev1=5.0, height_levw=5.0
    )
    assert fallback.used_run_def and fallback.zsamelev_uv
    assert (fallback.zlev_fixed, fallback.zlevuv_fixed) == (5.0, 5.0)

    source_defaults = forcing_vertical_ioipsl(None, force_id=-1)
    assert (source_defaults.zlev_fixed, source_defaults.zlevuv_fixed) == (2.0, 10.0)

    zero_id = forcing_vertical_ioipsl(
        _forcing_dataset(vertical={"Sigma": ((), np.array(0.9))}), force_id=0
    )
    assert zero_id.used_run_def and not zero_id.zsigma


def test_time_metadata_and_domain_errors_are_explicit():
    with pytest.raises(ForcingMetadataError, match="at least two"):
        forcing_info(_forcing_dataset(times=(0.0,)), date0=0.0)
    with pytest.raises(ForcingMetadataError, match="time units"):
        forcing_info(_forcing_dataset(units="fortnights"), date0=0.0)
    with pytest.raises(ForcingMetadataError, match="invalid south/north"):
        forcing_info(
            _forcing_dataset(), date0=0.0, limit_south=10.0, limit_north=10.0
        )


def test_temp_netcdf_path_executes_source_landpoint_discovery():
    source = _forcing_dataset()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "forcing.nc"
        source.to_netcdf(path)
        result = forcing_info(path, date0=0.0)
    assert result.filename == str(path)
    np.testing.assert_array_equal(result.land_index_fortran, [1, 3, 4, 5, 6])


def test_variable_and_uv_level_counts_must_match_metadata():
    source = _forcing_dataset(vertical={})
    for name in ("Tair", "Qair"):
        source[name] = source[name].expand_dims(lev=[0, 1], axis=1)
    for name in ("Wind_N", "Wind_E"):
        source[name] = source[name].expand_dims(lev_uv=[0, 1], axis=1)
    source["Sigma"] = (("lev",), [0.9, 0.7])
    source["Sigma_uv"] = (("lev_uv",), [0.8, 0.6])
    result = forcing_info(source, date0=0.0)
    assert result.vertical.thermodynamic_level_count == 2
    assert result.vertical.wind_level_count == 2

    assert result.vertical.thermodynamic_values.shape == (2,)
    assert result.vertical.wind_values.shape == (2,)


def test_lowercase_watchout_levels_do_not_preempt_ioipsl_vertical_priority():
    source = _forcing_dataset(vertical={})
    source["levels"] = (("record", "lat", "lon"), np.full((2, 2, 3), 30.0))
    for index, name in enumerate(
        ("SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy")
    ):
        source[name] = (("record", "lat", "lon"), np.full((2, 2, 3), index + 1.0))
    result = forcing_info(source, date0=0.0)
    assert not result.is_watchout
    assert result.vertical.used_run_def


def test_source_owner_ledger_has_exact_batch4_arm_partition():
    root = Path(__file__).resolve().parents[2]
    document = yaml.safe_load(
        (root / "docs/source_audits/pft14_source_owners_forcing_metadata.yaml").read_text(
            encoding="utf-8"
        )
    )
    entries = document["entries"]
    assert document["counts"] == {
        "forcing_info_arms": 35,
        "forcing_vertical_ioipsl_arms": 58,
        "total_arms": 93,
    }
    assert len(entries) == 2
    assert len({entry["id"] for entry in entries}) == 2
    assert {entry["fortran_subroutine"] for entry in entries} == {
        "forcing_info",
        "forcing_vertical_ioipsl",
    }
    assert all(entry["status"] == "implemented" for entry in entries)
