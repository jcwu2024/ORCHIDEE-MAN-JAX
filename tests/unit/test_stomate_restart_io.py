from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.reference import (  # noqa: E402
    read_stomate_daily_accumulator_state,
    read_stomate_ok_pc_restart_gas_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
)
from jax_orchidee.stomate.restart_io import (  # noqa: E402
    STOMATE_FIXED_PATH_CARRY_FIELDS,
    read_stomate_readstart_states_from_template,
    stomate_fixed_path_carry_state,
    stomate_writerestart_field_ledger,
    template_backed_restart_field_names,
    write_stomate_daily_accumulator_state_from_template,
    write_stomate_full_writerestart_states_from_template,
    write_stomate_restart_entry_state_from_template,
    write_stomate_restart_season_state_from_template,
    write_stomate_restart_states_from_template,
    writerestart_index_labels,
)


def _minimal_restart(path: Path, *, nvm: int = 2, nslm: int = 2) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("pft", nvm)
        dataset.createDimension("layer", nslm)
        dataset.createDimension("y", 1)
        dataset.createDimension("x", 1)


def test_readstart_empty_template_uses_exact_existing_contract_defaults(tmp_path: Path) -> None:
    restart = tmp_path / "empty.nc"
    _minimal_restart(restart)

    result = read_stomate_readstart_states_from_template(
        restart,
        t2m=np.array([281.5]),
        nvm=2,
        nslm=2,
        ndeep=3,
        val_exp=999999.0,
        sla=np.array([0.01, 0.02]),
        o2_init_conc=0.21,
        ch4_init_conc=0.0018,
        nvert=6,
        ns=4,
        months_num=3,
        nbpools=3,
    )

    np.testing.assert_array_equal(result.daily_state.tsurf_daily, [281.5])
    np.testing.assert_array_equal(result.daily_state.t2m_min_daily, [1.0e33])
    np.testing.assert_array_equal(result.season_state.tsoil_month, [[281.5, 281.5]])
    np.testing.assert_array_equal(result.season_state.gdd_init_date, [[365.0, 999999.0]])
    np.testing.assert_array_equal(result.entry_state.sla_calc, [[0.01, 0.02]])
    np.testing.assert_array_equal(result.gas_state.O2_soil, np.full((1, 3, 2), 0.21))
    np.testing.assert_array_equal(result.gas_state.CH4_snow, np.full((1, 3, 2), 0.0018))
    assert result.season_state.tau_longterm == 2.0
    assert "gdd_init_date" in result.report.defaulted_fields
    assert "carbon_32l" in result.report.defaulted_fields
    assert "DOC" in result.report.defaulted_fields
    np.testing.assert_array_equal(result.remainder_state.fireindex, np.zeros((1, 2)))
    np.testing.assert_array_equal(
        result.remainder_state.uo_0,
        [[500.0, 500.0, 500.0, 500.0, 0.0017, 0.0017]],
    )
    np.testing.assert_array_equal(result.remainder_state.uo_wet1, np.full((1, 6), 0.0017))
    np.testing.assert_array_equal(result.remainder_state.wet1day, [6.0])
    np.testing.assert_array_equal(result.remainder_state.previous_stock, np.full((1, 2, 3), 1.0e20))
    np.testing.assert_array_equal(result.remainder_state.MatrixV[0, 0], np.eye(3))
    assert result.remainder_state.read_input_thawed_humidity is True
    assert result.remainder_state.read_input_depth_organic_soil is True
    assert result.remainder_state.depth_deepsoil is None
    assert result.remainder_state.Global_years == 0
    assert isinstance(result.remainder_state.Global_years, int)
    assert result.report.source_undefined_fields == ("depth_deepsoil",)
    assert result.report.implemented is True
    assert result.report.uncovered_reachable_fields == ()
    assert result.report.out_of_span_fields == ()
    assert set(result.report.provenance_by_field) == set(result.entry_state._fields) | {
        field for field in result.season_state._fields if field != "provenance"
    } | {field for field in result.daily_state._fields if field != "provenance"} | {
        field for field in result.gas_state._fields if field != "provenance"
    } | {field for field in result.remainder_state._fields if field != "provenance"}


def test_readstart_partial_restart_preserves_all_semantics_and_bool_threshold(tmp_path: Path) -> None:
    restart = tmp_path / "partial.nc"
    _minimal_restart(restart)
    val_exp = 999999.0
    with Dataset(restart, "r+") as dataset:
        pft_dims = ("time", "pft", "y", "x")
        grid_dims = ("time", "y", "x")
        dataset.createVariable("PFTpresent", "f8", pft_dims)[:] = [[[[0.49]], [[0.5]]]]
        dataset.createVariable("npp_daily", "f8", pft_dims)[:] = [[[[val_exp]], [[7.0]]]]
        dataset.createVariable("t2m_longterm", "f8", grid_dims)[:] = [[[279.0]]]
        dataset.createVariable("thawed_humidity", "f8", grid_dims)[:] = [[[0.75]]]

    result = read_stomate_readstart_states_from_template(
        restart,
        t2m=np.array([281.5]),
        nvm=2,
        nslm=2,
        ndeep=3,
        val_exp=val_exp,
        tau_longterm_max=42.0,
        thawed_humidity_input=0.25,
        reset_thawed_humidity=True,
    )

    np.testing.assert_array_equal(result.entry_state.pft_present, [[False, True]])
    np.testing.assert_array_equal(result.entry_state.npp_daily, [[val_exp, 7.0]])
    np.testing.assert_array_equal(result.season_state.t2m_longterm, [279.0])
    np.testing.assert_array_equal(result.entry_state.thawed_humidity, [0.25])
    assert result.season_state.tau_longterm == 42.0
    assert "pft_present" in result.report.restart_fields
    assert "npp_daily" in result.report.mixed_sentinel_fields
    assert "tau_longterm" in result.report.defaulted_fields
    assert set(result.report.source_undefined_fields) == {
        "depth_deepsoil",
        "read_input_thawed_humidity",
    }


def test_readstart_real_fortran_template_matches_strict_physical_readers() -> None:
    template = _real_template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    daily = read_stomate_daily_accumulator_state(template)
    gas = read_stomate_ok_pc_restart_gas_state(template)

    result = read_stomate_readstart_states_from_template(
        template,
        t2m=np.full(entry.age.shape[0], 280.0),
        nvm=entry.age.shape[1],
        nslm=season.tsoil_month.shape[1],
        ndeep=entry.carbon_32l.shape[3],
    )

    _assert_state_equal(entry, result.entry_state)
    _assert_state_equal(season, result.season_state)
    _assert_state_equal(daily, result.daily_state)
    _assert_state_equal(gas, result.gas_state)
    assert set(result.report.defaulted_fields) == {
        "snowfall_daily",
        "snowmass_daily",
        "tmc_topgrass_daily",
    }
    assert result.report.mixed_sentinel_fields == ()
    assert set(result.report.source_undefined_fields) == {
        "read_input_depth_organic_soil",
        "read_input_thawed_humidity",
    }


def test_readstart_remainder_normalizes_axes_and_preserves_mixed_sentinel(tmp_path: Path) -> None:
    restart = tmp_path / "remainder.nc"
    _minimal_restart(restart, nvm=2)
    val_exp = 999999.0
    with Dataset(restart, "r+") as dataset:
        dataset.createDimension("deep", 3)
        dataset.createDimension("vertical", 4)
        dataset.createDimension("pool", 2)
        dataset.createVariable("uo_0", "f8", ("time", "vertical", "y", "x"))[:] = [
            [[[val_exp]], [[2.0]], [[3.0]], [[4.0]]]
        ]
        dataset.createVariable(
            "deepC_peat", "f8", ("time", "pft", "deep", "y", "x")
        )[:] = np.arange(6, dtype=np.float64).reshape(1, 2, 3, 1, 1)
        dataset.createVariable(
            "MatrixV", "f8", ("time", "pool", "pool", "pft", "y", "x")
        )[:] = np.arange(8, dtype=np.float64).reshape(1, 2, 2, 2, 1, 1)
        dataset.createVariable("ok_equilibrium", "f8", ("time", "y", "x"))[:] = [[[0.5]]]
        dataset.createVariable("depth_deepsoil", "f8", ("time", "pft", "y", "x"))[:] = val_exp

    result = read_stomate_readstart_states_from_template(
        restart,
        t2m=np.array([280.0]),
        nvm=2,
        nslm=2,
        ndeep=3,
        nvert=4,
        ns=2,
        months_num=2,
        nbpools=2,
        val_exp=val_exp,
    )

    np.testing.assert_array_equal(result.remainder_state.uo_0, [[val_exp, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(
        result.remainder_state.deepC_peat[0],
        [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
    )
    np.testing.assert_array_equal(
        result.remainder_state.MatrixV[0],
        np.arange(8, dtype=np.float64).reshape(2, 2, 2).transpose(2, 1, 0),
    )
    np.testing.assert_array_equal(result.remainder_state.ok_equilibrium, [True])
    np.testing.assert_array_equal(result.remainder_state.depth_deepsoil, np.zeros((1, 2)))
    assert "uo_0" in result.report.mixed_sentinel_fields
    assert "deepC_peat" in result.report.restart_fields
    assert "depth_deepsoil" in result.report.defaulted_fields


def _real_template() -> Path:
    case_root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    templates = sorted(case_root.rglob("stomate_start.nc"))
    if not templates:
        pytest.skip("no real stomate_start.nc exists under reference/OUT")
    return templates[0]


def _assert_state_equal(expected: object, actual: object) -> None:
    for field in expected._fields:
        if field == "provenance":
            continue
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if np.isscalar(expected_value):
            assert actual_value == expected_value, field
        else:
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=field)


def test_template_backed_entry_state_roundtrip_is_exact(tmp_path: Path) -> None:
    template = _real_template()
    original = read_stomate_restart_entry_state(template)
    output = tmp_path / "stomate_restart_roundtrip.nc"

    report = write_stomate_restart_entry_state_from_template(template, output, original)
    restored = read_stomate_restart_entry_state(output)

    assert output.is_file()
    assert set(report.written_fields) | set(report.validated_derived_fields) == set(original._fields)
    assert report.validated_derived_fields == ("prod100_total", "prod10_total", "soilc_total")
    assert report.unsupported_fields == ()
    _assert_state_equal(original, restored)


def test_template_writer_rejects_inconsistent_derived_state(tmp_path: Path) -> None:
    template = _real_template()
    state = read_stomate_restart_entry_state(template)
    invalid = state._replace(prod10_total=np.asarray(state.prod10_total) + 1.0)

    with pytest.raises(ValueError, match="prod10_total cannot be written independently"):
        write_stomate_restart_entry_state_from_template(
            template,
            tmp_path / "invalid.nc",
            invalid,
        )


def test_template_writer_persists_changed_direct_and_packed_fields(tmp_path: Path) -> None:
    template = _real_template()
    state = read_stomate_restart_entry_state(template)
    carbon_32l = np.asarray(state.carbon_32l).copy()
    doc = np.asarray(state.DOC).copy()
    carbon_32l[0, :, 13, 0] = [1.25, 2.5, 3.75]
    doc[0, 13, 0, :, 0, 0] = [4.5, 5.5]
    changed = state._replace(
        age=np.asarray(state.age) + 7.0,
        need_adjacent=np.logical_not(state.need_adjacent),
        carbon_32l=carbon_32l,
        DOC=doc,
    )
    output = tmp_path / "changed.nc"

    write_stomate_restart_entry_state_from_template(template, output, changed)
    restored = read_stomate_restart_entry_state(output)

    np.testing.assert_array_equal(restored.age, changed.age)
    np.testing.assert_array_equal(restored.need_adjacent, changed.need_adjacent)
    np.testing.assert_array_equal(restored.carbon_32l, changed.carbon_32l)
    np.testing.assert_array_equal(restored.DOC, changed.DOC)


def test_combined_template_roundtrip_covers_all_restart_reader_fields(tmp_path: Path) -> None:
    template = _real_template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    daily = read_stomate_daily_accumulator_state(template)
    output = tmp_path / "all_reader_states.nc"

    report = write_stomate_restart_states_from_template(
        template,
        output,
        entry_state=entry,
        season_state=season,
        daily_state=daily,
    )

    _assert_state_equal(entry, read_stomate_restart_entry_state(output))
    _assert_state_equal(season, read_stomate_restart_season_state(output))
    _assert_state_equal(daily, read_stomate_daily_accumulator_state(output))
    assert len(template_backed_restart_field_names()) == 113
    assert set(report.written_fields) | set(report.validated_derived_fields) == set(
        template_backed_restart_field_names()
    )
    assert len(report.written_fields) == 107
    assert len(report.validated_derived_fields) == 6
    assert report.unsupported_fields == ()


def test_season_and_daily_writers_persist_changes(tmp_path: Path) -> None:
    template = _real_template()
    season = read_stomate_restart_season_state(template)
    season_changed = season._replace(
        date=season.date + 3,
        begin_leaves=np.logical_not(season.begin_leaves),
        tsoil_month=np.asarray(season.tsoil_month) + 2.0,
    )
    season_output = tmp_path / "season.nc"
    season_report = write_stomate_restart_season_state_from_template(
        template,
        season_output,
        season_changed,
    )
    _assert_state_equal(season_changed, read_stomate_restart_season_state(season_output))
    assert len(season_report.written_fields) == 38

    daily = read_stomate_daily_accumulator_state(template)
    daily_changed = daily._replace(
        humrel_daily=np.asarray(daily.humrel_daily) + 0.125,
        t2m_daily=np.asarray(daily.t2m_daily) + 1.0,
        soilhum_daily=np.asarray(daily.soilhum_daily) + 0.25,
    )
    daily_output = tmp_path / "daily.nc"
    daily_report = write_stomate_daily_accumulator_state_from_template(
        template,
        daily_output,
        daily_changed,
    )
    _assert_state_equal(daily_changed, read_stomate_daily_accumulator_state(daily_output))
    assert len(daily_report.written_fields) == 13
    assert set(daily_report.validated_derived_fields) == {
        "snowfall_daily",
        "snowmass_daily",
        "tmc_topgrass_daily",
    }


def test_combined_writer_rejects_conflicting_shared_gpp(tmp_path: Path) -> None:
    template = _real_template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    daily = read_stomate_daily_accumulator_state(template)
    conflicting = daily._replace(gpp_daily=np.asarray(daily.gpp_daily) + 1.0)

    with pytest.raises(ValueError, match="gpp_daily differs"):
        write_stomate_restart_states_from_template(
            template,
            tmp_path / "conflict.nc",
            entry_state=entry,
            season_state=season,
            daily_state=conflicting,
        )


def test_writerestart_index_labels_follow_fortran_one_based_branch_order() -> None:
    labels = writerestart_index_labels(nlitt=2, nlevs=2, nelements=1)

    assert labels.litter == ("met", "str")
    assert labels.level == ("ab", "be")
    assert labels.element == ("",)
    assert labels.pools == (
        "str_ab",
        "str_be",
        "met_ab",
        "met_be",
        "actif ",
        "slow  ",
        "passif",
    )

    swapped = writerestart_index_labels(
        nlitt=2,
        nlevs=2,
        nelements=1,
        imetabolic=2,
        istructural=1,
        iabove=2,
        ibelow=1,
    )
    assert swapped.litter == ("str", "met")
    assert swapped.level == ("be", "ab")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nlitt": 3, "nlevs": 2, "nelements": 1}, "litter_str"),
        ({"nlitt": 2, "nlevs": 3, "nelements": 1}, "level_str"),
        ({"nlitt": 2, "nlevs": 2, "nelements": 2}, "element_str"),
    ],
)
def test_writerestart_index_labels_reject_unmapped_source_indices(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        writerestart_index_labels(**kwargs)


def test_writerestart_field_ledger_separates_state_from_physical_io() -> None:
    ledger = stomate_writerestart_field_ledger()

    assert len(ledger.serialized_fields) == 161
    assert len(ledger.validated_not_serialized_fields) == 6
    assert len(ledger.normalized_state_fields) == 167
    assert ledger.source_local_not_restart_fields == (
        "read_input_depth_organic_soil",
        "read_input_thawed_humidity",
    )
    assert ledger.independent_netcdf_boundaries == ()
    assert "O2_soil" in ledger.serialized_fields
    assert "MatrixV" in ledger.serialized_fields
    assert "soilc_total" in ledger.validated_not_serialized_fields


def test_full_writerestart_roundtrip_covers_gas_and_remainder_state(tmp_path: Path) -> None:
    template = _real_template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    with Dataset(template) as dataset:
        nvert = int(dataset.variables["uo_0"].shape[1])
        months_num = int(dataset.variables["fwet_series"].shape[1])
        nbpools = int(dataset.variables["MatrixV"].shape[1])
        nsnow = int(dataset.variables["O2_snow"].shape[2])

    original = read_stomate_readstart_states_from_template(
        template,
        t2m=np.asarray(read_stomate_daily_accumulator_state(template).t2m_daily),
        nvm=entry.age.shape[1],
        nslm=season.tsoil_month.shape[1],
        ndeep=entry.carbon_32l.shape[3],
        nsnow=nsnow,
        nvert=nvert,
        months_num=months_num,
        ncarb=entry.carbon.shape[1],
        nlitt=entry.litter.shape[1],
        nbpools=nbpools,
    )
    output = tmp_path / "full_writerestart.nc"

    report = write_stomate_full_writerestart_states_from_template(
        template,
        output,
        entry_state=original.entry_state,
        season_state=original.season_state,
        daily_state=original.daily_state,
        gas_state=original.gas_state,
        remainder_state=original.remainder_state,
    )
    restored = read_stomate_readstart_states_from_template(
        output,
        t2m=np.asarray(original.daily_state.t2m_daily),
        nvm=entry.age.shape[1],
        nslm=season.tsoil_month.shape[1],
        ndeep=entry.carbon_32l.shape[3],
        nsnow=nsnow,
        nvert=nvert,
        months_num=months_num,
        ncarb=entry.carbon.shape[1],
        nlitt=entry.litter.shape[1],
        nbpools=nbpools,
    )

    _assert_state_equal(original.entry_state, restored.entry_state)
    _assert_state_equal(original.season_state, restored.season_state)
    _assert_state_equal(original.daily_state, restored.daily_state)
    _assert_state_equal(original.gas_state, restored.gas_state)
    for field in original.remainder_state._fields:
        if field == "provenance" or field.startswith("read_input_"):
            continue
        expected = getattr(original.remainder_state, field)
        actual = getattr(restored.remainder_state, field)
        if np.isscalar(expected):
            assert actual == expected, field
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=field)
    assert len(report.written_fields) == 161
    assert len(report.validated_derived_fields) == 6
    assert report.unsupported_fields == ()


def test_fixed_paper_path_carries_only_source_proven_inactive_state() -> None:
    template = _real_template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    with Dataset(template) as dataset:
        nvert = int(dataset.variables["uo_0"].shape[1])
        months_num = int(dataset.variables["fwet_series"].shape[1])
        nbpools = int(dataset.variables["MatrixV"].shape[1])
        nsnow = int(dataset.variables["O2_snow"].shape[2])
    states = read_stomate_readstart_states_from_template(
        template,
        t2m=read_stomate_daily_accumulator_state(template).t2m_daily,
        nvm=entry.age.shape[1],
        nslm=season.tsoil_month.shape[1],
        ndeep=entry.carbon_32l.shape[3],
        nsnow=nsnow,
        nvert=nvert,
        months_num=months_num,
        ncarb=entry.carbon.shape[1],
        nlitt=entry.litter.shape[1],
        nbpools=nbpools,
    )
    carry = stomate_fixed_path_carry_state(
        entry_state=states.entry_state,
        season_state=states.season_state,
        gas_state=states.gas_state,
        remainder_state=states.remainder_state,
        fire_disable=True,
        ok_pc=False,
        ch4_calcul=False,
        ok_leak=True,
        ok_peat=False,
        peat_occur=False,
        dyn_peat=False,
        land_cover_change=False,
        enable_grazing=False,
        spinup_analytic=False,
        stomate_ok_dgvm=False,
        lpj_gap_const_mort=True,
    )

    assert tuple(carry.fields) == STOMATE_FIXED_PATH_CARRY_FIELDS
    assert len(carry.fields) == 65
    assert carry.fields["date"] == states.season_state.date
    assert carry.fields["O2_soil"] is states.gas_state.O2_soil
    assert carry.fields["MatrixV"] is states.remainder_state.MatrixV
    assert set(carry.provenance_by_field) == set(carry.fields)


def test_fixed_paper_path_carry_rejects_reachable_update_owner() -> None:
    template = _real_template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    with Dataset(template) as dataset:
        states = read_stomate_readstart_states_from_template(
            template,
            t2m=read_stomate_daily_accumulator_state(template).t2m_daily,
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

    with pytest.raises(ValueError, match="ok_pc"):
        stomate_fixed_path_carry_state(
            entry_state=states.entry_state,
            season_state=states.season_state,
            gas_state=states.gas_state,
            remainder_state=states.remainder_state,
            fire_disable=True,
            ok_pc=True,
            ch4_calcul=False,
            ok_leak=True,
            ok_peat=False,
            peat_occur=False,
            dyn_peat=False,
            land_cover_change=False,
            enable_grazing=False,
            spinup_analytic=False,
            stomate_ok_dgvm=False,
            lpj_gap_const_mort=True,
        )
