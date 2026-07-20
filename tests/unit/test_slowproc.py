from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.slowproc import (  # noqa: E402
    read_slowproc_restart_entry_state,
    slowproc_cold_start_vegetation_entry_state,
    slowproc_dynamic_peat_fraction_step,
    slowproc_dyn_peat_disabled_entry_state,
    slowproc_derivvar_explicit,
    slowproc_erosion_daily_zero_entry_state,
    slowproc_fire_disabled_entry_state,
    slowproc_lai_explicit,
    slowproc_no_lcc_entry_state,
    slowproc_static_entry_state,
    slowproc_stomate_nobio_boundary,
    slowproc_surface_update_explicit,
    slowproc_thermosoil_entry_init_state,
    slowproc_tot_bare_soil,
    slowproc_totfrac_nobio,
    slowproc_veget_explicit,
)

REFERENCE_RUN = (
    ROOT
    / "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/"
    / "001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658"
)


def _expected_slowproc_veget(
    *,
    lai,
    frac_nobio,
    veget_max,
    pref_soil_veg,
    ext_coeff_vegetfrac,
    nstm,
    min_vegfrac=1.0e-6,
    min_sechiba=1.0e-8,
):
    frac_nobio = np.asarray(frac_nobio, dtype=np.float64).copy()
    veget_max = np.asarray(veget_max, dtype=np.float64).copy()
    lai = np.asarray(lai, dtype=np.float64)
    npts, nvm = veget_max.shape
    for ji in range(npts):
        if np.sum(frac_nobio[ji, :]) < min_vegfrac:
            frac_nobio[ji, :] = 0.0
        veget_max[ji, veget_max[ji, :] < min_vegfrac] = 0.0
        total = np.sum(frac_nobio[ji, :]) + np.sum(veget_max[ji, :])
        frac_nobio[ji, :] = frac_nobio[ji, :] / total
        veget_max[ji, :] = veget_max[ji, :] / total

    veget = np.zeros_like(veget_max)
    veget[:, 0] = veget_max[:, 0]
    for jv in range(1, nvm):
        veget[:, jv] = veget_max[:, jv] * (1.0 - np.exp(-lai[:, jv] * ext_coeff_vegetfrac[jv]))

    totfrac_nobio = np.sum(frac_nobio, axis=1)
    soiltile = np.zeros((npts, nstm), dtype=np.float64)
    for jv, jst_fortran in enumerate(pref_soil_veg):
        soiltile[:, int(jst_fortran) - 1] += veget_max[:, jv]
    for ji in range(npts):
        if totfrac_nobio[ji] < (1.0 - min_sechiba):
            soiltile[ji, :] = soiltile[ji, :] / (1.0 - totfrac_nobio[ji])
    return frac_nobio, veget_max, veget, totfrac_nobio, soiltile


def test_slowproc_totfrac_nobio_sums_explicit_non_bio_fractions():
    frac_nobio = np.asarray([[0.10, 0.20], [0.0, 0.05]], dtype=np.float64)

    total = slowproc_totfrac_nobio(frac_nobio)

    np.testing.assert_allclose(np.asarray(total), [0.30, 0.05])


def test_read_slowproc_restart_entry_state_normalizes_sechiba_restart_axes():
    state = read_slowproc_restart_entry_state(REFERENCE_RUN / "sechiba_start.nc")

    assert state.lai.shape == (1, 14)
    assert state.height.shape == (1, 14)
    assert state.frac_age.shape == (1, 14, 4)
    assert state.veget.shape == (1, 14)
    assert state.veget_max.shape == (1, 14)
    assert state.frac_nobio.shape == (1, 1)
    assert np.isfinite(state.frac_age).all()
    assert np.any(state.frac_age != 0.0)
    assert any("slowproc_init lines 1685-1707" in item for item in state.provenance)


def test_slowproc_derivvar_explicit_matches_fortran_initialization():
    veget = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float64)
    lai = np.asarray([[5.0, 2.0, 0.5]], dtype=np.float64)

    result = slowproc_derivvar_explicit(
        veget=veget,
        lai=lai,
        vcmax_fix=np.asarray([0.0, 40.0, 55.0], dtype=np.float64),
        height_presc=np.asarray([0.0, 30.0, 1.0], dtype=np.float64),
        qsintcst=0.1,
    )

    np.testing.assert_allclose(np.asarray(result.qsintmax), [[0.0, 0.06, 0.02]])
    np.testing.assert_allclose(np.asarray(result.deadleaf_cover), [0.0])
    np.testing.assert_allclose(np.asarray(result.assim_param[:, :, 0]), [[0.0, 40.0, 55.0]])
    np.testing.assert_allclose(np.asarray(result.height), [[0.0, 30.0, 1.0]])
    np.testing.assert_allclose(np.asarray(result.temp_growth), [25.0])


def test_slowproc_no_lcc_entry_state_matches_paper_case_zero_lcc_branch():
    veget_max = np.asarray([[0.1, 0.9, 0.0]], dtype=np.float64)

    result = slowproc_no_lcc_entry_state(
        veget_max=veget_max,
        use_age_class=False,
        veget_update="0Y",
    )

    np.testing.assert_allclose(np.asarray(result.veget_max_new), veget_max)
    np.testing.assert_allclose(np.asarray(result.vegetnew_firstday), [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(result.totfrac_nobio_new), [0.0])
    np.testing.assert_allclose(np.asarray(result.glccNetLCC), np.zeros((1, 12)))
    np.testing.assert_allclose(np.asarray(result.harvest_matrix), np.zeros((1, 12)))
    np.testing.assert_allclose(np.asarray(result.bound_spa), np.zeros((1, 3)))
    assert any("effective veget_update=0" in item for item in result.provenance)


def test_slowproc_no_lcc_entry_state_treats_imposed_vegetation_as_effective_zero_update():
    veget_max = np.zeros((1, 14), dtype=np.float64)
    veget_max[0, 13] = 1.0

    result = slowproc_no_lcc_entry_state(
        veget_max=veget_max,
        use_age_class=False,
        veget_update="1Y",
        map_pft_format=True,
        impose_veg=True,
    )

    np.testing.assert_allclose(np.asarray(result.veget_max_new), veget_max)
    np.testing.assert_allclose(np.asarray(result.glccNetLCC), np.zeros((1, 12)))


def test_slowproc_no_lcc_entry_state_rejects_active_map_update_without_lcc_state():
    veget_max = np.zeros((1, 14), dtype=np.float64)

    with pytest.raises(ValueError, match="nonzero vegetation update"):
        slowproc_no_lcc_entry_state(
            veget_max=veget_max,
            use_age_class=False,
            veget_update="1Y",
            map_pft_format=True,
            impose_veg=False,
        )


def test_slowproc_fire_disabled_entry_state_matches_initialized_noop_branch():
    result = slowproc_fire_disabled_entry_state(kjpindex=2, fire_disable=True)

    for name in (
        "lightn",
        "popd",
        "observed_ba",
        "humign",
        "cf_fine",
        "cf_coarse",
        "ratio_flag",
        "ratio",
    ):
        np.testing.assert_allclose(np.asarray(getattr(result, name)), np.zeros((2,)))
    assert result.read_observed_ba is False
    assert result.read_cf_fine is False
    assert result.read_cf_coarse is False
    assert result.read_ratio_flag is False
    assert result.read_ratio is False
    assert any("FIRE_DISABLE=y" in item for item in result.provenance)

    with pytest.raises(ValueError, match="fire-enabled"):
        slowproc_fire_disabled_entry_state(kjpindex=1, fire_disable=False)


def test_slowproc_dyn_peat_disabled_entry_state_is_branch_inactive_dummy():
    result = slowproc_dyn_peat_disabled_entry_state(kjpindex=3, dyn_peat=False)

    np.testing.assert_array_equal(np.asarray(result.sat_duration), np.zeros((3,), dtype=np.int32))
    assert "branch-inactive dummy" in result.notes[0]
    assert any("DYN_PEAT=n" in item for item in result.provenance)

    with pytest.raises(ValueError, match="dynamic peat"):
        slowproc_dyn_peat_disabled_entry_state(kjpindex=1, dyn_peat=True)


def _dyn_peat_common(**overrides):
    npts, nvm = 3, 4
    args = {
        "veget_max": np.asarray(
            [
                [0.2, 0.0, 0.0, 0.3],
                [0.2, 0.0, 0.4, 0.3],
                [0.2, 0.0, 0.4, 0.3],
            ],
            dtype=np.float64,
        ),
        "veget_max_new": np.asarray(
            [
                [0.2, 0.1, 0.0, 0.3],
                [0.2, 0.1, 0.4, 0.3],
                [0.2, 0.1, 0.4, 0.3],
            ],
            dtype=np.float64,
        ),
        "fpeat": np.asarray([0.3, 0.6, 1.0e-9], dtype=np.float64),
        "is_peat": np.asarray([False, False, True, False]),
        "month": 6,
        "day": 1,
        "sec": 1800.0,
        "dt_sechiba": 1800.0,
        "dyn_peat": True,
        "dynpeat_PWT": False,
        "dynpeat_PC": False,
        "peatC_ok": np.zeros(npts, dtype=np.float64),
        "temp_growth": np.asarray([6.0, 4.0, 10.0], dtype=np.float64),
        "growth_day": np.asarray([5.0, 5.0, 5.0], dtype=np.float64),
        "precipitation_thissummer": np.zeros(npts, dtype=np.float64),
        "precipitation_lastsummer": np.asarray([100.0, 100.0, 100.0], dtype=np.float64),
        "peatPET_thisyear": np.zeros(npts, dtype=np.float64),
        "peatPET_lastyear": np.asarray([50.0, 50.0, 50.0], dtype=np.float64),
        "precip_rain": np.zeros(npts, dtype=np.float64),
        "precip_snow": np.zeros(npts, dtype=np.float64),
        "peat_PET": np.zeros(npts, dtype=np.float64),
        "GSL": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
    }
    args.update(overrides)
    return args


def test_slowproc_dynamic_peat_fraction_pwt_off_updates_peat_target_only():
    """Fortran: slowproc.f90::slowproc_main lines 850-918, dynpeat_PWT false branch."""

    result = slowproc_dynamic_peat_fraction_step(**_dyn_peat_common(min_stomate=1.0e-8))

    expected = _dyn_peat_common()["veget_max_new"].copy()
    expected[:, 2] = [0.3, 0.6, 0.0]
    np.testing.assert_allclose(np.asarray(result.veget_max_new), expected)
    assert result.update_peatfrac is True
    np.testing.assert_allclose(np.asarray(result.peatC_ok), np.zeros(3))
    assert any("slowproc_main lines 850-918" in item for item in result.provenance)


def test_slowproc_dynamic_peat_fraction_pwt_water_and_carbon_gates_expansion():
    """Fortran: slowproc.f90::slowproc_main lines 865-889."""

    result = slowproc_dynamic_peat_fraction_step(
        **_dyn_peat_common(
            dynpeat_PWT=True,
            dynpeat_PC=True,
            peatC_ok=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            precipitation_lastsummer=np.asarray([100.0, 20.0, 100.0], dtype=np.float64),
            peatPET_lastyear=np.asarray([50.0, 50.0, 50.0], dtype=np.float64),
            PWT_lim=0.0,
            fpeat=np.asarray([0.3, 0.6, 0.6], dtype=np.float64),
        )
    )

    expected = _dyn_peat_common()["veget_max_new"].copy()
    expected[:, 2] = [0.3, 0.4, 0.6]
    np.testing.assert_allclose(np.asarray(result.veget_max_new), expected)


def test_slowproc_dynamic_peat_fraction_updates_peatc_ok_on_january_second():
    """Fortran: slowproc.f90::slowproc_main lines 578-594."""

    result = slowproc_dynamic_peat_fraction_step(
        **_dyn_peat_common(
            month=1,
            day=2,
            dynpeat_PC=True,
            judge_pc=True,
            peatC=np.asarray([9.0, 10.0, 11.0], dtype=np.float64),
            PC_lim=10.0,
            fpeat=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        )
    )

    np.testing.assert_allclose(np.asarray(result.peatC_ok), [0.0, 1.0, 1.0])


def test_slowproc_thermosoil_entry_init_state_matches_source_backed_initial_values_only():
    znt = np.asarray([0.0005, 0.002, 0.01], dtype=np.float64)
    zlt = np.asarray([0.001, 0.005, 0.02], dtype=np.float64)

    result = slowproc_thermosoil_entry_init_state(kjpindex=2, nvm=4, znt=znt, zlt=zlt)

    assert result.tdeep.shape == (2, 3, 4)
    assert result.hsdeep.shape == (2, 3, 4)
    assert result.heat_Zimov.shape == (2, 3, 4)
    np.testing.assert_allclose(np.asarray(result.tdeep), np.full((2, 3, 4), 250.0))
    np.testing.assert_allclose(np.asarray(result.hsdeep), np.ones((2, 3, 4)))
    np.testing.assert_allclose(np.asarray(result.heat_Zimov), np.zeros((2, 3, 4)))
    np.testing.assert_allclose(np.asarray(result.zz_deep), znt)
    np.testing.assert_allclose(np.asarray(result.zz_coef_deep), zlt)
    assert any("sechiba_init lines 2704-2732" in item for item in result.provenance)
    assert "soilc_total" in result.notes[0]


def test_slowproc_thermosoil_entry_init_state_validates_vertical_vectors():
    with pytest.raises(ValueError, match="share shape"):
        slowproc_thermosoil_entry_init_state(
            kjpindex=1,
            nvm=1,
            znt=np.asarray([0.1, 0.2], dtype=np.float64),
            zlt=np.asarray([0.1], dtype=np.float64),
        )


def test_slowproc_static_entry_state_builds_fc_grazing_and_humcste_use_for_all_pfts():
    result = slowproc_static_entry_state(
        njsc=np.asarray([1, 6, 12], dtype=np.int32),
        soil_classif="usda",
        pft_to_mtc=np.asarray([1, 2, 3, 14], dtype=np.int32),
        zmaxh=2.0,
    )

    np.testing.assert_allclose(np.asarray(result.fc_grazing), [0.43, 0.43, 0.38])
    np.testing.assert_allclose(
        np.asarray(result.humcste_use),
        np.asarray(
            [
                [5.0, 0.8, 0.8, 4.0],
                [5.0, 0.8, 0.8, 4.0],
                [5.0, 0.8, 0.8, 4.0],
            ],
            dtype=np.float64,
        ),
    )
    assert any("slowproc_soilt lines 2190-2216" in item for item in result.provenance)
    assert any("hydrol_var_init lines 4093-4107" in item for item in result.provenance)


def test_slowproc_static_entry_state_rejects_unsupported_or_out_of_range_soil_classes():
    with pytest.raises(ValueError, match="outside the selected soil table"):
        slowproc_static_entry_state(
            njsc=np.asarray([13], dtype=np.int32),
            soil_classif="usda",
            pft_to_mtc=np.asarray([1], dtype=np.int32),
            zmaxh=2.0,
        )
    with pytest.raises(ValueError, match="soil_classif"):
        slowproc_static_entry_state(
            njsc=np.asarray([1], dtype=np.int32),
            soil_classif="mystery",
            pft_to_mtc=np.asarray([1], dtype=np.int32),
            zmaxh=2.0,
        )


def test_slowproc_erosion_daily_zero_entry_state_closes_deposition_but_not_erodepth():
    result = slowproc_erosion_daily_zero_entry_state(kjpindex=2, ncarb=3)

    np.testing.assert_allclose(np.asarray(result.sed_deposition_d), np.zeros((2,)))
    np.testing.assert_allclose(np.asarray(result.poc_deposition_d), np.zeros((2, 3)))
    assert any("sechiba_init lines 2461-2467" in item for item in result.provenance)
    assert "erodepth is not included" in result.notes[0]


def test_slowproc_stomate_nobio_boundary_matches_lcchange_branches():
    frac_last = np.asarray([[0.10, 0.05], [0.20, 0.10]], dtype=np.float64)
    frac_new = np.asarray([[0.30, 0.10], [0.0, 0.25]], dtype=np.float64)

    inactive = slowproc_stomate_nobio_boundary(
        frac_nobio_lastyear=frac_last,
        do_now_stomate_lcchange=False,
        use_age_class=False,
    )
    new_from_map = slowproc_stomate_nobio_boundary(
        frac_nobio_lastyear=frac_last,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        frac_nobio_new=frac_new,
    )
    new_from_age_class = slowproc_stomate_nobio_boundary(
        frac_nobio_lastyear=frac_last,
        do_now_stomate_lcchange=True,
        use_age_class=True,
    )

    np.testing.assert_allclose(np.asarray(inactive.totfrac_nobio_lastyear), [0.15, 0.30])
    np.testing.assert_allclose(np.asarray(inactive.totfrac_nobio_new), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(new_from_map.totfrac_nobio_new), [0.40, 0.25])
    np.testing.assert_allclose(
        np.asarray(new_from_age_class.totfrac_nobio_new),
        np.asarray(new_from_age_class.totfrac_nobio_lastyear),
    )


def test_slowproc_stomate_nobio_boundary_requires_new_fraction_for_active_map_branch():
    with pytest.raises(ValueError, match="frac_nobio_new"):
        slowproc_stomate_nobio_boundary(
            frac_nobio_lastyear=np.asarray([[0.1]], dtype=np.float64),
            do_now_stomate_lcchange=True,
            use_age_class=False,
        )


def test_slowproc_veget_explicit_matches_normalization_lai_and_soiltile_source_loops():
    lai = np.asarray([[0.0, 2.0, 0.5], [0.0, 0.0, 3.0]], dtype=np.float64)
    frac_nobio = np.asarray([[1.0e-7], [0.20]], dtype=np.float64)
    veget_max = np.asarray([[0.10, 0.30, 0.60], [1.0e-8, 0.40, 0.30]], dtype=np.float64)
    pref_soil_veg = np.asarray([1, 2, 4], dtype=np.int32)
    ext_coeff = np.asarray([0.0, 0.5, 0.3], dtype=np.float64)

    result = slowproc_veget_explicit(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff,
        nstm=4,
    )
    expected = _expected_slowproc_veget(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff,
        nstm=4,
    )

    np.testing.assert_allclose(np.asarray(result.frac_nobio), expected[0])
    np.testing.assert_allclose(np.asarray(result.veget_max), expected[1])
    np.testing.assert_allclose(np.asarray(result.veget), expected[2])
    np.testing.assert_allclose(np.asarray(result.totfrac_nobio), expected[3])
    np.testing.assert_allclose(np.asarray(result.soiltile), expected[4])
    assert any("slowproc_veget lines 2820-2925" in item for item in result.provenance)


def test_slowproc_cold_start_vegetation_entry_state_sets_lai_frac_age_height_and_surface():
    result = slowproc_cold_start_vegetation_entry_state(
        veget_max=np.asarray([[0.10, 0.30, 0.60]], dtype=np.float64),
        frac_nobio=np.asarray([[0.0]], dtype=np.float64),
        pref_soil_veg=np.asarray([1, 2, 3], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5, 0.3], dtype=np.float64),
        height_presc=np.asarray([0.0, 30.0, 1.0], dtype=np.float64),
        nstm=3,
        nleafages=4,
    )

    np.testing.assert_allclose(np.asarray(result.lai), np.zeros((1, 3)))
    np.testing.assert_allclose(np.asarray(result.height), [[0.0, 30.0, 1.0]])
    np.testing.assert_allclose(np.asarray(result.frac_age[:, :, 0]), np.ones((1, 3)))
    np.testing.assert_allclose(np.asarray(result.frac_age[:, :, 1:]), np.zeros((1, 3, 3)))
    np.testing.assert_allclose(np.asarray(result.vegetation.veget), [[0.10, 0.30, 0.60]])
    np.testing.assert_allclose(np.asarray(result.tot_bare_soil), [0.10])
    assert any("slowproc_init lines 2153-2170" in item for item in result.provenance)


def test_slowproc_lai_explicit_read_lai_matches_fortran_month_interpolation():
    laimap = np.zeros((1, 3, 12), dtype=np.float64)
    laimap[0, 1, :] = np.arange(1.0, 13.0)
    laimap[0, 2, :] = np.arange(10.0, 22.0)

    jan = slowproc_lai_explicit(
        read_lai=True,
        type_of_lai=("mean", "inter", "mean"),
        month=1,
        day=10,
        laimap=laimap,
    )
    np.testing.assert_allclose(np.asarray(jan.lai), [[0.0, 12.0 * (5.0 / 30.0) + 1.0 * (25.0 / 30.0), 21.0]])

    mid = slowproc_lai_explicit(
        read_lai=True,
        type_of_lai=("mean", "inter", "mean"),
        month=6,
        day=20,
        laimap=laimap,
    )
    np.testing.assert_allclose(np.asarray(mid.lai), [[0.0, 6.0 * (25.0 / 30.0) + 7.0 * (5.0 / 30.0), 21.0]])

    dec = slowproc_lai_explicit(
        read_lai=True,
        type_of_lai=("mean", "inter", "mean"),
        month=12,
        day=20,
        laimap=laimap,
    )
    np.testing.assert_allclose(np.asarray(dec.lai), [[0.0, 12.0 * (25.0 / 30.0) + 1.0 * (5.0 / 30.0), 21.0]])
    assert any("slowproc_lai lines 2949-3062" in item for item in jan.provenance)


def test_slowproc_lai_explicit_non_read_lai_requires_tempfunc_for_inter_branch():
    result = slowproc_lai_explicit(
        read_lai=False,
        type_of_lai=("mean", "mean", "inter"),
        month=7,
        day=12,
        npts=2,
        llaimax=np.asarray([0.0, 8.0, 6.0], dtype=np.float64),
        llaimin=np.asarray([0.0, 2.0, 1.0], dtype=np.float64),
        tempfunc_values=np.asarray([0.25, 0.75], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.lai), [[0.0, 5.0, 2.25], [0.0, 5.0, 4.75]])
    with pytest.raises(ValueError, match="tempfunc_values"):
        slowproc_lai_explicit(
            read_lai=False,
            type_of_lai=("mean", "inter"),
            month=7,
            day=12,
            npts=1,
            llaimax=np.asarray([0.0, 8.0], dtype=np.float64),
            llaimin=np.asarray([0.0, 2.0], dtype=np.float64),
        )


def test_slowproc_cold_start_read_lai_uses_explicit_laimap_for_surface_state():
    laimap = np.zeros((1, 3, 12), dtype=np.float64)
    laimap[0, 1, :] = 2.0
    laimap[0, 2, :] = 1.0

    result = slowproc_cold_start_vegetation_entry_state(
        veget_max=np.asarray([[0.10, 0.30, 0.60]], dtype=np.float64),
        frac_nobio=np.asarray([[0.0]], dtype=np.float64),
        pref_soil_veg=np.asarray([1, 2, 3], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5, 0.3], dtype=np.float64),
        height_presc=np.asarray([0.0, 30.0, 1.0], dtype=np.float64),
        nstm=3,
        read_lai=True,
        laimap=laimap,
        type_of_lai=("mean", "mean", "inter"),
        month=3,
        day=15,
    )

    expected_lai = np.asarray([[0.0, 2.0, 1.0]], dtype=np.float64)
    expected = _expected_slowproc_veget(
        lai=expected_lai,
        frac_nobio=np.asarray([[0.0]], dtype=np.float64),
        veget_max=np.asarray([[0.10, 0.30, 0.60]], dtype=np.float64),
        pref_soil_veg=np.asarray([1, 2, 3], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5, 0.3], dtype=np.float64),
        nstm=3,
    )

    np.testing.assert_allclose(np.asarray(result.lai), expected_lai)
    np.testing.assert_allclose(np.asarray(result.vegetation.veget), expected[2])
    np.testing.assert_allclose(np.asarray(result.tot_bare_soil), [0.10 + 0.30 - expected[2][0, 1] + 0.60 - expected[2][0, 2]])


def test_slowproc_veget_explicit_keeps_small_veget_max_when_dgvm_is_active():
    lai = np.asarray([[0.0, 1.0]], dtype=np.float64)
    frac_nobio = np.asarray([[0.0]], dtype=np.float64)
    veget_max = np.asarray([[1.0e-8, 1.0]], dtype=np.float64)

    dgvm = slowproc_veget_explicit(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=np.asarray([1, 2], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5], dtype=np.float64),
        nstm=2,
        ok_dgvm=True,
    )
    static = slowproc_veget_explicit(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=np.asarray([1, 2], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5], dtype=np.float64),
        nstm=2,
        ok_dgvm=False,
    )

    assert np.asarray(dgvm.veget_max)[0, 0] > 0.0
    assert np.asarray(static.veget_max)[0, 0] == pytest.approx(0.0, abs=0.0)


def test_slowproc_tot_bare_soil_matches_main_loop_and_is_distinct_from_totfrac_nobio():
    veget_max = np.asarray([[0.20, 0.50, 0.30]], dtype=np.float64)
    veget = np.asarray([[0.20, 0.30, 0.10]], dtype=np.float64)

    tot_bare = slowproc_tot_bare_soil(veget_max=veget_max, veget=veget)

    np.testing.assert_allclose(np.asarray(tot_bare), [0.60])


def test_slowproc_surface_update_explicit_returns_post_stomate_surface_boundary():
    result = slowproc_surface_update_explicit(
        lai=np.asarray([[0.0, 2.0, 0.5]], dtype=np.float64),
        frac_nobio=np.asarray([[0.10]], dtype=np.float64),
        veget_max=np.asarray([[0.10, 0.30, 0.50]], dtype=np.float64),
        pref_soil_veg=np.asarray([1, 2, 4], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5, 0.3], dtype=np.float64),
        nstm=4,
    )
    expected_bare = slowproc_tot_bare_soil(
        veget_max=result.vegetation.veget_max,
        veget=result.vegetation.veget,
    )

    np.testing.assert_allclose(np.asarray(result.tot_bare_soil), np.asarray(expected_bare))
    assert result.vegetation.veget.shape == (1, 3)
    assert result.vegetation.soiltile.shape == (1, 4)
    assert any("slowproc_main lines 1103-1121" in item for item in result.provenance)
