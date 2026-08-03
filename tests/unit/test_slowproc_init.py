import numpy as np
import pytest

from jax_orchidee.sechiba.slowproc import (
    VAL_EXP,
    SlowprocInitGapError,
    slowproc_init_pft14_explicit,
)


PREF_SOIL_VEG = np.asarray([1] * 13 + [4], dtype=np.int32)
EXT_COEFF = np.asarray([0.0] + [0.5] * 13, dtype=np.float64)
HEIGHT_PRESC = np.asarray(
    [0.0, 30.0, 30.0, 20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 0.5, 0.6, 1.0, 1.0, 30.0],
    dtype=np.float64,
)
PFT14_ONLY = np.asarray([0.0] * 13 + [1.0], dtype=np.float64)


def _complete_restart():
    veget = np.zeros((1, 14), dtype=np.float64)
    veget[0, 13] = 0.97318184
    veget_max = np.zeros((1, 14), dtype=np.float64)
    veget_max[0, 13] = 1.0
    lai = np.zeros((1, 14), dtype=np.float64)
    lai[0, 13] = 3.6186761
    frac_age = np.zeros((1, 14, 4), dtype=np.float64)
    frac_age[0, 13, :] = [0.23069987, 0.256, 0.257, 0.25630013]
    return {
        "veget": veget,
        "veget_max": veget_max,
        "frac_nobio": np.asarray([[0.0]]),
        "lai": lai,
        "height": HEIGHT_PRESC[None, :],
        "frac_age": frac_age,
        "njsc": np.asarray([2], dtype=np.int32),
        "clay_frac": np.asarray([0.2]),
        "sand_frac": np.asarray([0.4]),
        "bulk_dens": np.asarray([1650.0]),
        "soil_ph": np.asarray([7.0]),
        "poor_soils": np.asarray([0.0]),
        "reinf_slope": np.asarray([0.0]),
        "growth_day": np.asarray([17.0]),
    }


def _call(restart=None, **overrides):
    args = {
        "restart": _complete_restart() if restart is None else restart,
        "veget_max_default": PFT14_ONLY,
        "frac_nobio_default": 0.0,
        "height_presc": HEIGHT_PRESC,
        "pref_soil_veg": PREF_SOIL_VEG,
        "ext_coeff_vegetfrac": EXT_COEFF,
        "nstm": 6,
        "diaglev": np.asarray([0.1, 0.3, 1.0]),
        "active_pft_index": 13,
        "salinity_data": np.asarray([31.25]),
        "tide_height_data": np.asarray([[0.1, -0.2, 0.3]]),
    }
    args.update(overrides)
    return slowproc_init_pft14_explicit(**args)


def test_pft14_restart_path_preserves_primary_state_and_recomputes_writebacks():
    restart = _complete_restart()
    result = _call(restart)

    assert result.found_restart is True
    assert result.veget_update == 0
    assert result.lcanop == 3
    assert result.qsintcst == pytest.approx(0.1)
    assert result.dt_stomate == pytest.approx(86400.0)
    assert result.PWT_lim == pytest.approx(60.0)
    assert result.PC_lim == pytest.approx(50.0)
    assert result.sat_gsl == pytest.approx(1.0)
    np.testing.assert_array_equal(np.asarray(result.veget), restart["veget"])
    np.testing.assert_array_equal(np.asarray(result.veget_max), restart["veget_max"])
    np.testing.assert_array_equal(np.asarray(result.lai), restart["lai"])
    np.testing.assert_array_equal(np.asarray(result.frac_age), restart["frac_age"])
    np.testing.assert_allclose(np.asarray(result.totfrac_nobio), [0.0])
    np.testing.assert_allclose(np.asarray(result.soiltile), [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(result.tot_bare_soil), [1.0 - 0.97318184])
    np.testing.assert_allclose(np.asarray(result.siltfraction), [0.4])
    np.testing.assert_allclose(np.asarray(result.fc_grazing), [0.41])
    np.testing.assert_allclose(np.asarray(result.salinity), [31.25])
    np.testing.assert_allclose(np.asarray(result.tide_height), [[0.1, -0.2, 0.3]])
    np.testing.assert_allclose(np.asarray(result.peat.growth_day), [17.0])
    np.testing.assert_allclose(np.asarray(result.peat.GSL), [1.0])
    np.testing.assert_allclose(np.asarray(result.no_lcc.veget_max_new), restart["veget_max"])
    assert result.fire.read_ratio is False
    assert any("lines 2133-2277" in item for item in result.provenance)


def test_pft14_all_missing_restart_uses_exact_imposed_defaults_and_soil_boundary():
    soil_boundary = {
        "soilclass": np.asarray([[0.0, 0.8, 0.2] + [0.0] * 9]),
        "clayfraction": np.asarray([0.2]),
        "sandfraction": np.asarray([0.4]),
        "siltfraction": np.asarray([0.4]),
        "bulk_density": np.asarray([1650.0]),
        "soil_ph": np.asarray([7.0]),
        "poor_soils": np.asarray([0.0]),
    }

    result = _call(restart={}, soil_boundary=soil_boundary)

    assert result.found_restart is False
    np.testing.assert_allclose(np.asarray(result.veget_max), PFT14_ONLY[None, :])
    # slowproc_veget sees val_exp before lines 2156-2157 reset LAI, so PFT14 is fully covered.
    np.testing.assert_allclose(np.asarray(result.veget), PFT14_ONLY[None, :])
    np.testing.assert_allclose(np.asarray(result.lai), np.zeros((1, 14)))
    np.testing.assert_allclose(np.asarray(result.frac_age[:, :, 0]), np.ones((1, 14)))
    np.testing.assert_allclose(np.asarray(result.frac_age[:, :, 1:]), np.zeros((1, 14, 3)))
    np.testing.assert_allclose(np.asarray(result.height), HEIGHT_PRESC[None, :])
    np.testing.assert_array_equal(np.asarray(result.njsc), [2])
    np.testing.assert_allclose(np.asarray(result.reinf_slope), [0.0])
    np.testing.assert_allclose(np.asarray(result.peat.peatPET_lastyear), [1.0e20])
    np.testing.assert_allclose(np.asarray(result.peat.precipitation_lastsummer), [100.0])
    np.testing.assert_allclose(np.asarray(result.peat.precipitation_thissummer), [0.0])
    np.testing.assert_allclose(np.asarray(result.peat.GSL), [1.0])
    np.testing.assert_allclose(np.asarray(result.tot_bare_soil), [0.0])


def test_pft14_cold_start_vectorizes_over_independent_landpoints():
    veget_max = np.zeros((2, 14), dtype=np.float64)
    veget_max[:, 13] = [1.0, 0.8]
    soilclass = np.zeros((2, 12), dtype=np.float64)
    soilclass[0, 1] = 1.0
    soilclass[1, 4] = 1.0
    soil_boundary = {
        "soilclass": soilclass,
        "clay_frac": np.asarray([0.2, 0.1]),
        "sand_frac": np.asarray([0.4, 0.06]),
        "silt_frac": np.asarray([0.4, 0.84]),
        "bulk_dens": np.asarray([1650.0, 1385.0]),
        "soil_ph": np.asarray([7.0, 5.7]),
        "poor_soils": np.asarray([0.0, 0.01]),
    }

    result = _call(
        restart={},
        veget_max_default=veget_max,
        frac_nobio_default=np.asarray([0.0, 0.2]),
        soil_boundary=soil_boundary,
        salinity_data=np.asarray([31.25, 18.0]),
        tide_height_data=np.asarray([[0.1, -0.2, 0.3], [0.2, 0.0, -0.1]]),
    )

    assert result.found_restart is False
    assert result.veget.shape == (2, 14)
    assert result.frac_age.shape == (2, 14, 4)
    np.testing.assert_allclose(np.asarray(result.totfrac_nobio), [0.0, 0.2])
    np.testing.assert_allclose(np.asarray(result.soiltile[:, 3]), [1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(result.njsc), [2, 5])
    np.testing.assert_allclose(np.asarray(result.siltfraction), [0.4, 0.84])
    np.testing.assert_allclose(np.asarray(result.salinity), [31.25, 18.0])
    assert result.fire.ratio.shape[0] == 2


def test_pft14_state_contract_keeps_nvm_extensible():
    veget_max = np.zeros((1, 16), dtype=np.float64)
    veget_max[0, 13] = 1.0
    soil_boundary = {
        "soilclass": np.eye(12, dtype=np.float64)[[1]],
        "clay_frac": np.asarray([0.2]),
        "sand_frac": np.asarray([0.4]),
        "silt_frac": np.asarray([0.4]),
        "bulk_dens": np.asarray([1650.0]),
        "soil_ph": np.asarray([7.0]),
        "poor_soils": np.asarray([0.0]),
    }

    result = _call(
        restart={},
        veget_max_default=veget_max,
        height_presc=np.concatenate([HEIGHT_PRESC, [1.0, 1.0]]),
        pref_soil_veg=np.concatenate([PREF_SOIL_VEG, [1, 1]]),
        ext_coeff_vegetfrac=np.concatenate([EXT_COEFF, [0.5, 0.5]]),
        soil_boundary=soil_boundary,
    )

    assert result.veget.shape == (1, 16)
    assert result.frac_age.shape == (1, 16, 4)
    np.testing.assert_allclose(np.asarray(result.veget[0, [13]]), [1.0])
    np.testing.assert_allclose(np.asarray(result.veget[0, [14, 15]]), [0.0, 0.0])


def test_pft14_active_netcdf_reads_are_required_explicit_boundaries():
    with pytest.raises(SlowprocInitGapError, match="active salinity NetCDF"):
        _call(salinity_data=None)
    with pytest.raises(SlowprocInitGapError, match="active tide_height NetCDF"):
        _call(tide_height_data=None)


def test_pft14_missing_soil_requires_exact_slowproc_soilt_boundary():
    restart = _complete_restart()
    restart["soil_ph"] = np.asarray([VAL_EXP])

    with pytest.raises(SlowprocInitGapError, match="triggers slowproc_soilt at lines 2174-2193"):
        _call(restart)

    restart = _complete_restart()
    restart["njsc"] = np.asarray([13], dtype=np.int32)
    with pytest.raises(ValueError, match="USDA soil-class table"):
        _call(restart)


def test_pft14_mixed_vegetation_restart_and_unowned_branches_report_gaps():
    restart = _complete_restart()
    restart["veget"] = np.full((1, 14), VAL_EXP)
    with pytest.raises(SlowprocInitGapError, match="mixed present/missing"):
        _call(restart)

    with pytest.raises(SlowprocInitGapError, match="FIRE_DISABLE=n"):
        _call(fire_disable=False)
    with pytest.raises(SlowprocInitGapError, match="READ_LAI=y"):
        _call(read_lai=True)
    with pytest.raises(SlowprocInitGapError, match="prescribed vegetation is not PFT14-only"):
        bad_default = PFT14_ONLY.copy()
        bad_default[1] = 0.1
        _call(veget_max_default=bad_default)
