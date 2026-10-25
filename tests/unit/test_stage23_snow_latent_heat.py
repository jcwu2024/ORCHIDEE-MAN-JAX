"""Independent source-constant regressions for Stage23 S22-SNOW-LF.

Expected values use constantes_var.f90:374-376 and the cited source equations,
never an implementation constant or a saved333600-parameter oracle.
"""
import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol as snow


SOURCE_LF = 2.8345e6 - 2.5008e6
TP = 273.15
XCI = 2106.0
RTOL = 1e-12
ATOL = 1e-14


@pytest.fixture
def cold_start_snow_inputs():
    """Source initialize defaults followed by one step with1kg snowfall."""
    z = lambda: np.zeros(1, dtype=np.float64)
    layers = lambda value: np.full((1, 3), value, dtype=np.float64)
    return dict(
        precip_rain=z(), precip_snow=np.ones(1), temp_air=np.array([TP]),
        pb=np.array([1013.]), u=z(), v=z(), temp_sol_new=np.array([TP]),
        soilcap=np.array([2.e6]), pgflux=z(), frac_nobio=np.zeros((1, 1)),
        totfrac_nobio=z(), gtemp=np.array([TP]), lambda_snow=z(),
        cgrnd_snow=np.array([[TP, TP, 0.]]), dgrnd_snow=layers(0.),
        vevapsno=z(), snow_age=z(), snow_nobio_age=np.zeros((1, 1)),
        snow_nobio=np.zeros((1, 1)), snowrho=layers(50.), snowgrain=layers(0.),
        snowdz=layers(0.), snowtemp=layers(TP), snowheat=layers(0.), snow=z(),
        temp_sol_add=z(), snowliq=layers(0.), subsnownobio=np.zeros((1, 1)),
        grndflux=z(), snowmelt=z(), soilflxresid=z(),
    )


def test_real_hydrol_snow_adapter_one_kg_new_snow_source_enthalpy(cold_start_snow_inputs):
    # explicitsnow1209-1210/400-402; qsat_moisture823-824 at T=TP, no liquid.
    result = snow.hydrol_explicit_snow_step(**cold_start_snow_inputs)
    np.testing.assert_allclose(result.snow_state.snow, [1.], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snow_state.tot_melt, [0.], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.sum(result.snow_state.snowheat, axis=1), [-SOURCE_LF], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "narrow_jit"])
def test_enthalpy_and_heat_diagnosis_use_source_fusion_constant(compiled):
    rho = np.full((1, 3), 100.)
    depth = np.full((1, 3), .1)
    temperature = np.array([[TP - 2., TP, TP]])
    liquid = np.array([[0., .0001, 0.]])
    # qsat_moisture823-824,966-970,1196-1198; independently supplied source heat.
    source_heat = rho * depth * (XCI * (temperature - TP) - SOURCE_LF) + SOURCE_LF * 1000. * liquid
    def component(q, r, z, t, heat):
        return (snow.snow3lheat_explicit(q, r, z, t),
                snow.snow3ltemp_explicit(heat, r, z),
                snow.snow3lliq_explicit(heat, r, z, t))
    function = jax.jit(component) if compiled else component
    heat, diagnosed_temp, diagnosed_liq = function(liquid, rho, depth, temperature, source_heat)
    np.testing.assert_allclose(heat, source_heat, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(diagnosed_temp, temperature, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(diagnosed_liq, liquid, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("phase", ["partial_melt", "partial_refreeze"])
def test_partial_phase_change_matches_independent_source_energy(phase):
    # explicitsnow1459-1509. All layers stay below holding capacity; no flow.
    warm = phase == "partial_melt"
    rho = np.full((1, 3), 100.)
    depth = np.full((1, 3), .1)
    temperature = np.full((1, 3), TP + (1. if warm else -1.))
    liquid = np.full((1, 3), 0. if warm else .0001)
    mass_change = 100. * .1 * XCI / SOURCE_LF
    result = snow.explicitsnow_melt_refrz_step(
        precip_rain=np.zeros(1), pgflux=np.zeros(1), soilcap=np.array([2.e6]),
        snowtemp=temperature, snowdz=depth, snowrho=rho, snowliq=liquid,
        snowmelt=np.zeros(1), grndflux=np.zeros(1), temp_air=np.array([TP]),
        soilflxresid=np.zeros(1),
    )
    expected_liquid = liquid + (mass_change if warm else -mass_change) / 1000.
    expected_depth = depth * (1. - mass_change / 10.) if warm else depth
    expected_rho = 10. / expected_depth
    expected_melt = np.full((1, 3), mass_change if warm else 0.)
    expected_refreeze = np.full((1, 3), 0. if warm else mass_change)
    np.testing.assert_allclose(result.melt_mass_by_layer, expected_melt, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.refreeze_mass_by_layer, expected_refreeze, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snowliq, expected_liquid, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snowdz, expected_depth, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snowrho, expected_rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snowtemp, TP, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.liquid_percolation_by_interface, 0., rtol=RTOL, atol=ATOL)


def test_liquid_excess_ground_flux_uses_source_latent_energy():
    # explicitsnow388-391: exactly1kg excess liquid in first layer.
    rho, depth = np.full((1, 3), 100.), np.full((1, 3), .1)
    result = snow.explicitsnow_liquid_heat_excess_step(
        snowliq=np.array([[.002, .0005, 0.]]), snowdz=depth, snowrho=rho,
        grndflux=np.array([7.]),
    )
    source_flux = np.array([[SOURCE_LF / 1800., 0., 0.]])
    np.testing.assert_allclose(result.zliqheatxs, source_flux, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.grndflux, [7. + SOURCE_LF / 1800.], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snowliq, [[.001, .0005, 0.]], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "narrow_jit"])
@pytest.mark.parametrize("bottom,cap", [(10., 1.), (13., 3.)], ids=["moderate", "extreme"])
def test_maxmass_default_uses_source_regular_and_extreme_caps(bottom, cap, compiled):
    # explicitsnow329-337; mass3165 or4065, cap uses333700J/m2/K soilcap.
    rho, depth = np.full((1, 3), 300.), np.array([[.05, .5, bottom]])
    def component(c, r, z):
        result = snow.explicitsnow_maxmass_step(soilcap=c, snowrho=r, snowdz=z)
        return result.snowmelt_from_maxmass, result.snowdz
    function = jax.jit(component) if compiled else component
    removal, updated_depth = function(np.array([SOURCE_LF]), rho, depth)
    expected_depth = depth.copy()
    expected_depth[:, 2] -= cap / 300.
    np.testing.assert_allclose(removal, [cap], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(updated_depth, expected_depth, rtol=RTOL, atol=ATOL)


def test_nonbio_melt_default_uses_source_fusion_constant():
    # explicitsnow426: limited neither by available snow nor maxmass.
    result = snow.explicitsnow_age_ice_step(
        precip_snow=np.zeros(1), precip_rain=np.zeros(1), temp_sol_new_old=np.array([TP + 1.]),
        temp_sol_new=np.array([TP]), soilcap=np.array([SOURCE_LF]), frac_nobio=np.array([[.5]]),
        subsnownobio=np.zeros((1, 1)), snow=np.array([10.]), snow_age=np.zeros(1),
        snow_nobio=np.array([[10.]]), snow_nobio_age=np.zeros((1, 1)),
    )
    np.testing.assert_allclose(result.snowmelt_ice, [.5], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snow_nobio, [[9.5]], rtol=RTOL, atol=ATOL)


def test_bucket_default_uses_same_source_chalfu_not_soil_lhf():
    # hydrol_snow4811 uses chalfu0 (constantes_var376), unlike soil lhf569.
    result = snow.hydrol_bucket_snow_step(
        precip_rain=np.zeros(1), precip_snow=np.zeros(1), temp_sol_new=np.array([TP + 1.]),
        soilcap=np.array([SOURCE_LF]), frac_nobio=np.zeros((1, 1)), totfrac_nobio=np.zeros(1),
        vevapsno=np.zeros(1), snow=np.array([10.]), snow_age=np.zeros(1),
        snow_nobio=np.zeros((1, 1)), snow_nobio_age=np.zeros((1, 1)),
    )
    np.testing.assert_allclose(result.snowmelt, [1.], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.snow, [9.], rtol=RTOL, atol=ATOL)
