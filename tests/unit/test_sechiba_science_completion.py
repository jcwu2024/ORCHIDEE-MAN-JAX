import numpy as np
import pytest

from jax_orchidee.driver.interpolation_core12 import (
    AggregatePacket,
    InterpolationSource,
    InterpolationTarget,
)
from jax_orchidee.sechiba.condveg import (
    condveg_background_soilalb_source_routed,
    condveg_finalize_restart_packet,
)
from jax_orchidee.sechiba.diffuco import (
    diffuco_finalize_restart_packet,
    diffuco_main_source_routed,
    diffuco_trans_source_routed,
)
from jax_orchidee.sechiba.enerbil import enerbil_flux_source_routed
from jax_orchidee.sechiba.slowproc import (
    get_soilcorr_usda_source_routed,
    slowproc_change_frac_source_routed,
    slowproc_checkveget_source_routed,
    slowproc_finalize_restart_packet,
    slowproc_initialize_source_routed,
    slowproc_main_history_routing,
)


def test_diffuco_trans_routes_storage_activity_and_bare_pft_defaults():
    result = diffuco_trans_source_routed(
        swnet=[100.0],
        temp_air=[280.0],
        pb=[101325.0],
        qair=[0.005],
        rau=[0.0],
        u=[1.0],
        v=[0.0],
        q_cdrag=[0.1],
        humrel=[[1.0, 1.0]],
        veget=[[0.2, 0.5]],
        veget_max=[[0.2, 0.5]],
        lai=[[0.0, 2.0]],
        qsintveg=[[0.0, 0.2]],
        qsintmax=[[0.0, 1.0]],
        rstruct=[[0.0, 0.0]],
        vbeta23=[[0.0, 0.1]],
        kzero=[0.0, 2.0],
        rveg_pft=[1.0, 1.0],
    )

    expected_rveget = 11.5
    denominator = 1.0 + 0.1 * expected_rveget
    expected_beta = 0.5 * 0.8 / denominator + min(0.1, 0.5 * 0.2 / denominator)
    np.testing.assert_allclose(result.rveget[0, 1], expected_rveget)
    np.testing.assert_allclose(result.vbeta3[0, 1], expected_beta)
    np.testing.assert_allclose(result.vbeta3pot[0, 1], 0.5 / (1.0 + 0.1 * 5.75))
    assert result.vbeta3[0, 0] == 0.0
    assert result.rveget[0, 0] == 1.0e20
    np.testing.assert_array_equal(result.cimean, np.zeros((1, 2)))


def test_diffuco_finalize_selects_leaf_ci_only_on_co2_arm():
    base = dict(rstruct=np.ones((2, 3)), q_cdrag_pft=np.full((2, 3), 0.2))
    assert set(diffuco_finalize_restart_packet(**base, ok_co2=False)) == {
        "rstruct",
        "cdrag_pft",
    }
    packet = diffuco_finalize_restart_packet(
        **base, ok_co2=True, leaf_ci=np.ones((2, 3, 4))
    )
    assert set(packet) == {"rstruct", "cdrag_pft", "leaf_ci"}
    with pytest.raises(ValueError, match="leaf_ci"):
        diffuco_finalize_restart_packet(**base, ok_co2=True)


def test_diffuco_main_routes_coupled_drag_into_legacy_transpiration():
    trans = dict(
        swnet=[100.0],
        temp_air=[280.0],
        pb=[101325.0],
        qair=[0.005],
        rau=[0.0],
        u=[1.0],
        v=[0.0],
        humrel=[[1.0, 1.0]],
        veget=[[0.2, 0.5]],
        veget_max=[[0.2, 0.5]],
        lai=[[0.0, 2.0]],
        qsintveg=[[0.0, 0.2]],
        qsintmax=[[0.0, 1.0]],
        rstruct=[[0.0, 0.0]],
        vbeta23=[[0.0, 0.1]],
        kzero=[0.0, 2.0],
        rveg_pft=[1.0, 1.0],
    )
    result = diffuco_main_source_routed(
        ldq_cdrag_from_gcm=True,
        drag_inputs={"u": [1.0], "v": [0.0], "q_cdrag": [0.1], "nvm": 2},
        pb=[101325.0],
        temp_sol=[279.0],
        ok_co2=False,
        trans_inputs=trans,
        almaoutput=False,
        hist2_id=2,
    )
    np.testing.assert_allclose(result.drag.q_cdrag_pft, [[0.1, 0.1]])
    assert [name for name, _ in result.history_secondary] == [
        "raero",
        "cdrag",
        "Wind",
        "qsatt",
    ]
    assert result.transpiration.vbeta3[0, 1] > 0.0


def _enerbil_inputs(ok_explicitsnow=True):
    local = dict(
        emis=[0.98],
        temp_sol=[274.0],
        rau=[1.2],
        u=[2.0],
        v=[0.0],
        q_cdrag=[0.01],
        vbeta=[0.4],
        valpha=[0.8],
        vbeta1=[0.1],
        vbeta5=[0.0],
        qair=[0.003],
        epot_air=[275000.0],
        psnew=[276000.0],
        qsol_sat_new=[0.005],
        temp_sol_new=[275.0],
        lwdown=[300.0],
        swnet=[100.0],
        dt_sechiba=1800.0,
    )
    snow = dict(
        emis=[0.98],
        temp_sol=[274.0],
        temp_sol_new=[275.0],
        lwdown=[300.0],
        swnet=[100.0],
        rau=[1.2],
        u=[2.0],
        v=[0.0],
        q_cdrag=[0.01],
        vbeta=[0.4],
        valpha=[0.8],
        vbeta1=[0.1],
        vbeta5=[0.0],
        qair=[0.003],
        epot_air=[275000.0],
        qsol_sat_new=[0.005],
        pb=[101325.0],
        precip_rain=[0.0],
        snowdz=[[0.1, 0.0, 0.0]],
        temp_air=[276.0],
        pgflux=[-9.0],
        soilcap=[2.0e6],
        dt_sechiba=1800.0,
        ok_explicitsnow=ok_explicitsnow,
    )
    correction = dict(
        emis=[0.98],
        rau=[1.2],
        u=[2.0],
        v=[0.0],
        q_cdrag=[0.01],
        epot_air=[275000.0],
        psnew=[276000.0],
        pb=[101325.0],
    )
    return local, snow, correction


def test_enerbil_flux_composes_diagnostics_snow_and_milly_correction():
    local, snow, correction = _enerbil_inputs()
    result = enerbil_flux_source_routed(
        local_inputs=local, snow_inputs=snow, correction_inputs=correction
    )

    assert result.flux.evapot[0] > 0.0
    assert result.correction.evapot_corr[0] <= result.flux.evapot[0]
    assert result.snow.pgflux[0] != -9.0
    np.testing.assert_allclose(result.tair, np.asarray(local["epot_air"]) / 1004.675)


def test_condveg_background_routes_both_sources_and_retries_aggregate():
    seen = []
    attempts = {}

    def aggregate(request):
        seen.append(request)
        attempts[request.callsign] = attempts.get(request.callsign, 0) + 1
        npts = request.lalo.shape[0]
        index = np.zeros((npts, request.nbvmax, 2), dtype=np.int32)
        area = np.zeros((npts, request.nbvmax), dtype=np.float64)
        if attempts[request.callsign] == 1:
            return AggregatePacket(index, area, False)
        index[0, :2] = ((1, 1), (2, 2))
        area[0, :2] = (1.0, 3.0)
        return AggregatePacket(index, area, True)

    target = InterpolationTarget(
        lalo=np.array([[45.0, 2.0], [46.0, 3.0]]),
        resolution=np.full((2, 2), 2.0),
        neighbours=np.tile(np.arange(1, 9), (2, 1)),
        contfrac=np.ones(2),
    )
    coordinates = dict(
        longitude=np.array([0.0, 0.001]),
        latitude=np.array([45.0, 45.001]),
        mask_variable=np.ones((2, 2)),
    )
    result = condveg_background_soilalb_source_routed(
        bg_alb_vis=InterpolationSource(
            values=np.array([[0.1, 0.2], [0.3, 0.4]]),
            variable_name="bg_alb_vis",
            **coordinates,
        ),
        bg_alb_nir=InterpolationSource(
            values=np.array([[0.5, 0.6], [0.7, 0.8]]),
            variable_name="bg_alb_nir",
            **coordinates,
        ),
        target=target,
        aggregate=aggregate,
    )

    np.testing.assert_allclose(result.soilalb_bg[0], [0.325, 0.725])
    np.testing.assert_allclose(result.soilalb_bg[1], [0.247, 0.247])
    np.testing.assert_allclose(result.aalb_bg, [1.0, -1.0])
    assert [request.callsign for request in seen] == [
        "bg_alb_vis map",
        "bg_alb_vis map",
        "bg_alb_nir map",
        "bg_alb_nir map",
    ]
    assert seen[1].nbvmax == 2 * seen[0].nbvmax
    assert seen[3].nbvmax == 2 * seen[2].nbvmax
    np.testing.assert_array_equal(seen[0].mask, np.ones((2, 2), dtype=np.int32))


def test_condveg_finalize_routes_background_and_three_soil_albedo_fields():
    common = dict(
        z0m=[0.1], z0h=[0.01], roughheight=[1.0], roughheight_pft=[[0.0, 1.0]]
    )
    background = condveg_finalize_restart_packet(
        **common, alb_bg_modis=True, soilalb_bg=[[0.1, 0.2]]
    )
    assert "soilalbedo_bg" in background and "soilalbedo_dry" not in background
    ordinary = condveg_finalize_restart_packet(
        **common,
        alb_bg_modis=False,
        soilalb_dry=[[0.1, 0.2]],
        soilalb_wet=[[0.05, 0.1]],
        soilalb_moy=[[0.075, 0.15]],
    )
    assert {"soilalbedo_dry", "soilalbedo_wet", "soilalbedo_moy"} <= set(ordinary)


def test_usda_table_preserves_fortran_silt_sand_clay_order():
    table = get_soilcorr_usda_source_routed()
    np.testing.assert_allclose(table[0], [0.04, 0.93, 0.03])
    np.testing.assert_allclose(table[-1], [0.30, 0.15, 0.55])
    np.testing.assert_allclose(np.sum(table, axis=1), 1.0)
    with pytest.raises(ValueError, match="12 USDA"):
        get_soilcorr_usda_source_routed(11)


def test_slowproc_change_frac_recomputes_and_checks_surface_state():
    result = slowproc_change_frac_source_routed(
        agri_peat=False,
        veget_max_new=[[0.2, 0.8]],
        frac_nobio_new=[[0.0]],
        lai=[[0.0, 1.0]],
        pref_soil_veg=[1, 1],
        ext_coeff_vegetfrac=[0.5, 0.5],
        nstm=1,
    )
    assert result.vegetation.veget_max.shape == (1, 2)
    np.testing.assert_allclose(result.vegetation.soiltile, [[1.0]])
    check = slowproc_checkveget_source_routed(
        frac_nobio=result.vegetation.frac_nobio,
        veget_max=result.vegetation.veget_max,
        veget=result.vegetation.veget,
        tot_bare_soil=result.tot_bare_soil,
        soiltile=result.vegetation.soiltile,
    )
    assert check.warnings == ()


def test_slowproc_checkveget_keeps_soiltile_as_warning_but_fraction_errors_fatal():
    valid = dict(
        frac_nobio=[[0.0]],
        veget_max=[[0.2, 0.8]],
        veget=[[0.2, 0.6]],
        tot_bare_soil=[0.4],
    )
    check = slowproc_checkveget_source_routed(**valid, soiltile=[[0.9]])
    assert check.warnings == ("soiltile does not sum to one",)
    with pytest.raises(ValueError, match="does not equal 1"):
        slowproc_checkveget_source_routed(
            **{**valid, "veget_max": [[0.1, 0.8]]}, soiltile=[[1.0]]
        )


def test_slowproc_finalize_routes_four_source_gates():
    names = (
        "veget veget_max lai frac_nobio frac_age njsc clayfraction sandfraction height "
        "peatPET_lastyear growth_day GSL peatPET_thisyear precipitation_lastsummer "
        "precipitation_thissummer summerpet_long summerp_long peatC peatC_ok soil_ph "
        "poor_soils bulk_density reinf_slope laimap veget_year"
    ).split()
    state = {name: np.asarray([1.0]) for name in names}
    packet, call_stomate = slowproc_finalize_restart_packet(
        state=state,
        hydrol_cwrr=True,
        read_lai=True,
        map_pft_format=True,
        ok_stomate=True,
    )
    assert {"reinf_slope", "laimap", "veget_year"} <= set(packet)
    assert call_stomate is True
    packet, call_stomate = slowproc_finalize_restart_packet(
        state=state,
        hydrol_cwrr=False,
        read_lai=False,
        map_pft_format=False,
        ok_stomate=False,
    )
    assert not ({"reinf_slope", "laimap", "veget_year"} & set(packet))
    assert call_stomate is False


def test_slowproc_initialize_rejects_incoherent_timestep_before_state_build():
    with pytest.raises(ValueError, match="dt_stomate >= dt_sechiba"):
        slowproc_initialize_source_routed(
            init_kwargs={},
            dt_stomate=900.0,
            dt_sechiba=1800.0,
            ok_stomate=True,
            qsintcst=0.1,
        )


def test_slowproc_main_history_routes_hist2_and_computes_npp():
    values = dict(
        gpp=np.asarray([[9.0, 10.0, 20.0]]),
        resp_maint=np.asarray([[3.0, 2.0, 4.0]]),
        resp_growth=np.asarray([[2.0, 1.0, 3.0]]),
        resp_hetero=np.asarray([[1.0, 5.0, 6.0]]),
    )
    active = slowproc_main_history_routing(**values, hist2_id=2)
    np.testing.assert_allclose(active.npp, [[0.0, 7.0, 13.0]])
    assert tuple(name for name, _ in active.primary) == (
        "maint_resp",
        "hetero_resp",
        "growth_resp",
        "npp",
    )
    assert active.secondary == active.primary
    assert slowproc_main_history_routing(**values, hist2_id=-1).secondary == ()
