"""Bounded source-shaped inputs for real PFT14 DIFFUCO -> ENERBIL owners.

This is a component fixture, not a model step or an independent numerical
oracle. All 14 columns exist, but only PFT14 has supported C3 physics here.
The synthetic boundary state is declared explicitly; it is not a restart or a
claim that CONDVEG/THERMOSOIL/HYDROL produced that state.

Fortran phase: sechiba.f90:997-1019 (DIFFUCO then ENERBIL), 1021-1100
(subsequent HYDROL). Source-shaped dimensions and controls follow
diffuco.f90:629-710, 2214-2234, 2338-2400, 2705-2974.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CASE_NAMES = (
    "dry_active",
    "wet_active",
    "water_stress",
    "zero_humrel",
    "absent_pft14",
    "zero_lai",
    "dark",
    "warm_dew",
    "frozen_dew",
    "growth_temperature_mask",
)


class CanopyFixture(NamedTuple):
    case_names: tuple[str, ...]
    diffuco_precall: object
    static_kwargs: dict[str, object]
    driver: dict[str, np.ndarray]
    albedo: dict[str, np.ndarray]
    thermal: dict[str, np.ndarray]


class CanopyComponents(NamedTuple):
    fixture: CanopyFixture
    diffuco: object
    enerbil_precall: object
    enerbil: object | None


def make_canopy_fixture(indices: tuple[int, ...] | None = None) -> CanopyFixture:
    """Return ten explicit 14-PFT cases, optionally selecting point indices.

    Growth temperature and tphoto limits are degrees Celsius, while air and
    surface temperature are kelvin; pressure is hPa. Synthetic C3 constants
    exercise the original equations and are not calibrated paper-case inputs.
    """
    from jax_orchidee.sechiba.diffuco import DiffucoFirstStepPrecallAssembly

    n, nvm = len(CASE_NAMES), 14

    def grid(value):
        return np.full(n, value, dtype=np.float64)

    def pft(value=0.0):
        return np.full((n, nvm), value, dtype=np.float64)

    payload = {
        name: grid(value)
        for name, value in {
            "u": 2.0,
            "v": 0.5,
            "zlev": 35.0,
            "z0h": 0.08,
            "z0m": 0.2,
            "roughheight": 4.0,
            "temp_sol": 298.0,
            "temp_air": 300.0,
            "qsurf": 0.010,
            "qair": 0.009,
            "pb": 1000.0,
            "rau": 1.2,
            "snow": 0.0,
            "totfrac_nobio": 0.1,
            "frac_snow_veg": 0.0,
            "evapot": 2.0,
            "evapot_corr": 2.0,
            "flood_frac": 0.1,
            "flood_res": 0.01,
            "tot_bare_soil": 0.2,
            "evap_bare_lim": 0.25,
            "temp_growth": 25.0,
            "swdown": 500.0,
            "swnet": 400.0,
            "ccanopy": 410.0,
            "control_salinity": 0.8,
            "control_inudate": 0.7,
        }.items()
    }
    for name, value in {
        "roughheight_pft": 4.0,
        "temp_sol_pft": 298.0,
        "humrel": 0.0,
        "veget": 0.0,
        "veget_max": 0.0,
        "lai": 0.0,
        "qsintveg": 0.0,
        "qsintmax": 0.0,
        "rstruct": 25.0,
        "height": 4.0,
    }.items():
        payload[name] = pft(value)
    for name, value in {"humrel": 0.7, "veget": 0.6, "veget_max": 0.7, "lai": 1.4, "qsintmax": 0.2}.items():
        payload[name][:, 13] = value
    payload["veget_max"][:, 0] = 0.2
    payload["assim_param"] = np.full((n, nvm, 4), 42.0, dtype=np.float64)
    payload["frac_nobio"] = np.full((n, 2), 0.05, dtype=np.float64)
    payload["snow_nobio"] = np.zeros((n, 2), dtype=np.float64)
    payload["frac_snow_nobio"] = np.zeros((n, 2), dtype=np.float64)
    payload["snowdz"] = np.full((n, 3), 0.02, dtype=np.float64)
    payload["qsintveg"][1, 13] = 0.15
    payload["humrel"][2, 13] = 0.1
    payload["humrel"][3, 13] = 0.0
    payload["veget"][4, 13] = payload["veget_max"][4, 13] = 0.0
    payload["veget_max"][4, 0] = 0.9
    payload["lai"][5, 13] = 0.0
    payload["swdown"][6] = payload["swnet"][6] = 0.0
    payload["temp_sol"][7:] = [290.0, 265.0, 298.0]
    payload["temp_sol_pft"][7:, :] = payload["temp_sol"][7:, None]
    payload["temp_air"][7:] = [295.0, 270.0, 300.0]
    payload["qair"][7:9] = [0.05, 0.1]
    payload["snow"][8] = 0.4
    payload["frac_snow_veg"][8] = 0.5
    payload["temp_growth"][9] = -10.0
    ok_laidev = np.zeros(nvm, dtype=bool)
    ok_laidev[13] = True
    params = dict(
        tphoto_min=-5.0,
        tphoto_max=50.0,
        ext_coeff=0.5,
        a1=0.85,
        b1=0.02,
        stress_gs=0.1,
        e_kmc=79430.0,
        e_kmo=36380.0,
        e_sco=-24460.0,
        e_gamma_star=37830.0,
        e_rd=46390.0,
        e_jmax=43540.0,
        d_jmax=200000.0,
        asj=659.70,
        bsj=-0.75,
        e_vcmax=58520.0,
        d_vcmax=200000.0,
        asv=668.39,
        bsv=-1.07,
        e_gm=49600.0,
        d_gm=437400.0,
        s_gm=1400.0,
        arjv=2.59,
        brjv=-0.035,
        gm25=0.12,
        stress_gm=0.2,
        g0=0.01,
        kmc25=405.0,
        kmo25=278400.0,
        sco25=2590.0,
        gamma_star25=42.75,
        alpha_ll=0.372,
        theta=0.7,
        stress_vcmax=0.2,
        rveg_pft=1.0,
        rstruct_const=25.0,
    )
    static = dict(
        trans_co2_pft_params=params,
        rstruct_const=np.full(nvm, 25.0),
        dt_sechiba=1800.0,
        laimax=8.0,
        lai_level_depth=0.15,
        min_wind=0.1,
        snowcri=1.5,
        ok_snowfact=True,
        rough_dyn=True,
        ok_laidev=ok_laidev,
    )
    selected = tuple(range(n)) if indices is None else tuple(indices)
    payload = {name: value[list(selected)] for name, value in payload.items()}
    precall = DiffucoFirstStepPrecallAssembly(
        payload=payload,
        restart_inputs=(),
        driver_inputs=(),
        source_kernel_inputs=(),
        control_inputs=("control_salinity", "control_inudate"),
        missing_inputs=(),
        missing_local_process_inputs=(),
        provenance=(
            "Synthetic explicitly declared component boundary; not a restart or completed upstream owner.",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:629-710,2214-2234,2338-2400,2705-2974",
        ),
        notes=("Only PFT14 is scientifically computed. Other PFT columns carry declared source backgrounds.",),
    )
    driver = {name: payload[name] for name in ("zlev", "swdown", "temp_air", "u", "v", "qair", "pb")}
    driver.update(lwdown=np.full(len(selected), 350.0), precip_rain=np.zeros(len(selected)))
    thermal = dict(
        soilcap=np.full(len(selected), 4.0e5),
        soilcap_pft=np.full((len(selected), nvm), 4.0e5),
        soilflx=np.full(len(selected), 10.0),
        soilflx_pft=np.full((len(selected), nvm), 10.0),
    )
    albedo = dict(albedo=np.full((len(selected), 2), 0.2))
    return CanopyFixture(tuple(CASE_NAMES[i] for i in selected), precall, static, driver, albedo, thermal)


def run_canopy_components(
    fixture: CanopyFixture | None = None,
    *,
    diffuco_use_jit: bool = False,
    enerbil_use_jit: bool = False,
    run_enerbil: bool = True,
) -> CanopyComponents:
    """Call the actual first-step wrappers, without a calendar/model driver."""
    from jax_orchidee.sechiba.diffuco import run_pft14_local_enerbil_precall_from_first_step_precall
    from jax_orchidee.sechiba.enerbil import (
        assemble_enerbil_first_step_precall_payload,
        run_enerbil_first_step_local_from_precall,
    )

    fixture = make_canopy_fixture() if fixture is None else fixture
    diffuco = run_pft14_local_enerbil_precall_from_first_step_precall(
        fixture.diffuco_precall, run_def_values={}, static_kwargs=fixture.static_kwargs, use_jit=diffuco_use_jit
    )
    enerbil_precall = assemble_enerbil_first_step_precall_payload(
        driver_or_intersurf_payload=fixture.driver,
        driver_albedo_payload=fixture.albedo,
        soil_thermal_restart_payload=fixture.thermal,
        after_diffuco_payload=diffuco.result.payload.payload,
        condveg_impaze=False,
    )
    enerbil = None
    if run_enerbil:
        enerbil = run_enerbil_first_step_local_from_precall(
            enerbil_precall,
            ok_laidev=fixture.static_kwargs["ok_laidev"],
            dt_sechiba=fixture.static_kwargs["dt_sechiba"],
            ok_explicitsnow=False,
            use_jit=enerbil_use_jit,
        )
    return CanopyComponents(fixture, diffuco, enerbil_precall, enerbil)
