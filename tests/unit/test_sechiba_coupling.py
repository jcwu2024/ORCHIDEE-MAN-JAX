from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.coupling import (  # noqa: E402
    assemble_enerbil_precall_payload,
    assemble_thermosoil_explicit_kwargs,
    assemble_thermosoil_no_explicit_snow_kwargs,
    enerbil_precall_field_names,
    thermosoil_downstream_payload,
)
from jax_orchidee.sechiba.hydrol import HydrolThermosoilMoistureInputs  # noqa: E402
from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    run_thermosoil_first_step_module,
    thermosoil_explicit_step,
    thermosoil_no_explicit_snow_step,
)


def _moisture_payload():
    return HydrolThermosoilMoistureInputs(
        shumdiag_perma=np.array([[0.2, 0.3]], dtype=np.float64),
        mc_layh=np.array([[0.2, 0.3]], dtype=np.float64),
        mcl_layh=np.array([[0.2, 0.3]], dtype=np.float64),
        tmc_layh=np.array([[20.0, 30.0]], dtype=np.float64),
        mc_layh_pft=np.array([[[0.2], [0.3]]], dtype=np.float64),
        mcl_layh_pft=np.array([[[0.2], [0.3]]], dtype=np.float64),
        tmc_layh_pft=np.array([[[20.0], [30.0]]], dtype=np.float64),
    )


def _boundary_kwargs():
    return {
        "moisture": _moisture_payload(),
        "ptn": np.array([[[280.0], [281.0], [282.0]]], dtype=np.float64),
        "cgrnd": np.array([[[275.0], [276.0]]], dtype=np.float64),
        "dgrnd": np.array([[[0.2], [0.3]]], dtype=np.float64),
        "cgrnd_snow": np.array([[260.0, 261.0, 262.0]], dtype=np.float64),
        "dgrnd_snow": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "lambda_snow": np.array([0.5], dtype=np.float64),
        "temp_sol_new": np.array([279.0], dtype=np.float64),
        "temp_sol_new_pft": np.array([[279.0]], dtype=np.float64),
        "snowrho": np.array([[100.0, 200.0, 300.0]], dtype=np.float64),
        "snowtemp": np.array([[270.0, 268.0, 266.0]], dtype=np.float64),
        "snowdz": np.array([[0.01, 0.02, 0.04]], dtype=np.float64),
        "njsc": np.array([1], dtype=np.int32),
        "veget_max": np.array([[1.0]], dtype=np.float64),
        "pb": np.array([1000.0], dtype=np.float64),
        "dlt": np.array([0.1, 0.2, 0.4], dtype=np.float64),
        "dz1": np.array([10.0, 3.0], dtype=np.float64),
        "zlt": np.array([0.1, 0.3, 0.7], dtype=np.float64),
        "znt": np.array([0.05, 0.2, 0.5], dtype=np.float64),
        "dz5": np.array([0.2, 0.4], dtype=np.float64),
        "dt_sechiba": 1800.0,
        "lambda_thermal": 0.5,
        "frac_snow_veg": np.array([0.4], dtype=np.float64),
        "frac_snow_nobio": np.array([[0.0]], dtype=np.float64),
        "totfrac_nobio": np.array([0.0], dtype=np.float64),
        "temp_sol_beg": np.array([278.0], dtype=np.float64),
        "soilcap_initial": np.array([1000.0], dtype=np.float64),
        "refsoc": np.array([[0.0, 10.0, 20.0]], dtype=np.float64),
        "ok_laidev": np.array([False]),
    }


def test_assemble_enerbil_precall_payload_requires_all_fortran_boundary_inputs():
    complete = {name: object() for name in enerbil_precall_field_names()}

    boundary = assemble_enerbil_precall_payload(
        {name: complete[name] for name in ("lai", "gpp", "veget_max", "vbeta3")},
        {name: value for name, value in complete.items() if name not in {"lai", "gpp", "veget_max", "vbeta3"}},
    )

    assert boundary.provenance[0].endswith("sechiba.f90::sechiba_main lines 997-1019")
    assert set(enerbil_precall_field_names()) <= set(boundary.payload)
    assert boundary.payload["gpp"] is complete["gpp"]

    incomplete = dict(complete)
    incomplete.pop("soilflx")
    with pytest.raises(ValueError, match="soilflx"):
        assemble_enerbil_precall_payload(incomplete)


def test_assemble_enerbil_precall_payload_rejects_conflicting_duplicate_sources():
    complete = {name: object() for name in enerbil_precall_field_names()}

    with pytest.raises(ValueError, match="duplicate ENERBIL pre-call field"):
        assemble_enerbil_precall_payload(complete, {"gpp": object()})


def test_assemble_thermosoil_explicit_kwargs_rejects_missing_organic_fraction_source():
    kwargs = _boundary_kwargs()
    kwargs.pop("refsoc")

    with pytest.raises(ValueError, match="provide exactly one"):
        assemble_thermosoil_explicit_kwargs(**kwargs)


def test_sechiba_thermosoil_coupling_runs_explicit_step_and_exports_downstream_payload():
    boundary = assemble_thermosoil_explicit_kwargs(**_boundary_kwargs())

    assert boundary.provenance[0].endswith("sechiba.f90::sechiba_main lines 1049-1118")
    assert boundary.kwargs["tmc_layh"] is boundary.advertised_payload["soilmoist"]
    assert boundary.kwargs["tmc_layh_pft"] is boundary.advertised_payload["soilmoist_pft"]

    result = thermosoil_explicit_step(**boundary.kwargs)
    payload = thermosoil_downstream_payload(result)

    assert {
        "stempdiag",
        "soilcap",
        "soilcap_pft",
        "soilflx",
        "soilflx_pft",
        "gtemp",
        "ptnlev1",
        "ptn_pftmean",
        "pkappa_pftmean",
        "deephum_prof",
        "deeptemp_prof",
        "cgrnd",
        "dgrnd",
        "cgrnd_snow",
        "dgrnd_snow",
        "lambda_snow",
    } <= set(payload)
    np.testing.assert_allclose(np.asarray(payload["stempdiag"]), np.asarray(result.profile.stempdiag))
    np.testing.assert_allclose(np.asarray(payload["soilcap"]), np.asarray(result.coef.soilcap))
    np.testing.assert_allclose(np.asarray(payload["soilflx"]), np.asarray(result.coef.soilflx))
    np.testing.assert_allclose(np.asarray(payload["gtemp"]), np.asarray(result.final.gtemp))


def test_sechiba_thermosoil_coupling_runs_no_explicit_snow_step_without_three_layer_snow():
    kwargs = _boundary_kwargs()
    boundary = assemble_thermosoil_no_explicit_snow_kwargs(
        moisture=kwargs["moisture"],
        temp_sol_new=kwargs["temp_sol_new"],
        temp_sol_new_pft=kwargs["temp_sol_new_pft"],
        snow=np.array([0.02 * 330.0], dtype=np.float64),
        ptn=np.array([[[280.0], [281.0], [282.0], [283.0]]], dtype=np.float64),
        cgrnd=np.array([[[275.0], [276.0], [277.0]]], dtype=np.float64),
        dgrnd=np.array([[[0.2], [0.3], [0.4]]], dtype=np.float64),
        cgrnd_snow=kwargs["cgrnd_snow"],
        dgrnd_snow=kwargs["dgrnd_snow"],
        lambda_snow=kwargs["lambda_snow"],
        njsc=kwargs["njsc"],
        veget_max=kwargs["veget_max"],
        dlt=np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64),
        dz1=np.array([10.0, 3.0, 1.0], dtype=np.float64),
        zlt=np.array([0.1, 0.3, 0.7, 1.5], dtype=np.float64),
        znt=np.array([0.05, 0.2, 0.5, 1.1], dtype=np.float64),
        dz5=np.array([0.2, 0.4, 0.8], dtype=np.float64),
        dt_sechiba=kwargs["dt_sechiba"],
        lambda_thermal=kwargs["lambda_thermal"],
        temp_sol_beg=kwargs["temp_sol_beg"],
        soilcap_initial=kwargs["soilcap_initial"],
        ok_laidev=kwargs["ok_laidev"],
    )

    assert "snowrho" not in boundary.kwargs
    result = thermosoil_no_explicit_snow_step(**boundary.kwargs)
    payload = thermosoil_downstream_payload(result)

    np.testing.assert_allclose(np.asarray(payload["cgrnd_snow"]), 0.0)
    np.testing.assert_allclose(np.asarray(payload["dgrnd_snow"]), 0.0)
    np.testing.assert_allclose(np.asarray(payload["lambda_snow"]), [0.5])


def test_thermosoil_first_step_wrapper_dispatches_no_explicit_snow_without_refsoc():
    restart = SimpleNamespace(
        ptn=np.array([[[280.0], [281.0], [282.0], [283.0]]], dtype=np.float64),
        cgrnd=np.array([[[275.0], [276.0], [277.0]]], dtype=np.float64),
        dgrnd=np.array([[[0.2], [0.3], [0.4]]], dtype=np.float64),
        cgrnd_snow=np.ones((1, 3), dtype=np.float64),
        dgrnd_snow=np.ones((1, 3), dtype=np.float64),
        lambda_snow=np.array([0.5], dtype=np.float64),
        gtemp=np.array([278.0], dtype=np.float64),
        refsoc=None,
    )
    closure = run_thermosoil_first_step_module(
        moisture=_moisture_payload(),
        thermosoil_restart=restart,
        enerbil_payload={
            "temp_sol_new": np.array([279.0], dtype=np.float64),
            "temp_sol_new_pft": np.array([[279.0]], dtype=np.float64),
            "soilcap": np.array([1000.0], dtype=np.float64),
        },
        condveg_result=SimpleNamespace(frac_snow_veg=np.array([0.0]), frac_snow_nobio=np.array([[0.0]])),
        snow=np.array([0.02 * 330.0], dtype=np.float64),
        njsc=np.array([1], dtype=np.int32),
        veget_max=np.array([[1.0]], dtype=np.float64),
        pb=np.array([1000.0], dtype=np.float64),
        totfrac_nobio=np.array([0.0], dtype=np.float64),
        dlt=np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64),
        dz1=np.array([10.0, 3.0, 1.0], dtype=np.float64),
        zlt=np.array([0.1, 0.3, 0.7, 1.5], dtype=np.float64),
        znt=np.array([0.05, 0.2, 0.5, 1.1], dtype=np.float64),
        dz5=np.array([0.2, 0.4, 0.8], dtype=np.float64),
        dt_sechiba=1800.0,
        ok_laidev=np.array([False]),
        ok_explicitsnow=False,
    )

    assert closure.ok
    assert closure.missing_inputs == ()
    np.testing.assert_allclose(np.asarray(closure.downstream_payload["cgrnd_snow"]), 0.0)
