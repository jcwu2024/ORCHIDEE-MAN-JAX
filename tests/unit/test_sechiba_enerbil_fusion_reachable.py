from types import SimpleNamespace

import numpy as np

import jax_orchidee.sechiba.sechiba_step as sechiba_step


def test_bucket_snow_fusion_runs_after_hydrol_and_feeds_thermosoil(monkeypatch):
    """sechiba.f90:1077-1118: fusion-corrected temperature reaches THERMOSOIL."""

    monkeypatch.setattr(sechiba_step, "ENERBIL_LOCAL_STEP_FIELDS", ())
    monkeypatch.setattr(
        sechiba_step,
        "assemble_enerbil_precall_payload",
        lambda *payloads: SimpleNamespace(payload={}),
    )
    monkeypatch.setattr(sechiba_step, "enerbil_explicit_local_step", lambda **kwargs: object())
    monkeypatch.setattr(
        sechiba_step,
        "enerbil_downstream_payload",
        lambda result: {
            "temp_sol_new": np.array([275.0]),
            "temp_sol_new_pft": np.array([[275.0, 276.0]]),
            "vevapwet": np.zeros((1, 2)),
            "vevapnu": np.zeros(1),
            "vevapnu_pft": np.zeros((1, 2)),
            "vevapflo": np.zeros(1),
            "transpir": np.zeros((1, 2)),
        },
    )
    monkeypatch.setattr(sechiba_step, "hydrol_module_explicit_step", lambda **kwargs: object())
    monkeypatch.setattr(sechiba_step, "hydrol_module_outputs", lambda result, **kwargs: object())
    monkeypatch.setattr(sechiba_step, "hydrol_module_diagnostics", lambda result, **kwargs: object())
    monkeypatch.setattr(sechiba_step, "hydrol_to_thermosoil_moisture_inputs", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sechiba_step,
        "condveg_main_minimal",
        lambda **kwargs: SimpleNamespace(frac_snow_veg=np.zeros(1), frac_snow_nobio=np.zeros((1, 1))),
    )

    captured = {}

    def assemble_thermosoil(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kwargs={})

    monkeypatch.setattr(sechiba_step, "assemble_thermosoil_explicit_kwargs", assemble_thermosoil)
    monkeypatch.setattr(sechiba_step, "thermosoil_explicit_step", lambda **kwargs: object())
    monkeypatch.setattr(sechiba_step, "thermosoil_downstream_payload", lambda result: {})

    result = sechiba_step.sechiba_explicit_coupled_step(
        diffuco_payload={},
        enerbil_inputs={
            "soilcap": np.array([1000.0]),
            "soilcap_pft": np.array([[1000.0, 2000.0]]),
            "ok_laidev": np.array([False, True]),
            "dt_sechiba": 10.0,
        },
        hydrol_inputs={
            "tot_melt": np.array([0.01]),
            "soiltile": np.ones((1, 1)),
            "pref_soil_veg": np.array([1, 1]),
        },
        hydrol_diagnostic_inputs={},
        condveg_inputs={
            "ok_explicitsnow": False,
            "snowdz": np.zeros((1, 3)),
            "snowrho": np.ones((1, 3)),
        },
        thermosoil_inputs={"snowtemp": np.ones((1, 3))},
    )

    expected_grid = 275.0 - 0.01 * (2.8345e6 - 2.5008e6) / 1000.0
    expected_pft2 = 276.0 - 0.01 * (2.8345e6 - 2.5008e6) / 2000.0
    np.testing.assert_allclose(result.fusion.temp_sol_new, [expected_grid])
    np.testing.assert_allclose(result.fusion.temp_sol_new_pft, [[expected_grid, expected_pft2]])
    np.testing.assert_allclose(captured["temp_sol_new"], result.fusion.temp_sol_new)
    np.testing.assert_allclose(captured["temp_sol_new_pft"], result.fusion.temp_sol_new_pft)
