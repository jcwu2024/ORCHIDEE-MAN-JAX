from types import SimpleNamespace

import numpy as np

import jax_orchidee.sechiba.thermosoil as thermosoil


def test_thermosoil_transition_consumes_old_pcapa_and_writes_new_recurrence(monkeypatch):
    """thermosoil.f90:905-913,995-1009: old coefficients in, new coefficients out."""

    old_pcapa_en = np.full((2, 3, 14), 11.0)
    state = thermosoil.ThermosoilRecurrenceState(
        ptn=np.full((2, 3, 14), 270.0),
        cgrnd=np.full((2, 2, 14), 1.0),
        dgrnd=np.full((2, 2, 14), 2.0),
        cgrnd_snow=np.full((2, 3), 3.0),
        dgrnd_snow=np.full((2, 3), 4.0),
        lambda_snow=np.full(2, 0.5),
        pcapa_en=old_pcapa_en,
        temp_sol_beg=np.full(2, 269.0),
        soilcap=np.full(2, 1000.0),
    )
    new_pcapa_en = np.full((2, 3, 14), 22.0)
    captured = {}

    def fake_step(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            profile=SimpleNamespace(ptn=np.full((2, 3, 14), 271.0)),
            getdiff=SimpleNamespace(pcapa_en=new_pcapa_en),
            energy=SimpleNamespace(temp_sol_beg=np.full(2, 272.0)),
            coef=SimpleNamespace(
                soil=SimpleNamespace(
                    cgrnd=np.full((2, 2, 14), 5.0),
                    dgrnd=np.full((2, 2, 14), 6.0),
                ),
                cgrnd_snow=np.full((2, 3), 7.0),
                dgrnd_snow=np.full((2, 3), 8.0),
                lambda_snow=np.full(2, 0.7),
                soilcap=np.full(2, 2000.0),
            ),
        )

    monkeypatch.setattr(thermosoil, "thermosoil_explicit_step", fake_step)
    result = thermosoil.thermosoil_explicit_transition(state=state, forcing_marker="landpoint-varying")

    assert captured["pcapa_en_previous"] is old_pcapa_en
    assert captured["ptn"] is state.ptn
    assert captured["temp_sol_beg"] is state.temp_sol_beg
    np.testing.assert_allclose(result.next_state.pcapa_en, new_pcapa_en)
    np.testing.assert_allclose(result.next_state.ptn, 271.0)
    np.testing.assert_allclose(result.next_state.soilcap, 2000.0)
    np.testing.assert_allclose(result.next_state.temp_sol_beg, 272.0)

