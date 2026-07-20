from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.sechiba.condveg import (
    CondvegInitializeBoundaryError,
    condveg_initialize,
)


def _inputs(*, npts=3, nvm=16):
    veget = np.zeros((npts, nvm))
    veget[:, 13] = np.linspace(0.3, 0.7, npts)
    veget_max = veget.copy()
    return {
        "l_first": True,
        "veget": veget,
        "veget_max": veget_max,
        "frac_nobio": np.zeros((npts, 1)),
        "totfrac_nobio": np.zeros(npts),
        "zlev": np.full(npts, 10.0),
        "drysoil_frac": np.linspace(0.1, 0.8, npts),
        "height": np.full((npts, nvm), 0.5),
        "snowdz": np.zeros((npts, 3)),
        "snowrho": np.full((npts, 3), 100.0),
        "tot_bare_soil": 1.0 - veget[:, 13],
        "snow": np.zeros(npts),
        "snow_age": np.zeros(npts),
        "snow_nobio": np.zeros((npts, 1)),
        "snow_nobio_age": np.zeros(npts),
        "temp_air": np.full(npts, 285.0),
        "pb": np.full(npts, 1013.0),
        "u": np.full(npts, 2.0),
        "v": np.full(npts, 1.0),
        "lai": np.full((npts, nvm), 2.0),
        "alb_leaf_vis": np.linspace(0.1, 0.2, nvm),
        "alb_leaf_nir": np.linspace(0.2, 0.4, nvm),
        "fixed_snow_albedo": 0.75,
    }


def _restart(npts, nvm):
    return {
        "soilalbedo_bg": np.column_stack((np.linspace(0.1, 0.2, npts), np.linspace(0.2, 0.3, npts))),
        "z0m": np.linspace(0.10, 0.20, npts),
        "z0h": np.linspace(0.01, 0.02, npts),
        "roughheight": np.linspace(0.2, 0.4, npts),
        "roughheight_pft": np.full((npts, nvm), 0.33),
    }


def test_restart_state_is_preserved_and_owner_payload_is_mergeable():
    npts, nvm = 3, 16
    restart = _restart(npts, nvm)
    result = condveg_initialize(**_inputs(npts=npts, nvm=nvm), restart_fields=restart)

    assert result.l_first is False
    assert result.recomputed_roughness is False
    np.testing.assert_array_equal(result.z0m, restart["z0m"])
    np.testing.assert_array_equal(result.roughheight_pft, restart["roughheight_pft"])
    np.testing.assert_array_equal(result.emis, np.ones(npts))
    assert result.albedo.shape == (npts, 2)
    assert result.frac_snow_nobio.shape == (npts, 1)
    assert set(result.as_payload()) >= {"albedo", "z0m", "emis", "soilalb_bg"}
    assert "condveg_initialize lines 102-310" in result.provenance[0]


def test_any_missing_primary_roughness_restart_recomputes_the_whole_group():
    npts, nvm = 2, 18
    restart = _restart(npts, nvm)
    restart["z0h"] = np.full(npts, 999999.0)
    result = condveg_initialize(**_inputs(npts=npts, nvm=nvm), restart_fields=restart)

    assert result.recomputed_roughness is True
    assert not np.any(np.asarray(result.z0h) == 999999.0)
    assert not np.array_equal(np.asarray(result.z0m), restart["z0m"])
    assert not np.array_equal(np.asarray(result.roughheight_pft), restart["roughheight_pft"])


def test_mixed_soil_restart_is_preserved_but_wholly_missing_map_is_external():
    values = _inputs(npts=2, nvm=14)
    restart = _restart(2, 14)
    restart["soilalbedo_bg"][0, 0] = 999999.0
    result = condveg_initialize(**values, restart_fields=restart)
    assert float(result.soilalb_bg[0, 0]) == 999999.0

    restart["soilalbedo_bg"][:] = 999999.0
    with pytest.raises(CondvegInitializeBoundaryError, match="map interpolation"):
        condveg_initialize(**values, restart_fields=restart)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"l_first": False}, "repeated call"),
        ({"alb_bg_modis": False}, "condveg_soilalb"),
        ({"impaze": True}, "impaze=True"),
        ({"rough_dyn": False}, "rough_dyn=False"),
    ],
)
def test_target_external_structural_branches_are_rejected(override, message):
    values = _inputs(npts=2, nvm=14)
    values.update(override)
    error = RuntimeError if override == {"l_first": False} else CondvegInitializeBoundaryError
    with pytest.raises(error, match=message):
        condveg_initialize(**values, restart_fields=_restart(2, 14))

