from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.sechiba.diffuco import (
    DiffucoInitializeBoundaryError,
    diffuco_initialize,
)


def test_restart_defaults_are_source_ordered_and_land_dimension_is_dynamic():
    npts, nvm, nlai = 3, 16, 4
    rstruct_const = np.linspace(10.0, 25.0, nvm)
    leaf_ci = np.full((npts, nvm, nlai), 999999.0)
    leaf_ci[1, 13, 2] = 187.5
    cdrag_pft = np.arange(npts * nvm, dtype=float).reshape(npts, nvm) / 100.0

    result = diffuco_initialize(
        q_cdrag=np.zeros(npts),
        rstruct_const=rstruct_const,
        nlai=nlai,
        restart_fields={"leaf_ci": leaf_ci, "cdrag_pft": cdrag_pft},
        diffuco_leafci=231.25,
    )

    np.testing.assert_allclose(result.rstruct, np.broadcast_to(rstruct_const, (npts, nvm)))
    assert result.leaf_ci.shape == (npts, nvm, nlai)
    assert float(result.leaf_ci[1, 13, 2]) == 187.5
    assert float(result.leaf_ci[0, 13, 0]) == 999999.0
    np.testing.assert_array_equal(result.q_cdrag_pft, cdrag_pft)
    assert result.ldq_cdrag_from_gcm is False
    assert result.wind is result.control_salinity is result.control_inudate is None
    assert "wind" not in result.as_payload()
    assert "diffuco_initialize lines 127-233" in result.provenance[0]

    cold = diffuco_initialize(
        q_cdrag=np.zeros(npts),
        rstruct_const=rstruct_const,
        nlai=nlai,
        diffuco_leafci=231.25,
    )
    np.testing.assert_array_equal(cold.leaf_ci, np.full((npts, nvm, nlai), 231.25))


def test_whole_array_restart_rules_preserve_mixed_values_and_no_cdrag_assignment():
    npts, nvm = 2, 14
    rstruct = np.full((npts, nvm), 999999.0)
    rstruct[0, 13] = 42.0
    result = diffuco_initialize(
        q_cdrag=np.array([-0.2, -0.1]),
        rstruct_const=np.arange(nvm, dtype=float),
        nlai=2,
        restart_fields={"rstruct": rstruct, "cdrag_pft": np.full((npts, nvm), 999999.0)},
    )

    np.testing.assert_array_equal(result.rstruct, rstruct)
    assert result.q_cdrag_pft is None
    assert "q_cdrag_pft" not in result.as_payload()
    # Fortran uses ABS(MAXVAL(q_cdrag)), not MAXVAL(ABS(q_cdrag)).
    assert result.ldq_cdrag_from_gcm is True


def test_run_def_cdrag_override_and_external_branches_are_explicit():
    common = dict(q_cdrag=np.zeros(2), rstruct_const=np.ones(14), nlai=1)
    assert diffuco_initialize(**common, cdrag_from_gcm=True).ldq_cdrag_from_gcm is True

    with pytest.raises(DiffucoInitializeBoundaryError, match="chemistry_initialize"):
        diffuco_initialize(**common, ok_bvoc=True)
    with pytest.raises(DiffucoInitializeBoundaryError, match="ok_co2=False"):
        diffuco_initialize(**common, ok_co2=False)
