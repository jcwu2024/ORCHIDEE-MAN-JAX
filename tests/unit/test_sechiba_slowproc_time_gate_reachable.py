import numpy as np

from jax_orchidee.sechiba.slowproc import slowproc_surface_transition_explicit


def test_slowproc_non_daily_step_preserves_landpoint_surface_state():
    """slowproc.f90:654-663,1078-1121: do_slow=false carries state unchanged."""

    lai = np.array([[0.0, 2.0, 3.0], [0.0, 1.0, 4.0]])
    frac_nobio = np.array([[0.05], [0.20]])
    veget_max = np.array([[0.10, 0.50, 0.35], [0.20, 0.25, 0.35]])
    veget = np.array([[0.10, 0.30, 0.20], [0.20, 0.10, 0.25]])
    totfrac_nobio = np.array([0.05, 0.20])
    soiltile = np.array([[0.2, 0.8], [0.6, 0.4]])

    result = slowproc_surface_transition_explicit(
        do_slow=False,
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        veget=veget,
        totfrac_nobio=totfrac_nobio,
        soiltile=soiltile,
    )

    np.testing.assert_array_equal(result.surface.vegetation.frac_nobio, frac_nobio)
    np.testing.assert_array_equal(result.surface.vegetation.veget_max, veget_max)
    np.testing.assert_array_equal(result.surface.vegetation.veget, veget)
    np.testing.assert_array_equal(result.surface.vegetation.totfrac_nobio, totfrac_nobio)
    np.testing.assert_array_equal(result.surface.vegetation.soiltile, soiltile)
    expected_bare = veget_max[:, 0] + np.sum(veget_max[:, 1:] - veget[:, 1:], axis=1)
    np.testing.assert_allclose(result.surface.tot_bare_soil, expected_bare)

