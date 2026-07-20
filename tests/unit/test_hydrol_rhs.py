from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (
    HydrolSoilSetupResult,
    hydrol_alt_residual_solve_step,
    hydrol_soil_residual_boundary_rhs,
    hydrol_soil_rhs_main,
    hydrol_soil_setup_coefficients,
    hydrol_soil_tridiag_solve,
    hydrol_total_moisture_content,
)


def _dense_tridiag(e: np.ndarray, f: np.ndarray, g1: np.ndarray) -> np.ndarray:
    matrix = np.zeros((f.size, f.size), dtype=np.float64)
    np.fill_diagonal(matrix, f)
    np.fill_diagonal(matrix[1:, :-1], e[1:])
    np.fill_diagonal(matrix[:-1, 1:], g1[:-1])
    return matrix


def test_hydrol_soil_rhs_main_matches_source_formula_by_layer():
    setup = HydrolSoilSetupResult(
        e=np.asarray([[0.0, -0.2, -0.3, -0.4]], dtype=np.float64),
        f=np.asarray([[2.0, 2.1, 2.2, 2.3]], dtype=np.float64),
        g1=np.asarray([[-0.5, -0.6, -0.7, 0.0]], dtype=np.float64),
        ep=np.asarray([[0.0, 0.1, 0.2, 0.3]], dtype=np.float64),
        fp=np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float64),
        gp=np.asarray([[0.4, 0.5, 0.6, 0.0]], dtype=np.float64),
    )
    mcl = np.asarray([[0.30, 0.31, 0.32, 0.33]], dtype=np.float64)
    b = np.asarray([[0.10, 0.20, 0.30, 0.40]], dtype=np.float64)
    rootsink = np.asarray([[0.01, 0.02, 0.03, 0.04]], dtype=np.float64)
    dt_days = 0.25
    flux_top = np.asarray([0.05], dtype=np.float64)
    free_drain_coef = np.asarray([0.75], dtype=np.float64)

    result = hydrol_soil_rhs_main(
        setup=setup,
        mcl=mcl,
        b=b,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=flux_top,
        rootsink=rootsink,
    )
    rhs = np.asarray(result.rhs)

    expected_top = (
        setup.fp[0][0] * mcl[0, 0]
        + setup.gp[0][0] * mcl[0, 1]
        - flux_top[0]
        - (b[0, 0] + b[0, 1]) * dt_days / 2.0
        - rootsink[0, 0]
    )
    expected_mid = (
        setup.ep[0][1] * mcl[0, 0]
        + setup.fp[0][1] * mcl[0, 1]
        + setup.gp[0][1] * mcl[0, 2]
        + (b[0, 0] - b[0, 2]) * dt_days / 2.0
        - rootsink[0, 1]
    )
    expected_bottom = (
        setup.ep[0][3] * mcl[0, 2]
        + setup.fp[0][3] * mcl[0, 3]
        + (b[0, 2] + b[0, 3] * (1.0 - 2.0 * free_drain_coef[0])) * dt_days / 2.0
        - rootsink[0, 3]
    )

    assert np.allclose(rhs[0, 0], expected_top)
    assert np.allclose(rhs[0, 1], expected_mid)
    assert np.allclose(rhs[0, 3], expected_bottom)
    assert np.array_equal(np.asarray(result.e), np.asarray(setup.e))
    assert np.array_equal(np.asarray(result.f), np.asarray(setup.f))
    assert np.array_equal(np.asarray(result.g1), np.asarray(setup.g1))


def test_hydrol_rhs_setup_and_tridiag_chain_has_small_dense_residual():
    a = np.asarray([[0.09, 0.08, 0.07, 0.06]], dtype=np.float64)
    d = np.asarray([[4.0, 4.5, 5.0, 5.5]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64)
    setup = hydrol_soil_setup_coefficients(
        a=a,
        d=d,
        dz_mm=dz_mm,
        dt_days=1.0 / 48.0,
        free_drain_coef=np.asarray([1.0]),
    )
    rhs_result = hydrol_soil_rhs_main(
        setup=setup,
        mcl=np.asarray([[0.30, 0.31, 0.32, 0.33]], dtype=np.float64),
        b=np.asarray([[0.1, 0.12, 0.14, 0.16]], dtype=np.float64),
        dt_days=1.0 / 48.0,
        free_drain_coef=np.asarray([1.0]),
        flux_top=np.asarray([0.02]),
        rootsink=np.asarray([[0.0, 0.01, 0.01, 0.0]]),
    )
    solved = hydrol_soil_tridiag_solve(
        e=rhs_result.e,
        f=rhs_result.f,
        g1=rhs_result.g1,
        rhs=rhs_result.rhs,
    )

    e = np.asarray(rhs_result.e)[0]
    f = np.asarray(rhs_result.f)[0]
    g1 = np.asarray(rhs_result.g1)[0]
    rhs = np.asarray(rhs_result.rhs)[0]
    mcl_after = np.asarray(solved.mcl)[0]
    residual = _dense_tridiag(e, f, g1) @ mcl_after - rhs

    assert mcl_after.shape == rhs.shape
    assert np.all(np.isfinite(mcl_after))
    assert np.allclose(residual, 0.0, rtol=0.0, atol=1e-13)


def test_hydrol_residual_boundary_rhs_replaces_top_equation_only():
    saved = hydrol_soil_rhs_main(
        setup=HydrolSoilSetupResult(
            e=np.asarray([[0.0, -0.2, -0.3]], dtype=np.float64),
            f=np.asarray([[2.0, 2.2, 2.4]], dtype=np.float64),
            g1=np.asarray([[-0.4, -0.5, 0.0]], dtype=np.float64),
            ep=np.asarray([[0.0, 0.1, 0.2]], dtype=np.float64),
            fp=np.asarray([[1.0, 1.1, 1.2]], dtype=np.float64),
            gp=np.asarray([[0.3, 0.4, 0.0]], dtype=np.float64),
        ),
        mcl=np.asarray([[0.2, 0.3, 0.4]], dtype=np.float64),
        b=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        dt_days=0.5,
        free_drain_coef=np.asarray([1.0]),
    )
    residual = hydrol_soil_residual_boundary_rhs(saved, mcr=np.asarray([0.057]))

    assert np.allclose(np.asarray(residual.rhs)[0, 0], 0.057)
    assert np.allclose(np.asarray(residual.f)[0, 0], 1.0)
    assert np.allclose(np.asarray(residual.g1)[0, 0], 0.0)
    assert np.array_equal(np.asarray(residual.e), np.asarray(saved.e))
    assert np.array_equal(np.asarray(residual.rhs)[0, 1:], np.asarray(saved.rhs)[0, 1:])

    solved = hydrol_soil_tridiag_solve(
        e=residual.e,
        f=residual.f,
        g1=residual.g1,
        rhs=residual.rhs,
    )
    assert np.allclose(np.asarray(solved.mcl)[0, 0], 0.057)


def test_hydrol_alt_residual_solve_step_applies_residual_boundary_when_triggered():
    setup = HydrolSoilSetupResult(
        e=np.asarray([[0.0, -0.2, -0.2]], dtype=np.float64),
        f=np.asarray([[1.0, 1.2, 1.3]], dtype=np.float64),
        g1=np.asarray([[-0.1, -0.1, 0.0]], dtype=np.float64),
        ep=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        fp=np.asarray([[0.02, 0.02, 0.02]], dtype=np.float64),
        gp=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
    )
    result = hydrol_alt_residual_solve_step(
        setup=setup,
        mcl_before=np.asarray([[0.04, 0.05, 0.06]], dtype=np.float64),
        mc_before=np.asarray([[0.04, 0.05, 0.06]], dtype=np.float64),
        profil_froz=np.zeros((1, 3), dtype=np.float64),
        mcr=np.asarray([0.057], dtype=np.float64),
        b=np.zeros((1, 3), dtype=np.float64),
        dz_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        dt_days=1.0,
        free_drain_coef=np.asarray([1.0], dtype=np.float64),
        flux_top=np.asarray([0.1], dtype=np.float64),
        resolv=np.asarray([True]),
        mask_soiltile=np.asarray([1.0], dtype=np.float64),
        k_bottom=np.asarray([2.0], dtype=np.float64),
    )

    assert np.asarray(result.resolv_alt)[0]
    np.testing.assert_allclose(np.asarray(result.second.mcl)[0, 0], 0.057)
    np.testing.assert_allclose(np.asarray(result.mc_after), np.asarray(result.second.mcl))
    np.testing.assert_allclose(np.asarray(result.flux_bottom), np.asarray([2.0]))
    np.testing.assert_allclose(
        np.asarray(result.tmc),
        np.asarray(hydrol_total_moisture_content(result.mc_after, np.asarray([0.0, 2.0, 4.0]))),
    )


def test_hydrol_alt_residual_solve_step_preserves_first_solve_when_not_triggered():
    setup = HydrolSoilSetupResult(
        e=np.asarray([[0.0, -0.1, -0.1]], dtype=np.float64),
        f=np.asarray([[1.0, 1.2, 1.2]], dtype=np.float64),
        g1=np.asarray([[-0.1, -0.1, 0.0]], dtype=np.float64),
        ep=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        fp=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
        gp=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
    )
    result = hydrol_alt_residual_solve_step(
        setup=setup,
        mcl_before=np.asarray([[0.20, 0.21, 0.22]], dtype=np.float64),
        mc_before=np.asarray([[0.20, 0.21, 0.22]], dtype=np.float64),
        profil_froz=np.zeros((1, 3), dtype=np.float64),
        mcr=np.asarray([0.057], dtype=np.float64),
        b=np.zeros((1, 3), dtype=np.float64),
        dz_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        dt_days=1.0,
        free_drain_coef=np.asarray([1.0], dtype=np.float64),
        flux_top=np.asarray([0.0], dtype=np.float64),
        resolv=np.asarray([True]),
        mask_soiltile=np.asarray([1.0], dtype=np.float64),
        k_bottom=np.asarray([2.0], dtype=np.float64),
    )

    assert not np.asarray(result.resolv_alt)[0]
    np.testing.assert_allclose(np.asarray(result.second.mcl), np.asarray(result.first.mcl))
