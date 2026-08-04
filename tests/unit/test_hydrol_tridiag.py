from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (
    hydrol_soil_setup_coefficients,
    hydrol_soil_tridiag_solve,
)


def _dense_tridiag(e: np.ndarray, f: np.ndarray, g1: np.ndarray) -> np.ndarray:
    matrix = np.zeros((f.size, f.size), dtype=np.float64)
    np.fill_diagonal(matrix, f)
    np.fill_diagonal(matrix[1:, :-1], e[1:])
    np.fill_diagonal(matrix[:-1, 1:], g1[:-1])
    return matrix


def test_hydrol_soil_tridiag_solve_matches_numpy_dense_solve():
    e = np.asarray(
        [
            [0.0, -0.40, -0.30, -0.20],
            [0.0, -0.15, -0.25, -0.35],
        ],
        dtype=np.float64,
    )
    f = np.asarray(
        [
            [3.0, 3.5, 3.2, 2.9],
            [2.8, 3.1, 3.3, 3.0],
        ],
        dtype=np.float64,
    )
    g1 = np.asarray(
        [
            [-0.25, -0.35, -0.45, 0.0],
            [-0.20, -0.30, -0.20, 0.0],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray(
        [
            [1.0, 2.0, 1.5, 0.8],
            [0.5, 1.1, 1.7, 2.2],
        ],
        dtype=np.float64,
    )

    result = hydrol_soil_tridiag_solve(e=e, f=f, g1=g1, rhs=rhs)
    actual = np.asarray(result.mcl)
    expected = np.vstack(
        [
            np.linalg.solve(_dense_tridiag(e[0], f[0], g1[0]), rhs[0]),
            np.linalg.solve(_dense_tridiag(e[1], f[1], g1[1]), rhs[1]),
        ]
    )

    assert np.allclose(actual, expected, rtol=0.0, atol=1e-13)
    assert np.allclose(np.asarray(result.gam)[:, 0], 0.0)
    assert np.all(np.isfinite(np.asarray(result.bet)))


def test_hydrol_soil_tridiag_resolv_false_preserves_initial_profile():
    e = np.asarray([[0.0, -0.2, -0.2], [0.0, -0.3, -0.3]], dtype=np.float64)
    f = np.asarray([[2.0, 2.5, 2.0], [2.0, 2.5, 2.0]], dtype=np.float64)
    g1 = np.asarray([[-0.1, -0.1, 0.0], [-0.4, -0.4, 0.0]], dtype=np.float64)
    rhs = np.asarray([[1.0, 0.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float64)
    initial = np.asarray([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float64)

    result = hydrol_soil_tridiag_solve(
        e=e,
        f=f,
        g1=g1,
        rhs=rhs,
        resolv=np.asarray([True, False]),
        initial_mcl=initial,
    )

    solved_first = np.linalg.solve(_dense_tridiag(e[0], f[0], g1[0]), rhs[0])
    actual = np.asarray(result.mcl)
    assert np.allclose(actual[0], solved_first, rtol=0.0, atol=1e-13)
    assert np.array_equal(actual[1], initial[1])


def test_hydrol_soil_tridiag_resolv_false_zero_coefficients_have_finite_gradient():
    zeros = jnp.zeros((1, 3), dtype=jnp.float64)
    initial = jnp.asarray([[6.0, 5.0, 4.0]], dtype=jnp.float64)

    def objective(rhs_offset):
        result = hydrol_soil_tridiag_solve(
            e=zeros,
            f=zeros,
            g1=zeros,
            rhs=zeros + rhs_offset,
            resolv=jnp.asarray([False]),
            initial_mcl=initial,
        )
        return jnp.sum(result.mcl)

    value = jnp.asarray(0.0, dtype=jnp.float64)
    forward = jax.jacfwd(objective)(value)
    reverse = jax.grad(objective)(value)

    assert np.asarray(forward) == 0.0
    assert np.asarray(reverse) == 0.0


def test_hydrol_soil_tridiag_accepts_setup_coefficients_and_residual_is_small():
    a = np.asarray([[0.12, 0.10, 0.08, 0.06]], dtype=np.float64)
    d = np.asarray([[4.0, 5.0, 6.0, 7.0]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 4.0, 8.0, 8.0], dtype=np.float64)
    setup = hydrol_soil_setup_coefficients(
        a=a,
        d=d,
        dz_mm=dz_mm,
        dt_days=1.0 / 48.0,
        free_drain_coef=np.asarray([1.0]),
    )
    e = np.asarray(setup.e)
    f = np.asarray(setup.f)
    g1 = np.asarray(setup.g1)
    rhs = np.asarray([[0.8, 0.7, 0.6, 0.5]], dtype=np.float64)

    result = hydrol_soil_tridiag_solve(e=e, f=f, g1=g1, rhs=rhs)
    mcl = np.asarray(result.mcl)
    residual = _dense_tridiag(e[0], f[0], g1[0]) @ mcl[0] - rhs[0]

    assert mcl.shape == rhs.shape
    assert np.all(np.isfinite(mcl))
    assert np.allclose(residual, 0.0, rtol=0.0, atol=1e-13)
