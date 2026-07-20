"""Full-pivot Gauss-Jordan and passive-pool error from the Fortran source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_STOMATE = 1.0e-8
IPASSIVE_POOL = 7


class GaussJordanError(ValueError):
    """Fatal source failure for an invalid or singular linear system."""


@dataclass(frozen=True)
class GaussJordanResult:
    matrix_a: np.ndarray
    vector_b: np.ndarray


def gauss_jordan_method(matrix_a, vector_b, *, inplace: bool = False) -> GaussJordanResult:
    """Full-pivot source algorithm from lines 61-157.

    The transformed ``vector_b`` is the solution and ``matrix_a`` is the
    inverse. Ties use ``>=`` exactly as Fortran, selecting the last scanned
    candidate. Set ``inplace=True`` to preserve the ``INTENT(inout)`` contract.
    """

    source_a = np.asarray(matrix_a)
    source_b = np.asarray(vector_b)
    if source_a.dtype.kind != "f" or source_b.dtype.kind != "f":
        raise TypeError("matrix_a and vector_b must have floating-point dtype")
    if source_a.ndim != 2 or source_a.shape[0] != source_a.shape[1]:
        raise ValueError("matrix_a must be square")
    n = source_a.shape[0]
    if source_b.shape != (n,):
        raise ValueError("vector_b must have shape (n,)")
    if n == 0:
        raise GaussJordanError("gauss_jordan_method requires n >= 1")
    if inplace:
        if not source_a.flags.writeable or not source_b.flags.writeable:
            raise ValueError("inplace arrays must be writeable")
        a, b = source_a, source_b
    else:
        a, b = source_a.copy(), source_b.copy()

    index_pivot = np.zeros(n, dtype=np.int64)
    index_col = np.empty(n, dtype=np.int64)
    index_row = np.empty(n, dtype=np.int64)
    col = -1
    row = -1
    for i in range(n):
        pivot_max = 0.0
        for j in range(n):
            if index_pivot[j] != 1:
                for k in range(n):
                    if index_pivot[k] == 0:
                        if abs(a[j, k]) >= pivot_max:
                            pivot_max = abs(a[j, k])
                            row = j
                            col = k
        if col == -1:
            raise GaussJordanError("gauss_jordan_method failed to locate a pivot")
        index_pivot[col] += 1
        if row != col:
            for j in range(n):
                temp = a[row, j].copy()
                a[row, j] = a[col, j]
                a[col, j] = temp
            temp = b[row].copy()
            b[row] = b[col]
            b[col] = temp
        index_row[i] = row
        index_col[i] = col
        if a[col, col] == 0.0:
            raise GaussJordanError("the matrix A is not inversible")
        inv_pivot = 1.0 / a[col, col]
        for j in range(n):
            a[col, j] *= inv_pivot
        b[col] *= inv_pivot
        for ii in range(n):
            if ii != col:
                temp = a[ii, col].copy()
                a[ii, col] = 0.0
                for jj in range(n):
                    a[ii, jj] -= a[col, jj] * temp
                b[ii] -= b[col] * temp
    for j in range(n - 1, -1, -1):
        if index_row[j] != index_col[j]:
            for i in range(n):
                temp = a[i, index_row[j]].copy()
                a[i, index_row[j]] = a[i, index_col[j]]
                a[i, index_col[j]] = temp
    return GaussJordanResult(a, b)


def error_l1_passive(
    current_value,
    previous_value,
    veget_max,
    criterion: float,
    *,
    ipassive_pool: int = IPASSIVE_POOL,
    min_stomate: float = MIN_STOMATE,
) -> np.ndarray:
    """Passive-pool convergence flags from source lines 177-241.

    ``ipassive_pool`` retains its Fortran 1-based meaning.
    """

    current = np.asarray(current_value)
    previous = np.asarray(previous_value)
    vegetation = np.asarray(veget_max)
    if current.ndim != 3 or current.shape != previous.shape:
        raise ValueError("current_value and previous_value must share shape (npts,nb_veget,nb_pools)")
    if vegetation.shape != current.shape[:2]:
        raise ValueError("veget_max must have shape (npts,nb_veget)")
    if ipassive_pool < 1 or ipassive_pool > current.shape[2]:
        raise IndexError("ipassive_pool is outside the pool dimension")
    dtype = np.result_type(current, previous, vegetation, np.float64)
    previous_stock = np.zeros(current.shape[0], dtype=dtype)
    current_stock = np.zeros(current.shape[0], dtype=dtype)
    pool = ipassive_pool - 1
    for vegetation_index in range(current.shape[1]):
        previous_stock += previous[:, vegetation_index, pool] * vegetation[:, vegetation_index]
    for vegetation_index in range(current.shape[1]):
        current_stock += current[:, vegetation_index, pool] * vegetation[:, vegetation_index]
    difference = current_stock - previous_stock
    error_global = np.empty_like(previous_stock)
    relative = previous_stock > min_stomate
    error_global[relative] = 100.0 * np.abs(difference[relative]) / previous_stock[relative]
    error_global[~relative] = np.abs(difference[~relative])
    return error_global <= criterion
