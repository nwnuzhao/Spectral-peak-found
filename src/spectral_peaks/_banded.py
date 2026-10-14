"""Banded symmetric linear algebra, implemented with NumPy arrays only.

The Whittaker-type smoothers used for baseline estimation (:func:`airpls_baseline`,
:func:`als_baseline`) repeatedly solve a linear system

    ``(W + lambda * D.T @ D) z = W y``

where ``W = diag(w)`` is diagonal and ``D`` is the ``order``-th difference
operator.  That matrix is symmetric, banded (bandwidth ``p == order``) and
positive definite, so it can be factorised in ``O(n * p**2)`` instead of the
``O(n**3)`` a dense solve would cost.  With ``p == 2`` this turns a multi-second
solve on a 10 000-point spectrum into a few milliseconds.

Storage convention
------------------
A symmetric banded matrix is stored as an ``(p + 1, n)`` array ``B`` with

    ``B[k, i] == A[i, i + k]``  for ``k = 0 .. p``

i.e. row 0 holds the main diagonal and row ``k`` the ``k``-th super-diagonal.
Entries outside the matrix are not stored, so the trailing elements of rows
``k > 0`` may hold anything.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "banded_from_dense",
    "banded_to_dense",
    "difference_penalty_bands",
    "ldl_factor_banded",
    "solve_banded",
    "solve_difference_system",
]


def banded_from_dense(matrix: np.ndarray, p: int) -> np.ndarray:
    """Extract the ``(p + 1, n)`` banded representation of a dense symmetric matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    banded = np.zeros((p + 1, n))
    for k in range(p + 1):
        banded[k, : n - k] = np.diagonal(matrix, offset=k)
    return banded


def banded_to_dense(banded: np.ndarray, p: int, n: int | None = None) -> np.ndarray:
    """Expand the banded representation back into a dense symmetric matrix."""
    banded = np.asarray(banded, dtype=float)
    if n is None:
        n = banded.shape[1]
    dense = np.zeros((n, n))
    for k in range(p + 1):
        if k == 0:
            dense[np.arange(n), np.arange(n)] = banded[0, :n]
        else:
            idx = np.arange(n - k)
            dense[idx, idx + k] = banded[k, : n - k]
            dense[idx + k, idx] = banded[k, : n - k]
    return dense


def difference_penalty_bands(n: int, order: int) -> tuple[int, np.ndarray]:
    """Banded form of ``D.T @ D`` for the ``order``-th difference operator ``D``.

    ``D`` is the ``(n - order, n)`` forward-difference matrix
    ``np.diff(np.eye(n), n=order, axis=0)``; the bandwidth is ``order``.

    Analytic bands are used for ``order`` in ``{0, 1, 2}`` (the values used
    throughout the baseline-correction literature); small problems and higher
    orders fall back to an explicit dense construction, which is cheap there.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if order < 0 or order >= n:
        raise ValueError("order must satisfy 0 <= order < n")
    if order == 0:
        return 0, np.ones((1, n))

    if order == 1 and n >= 3:
        main = np.full((1, n), 2.0)
        main[0, 0] = main[0, -1] = 1.0
        off = np.full((1, n), -1.0)
        off[0, -1] = 0.0
        return 1, np.vstack([main, off])

    if order == 2 and n >= 4:
        main = np.empty((1, n))
        main[0] = 6.0
        main[0, 0] = main[0, -1] = 1.0
        main[0, 1] = main[0, -2] = 5.0
        off1 = np.empty((1, n))
        off1[0] = -4.0
        off1[0, 0] = off1[0, -2] = -2.0
        off1[0, -1] = 0.0
        off2 = np.empty((1, n))
        off2[0] = 1.0
        off2[0, -2:] = 0.0
        return 2, np.vstack([main, off1, off2])

    d = np.diff(np.eye(n), n=order, axis=0)
    penalty = d.T @ d
    return order, banded_from_dense(penalty, order)


def ldl_factor_banded(banded: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """``L D L.T`` factorisation of a symmetric positive-definite banded matrix.

    Returns ``(d, lb)`` where ``d`` has shape ``(n,)`` (the diagonal of ``D``)
    and ``lb`` has shape ``(p + 1, n)`` with ``lb[m, i] == L[i, i - m]``; the
    unit diagonal of ``L`` is implicit and ``lb[0]`` is unused.

    This is the textbook banded LDL^T recurrence -- no pivoting is needed
    because the matrix is positive definite -- costing ``O(n * p**2)``.
    Raises ``numpy.linalg.LinAlgError`` on a non-positive pivot.
    """
    banded = np.asarray(banded, dtype=float)
    n = banded.shape[1]
    d = np.zeros(n)
    lb = np.zeros((p + 1, n))
    for i in range(n):
        # Off-diagonal entries of row i, from the outermost band inwards.
        #   A[i, i-m] = L[i, i-m] * D[i-m]
        #             + sum_t L[i, i-m-t] * D[i-m-t] * L[i-m, i-m-t]
        for m in range(min(p, i), 0, -1):
            j = i - m
            acc = banded[m, j]
            for t in range(1, min(p - m, j) + 1):
                acc -= lb[m + t, i] * d[j - t] * lb[t, j]
            lb[m, i] = acc / d[j]
        # Diagonal entry:  A[i, i] = D[i] + sum_m L[i, i-m]**2 * D[i-m]
        acc = banded[0, i]
        for m in range(1, min(p, i) + 1):
            acc -= lb[m, i] * lb[m, i] * d[i - m]
        if acc <= 0.0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        d[i] = acc
    return d, lb


def _solve_factor(d: np.ndarray, lb: np.ndarray, p: int, b: np.ndarray) -> np.ndarray:
    """Forward/back substitution through a factorisation from :func:`ldl_factor_banded`."""
    n = d.size
    z = np.empty(n)
    for i in range(n):  # L z = b
        acc = b[i]
        for m in range(1, min(p, i) + 1):
            acc -= lb[m, i] * z[i - m]
        z[i] = acc
    z /= d  # D y = z
    x = np.empty(n)
    for i in range(n - 1, -1, -1):  # L.T x = y
        acc = z[i]
        for m in range(1, min(p, n - 1 - i) + 1):
            acc -= lb[m, i + m] * x[i + m]
        x[i] = acc
    return x


def solve_banded(banded: np.ndarray, p: int, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` for a banded, symmetric positive-definite ``A``."""
    b = np.asarray(b, dtype=float)
    d, lb = ldl_factor_banded(banded, p)
    return _solve_factor(d, lb, p, b)


def solve_difference_system(
    y: np.ndarray, weights: np.ndarray, lam: float, order: int
) -> np.ndarray:
    """Solve ``(diag(weights) + lam * D.T @ D) z = weights * y``.

    Inner step of every Whittaker-style baseline smoother.  The factorisation
    is recomputed on each call because the weights change between outer
    iterations; for ``order <= 2`` the cost is ``O(n)`` with a small constant.
    """
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = y.size
    if weights.size != n:
        raise ValueError("weights and y must have the same length")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    p, penalty = difference_penalty_bands(n, order)
    banded = penalty * lam
    banded[0] = banded[0] + weights
    return solve_banded(banded, p, weights * y)
