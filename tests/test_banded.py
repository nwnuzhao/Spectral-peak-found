"""Tests for the banded linear algebra behind the Whittaker baselines."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks._banded import (
    banded_from_dense,
    banded_to_dense,
    difference_penalty_bands,
    ldl_factor_banded,
    solve_banded,
    solve_difference_system,
)


def _dense_penalty(n: int, order: int) -> np.ndarray:
    if order == 0:
        return np.eye(n)
    d = np.diff(np.eye(n), n=order, axis=0)
    return d.T @ d


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 17, 64, 101])
def test_penalty_bands_match_dense_construction(n, order):
    if order >= n:
        pytest.skip("order must be smaller than n")
    p, banded = difference_penalty_bands(n, order)
    assert p == order
    np.testing.assert_allclose(banded_to_dense(banded, p, n), _dense_penalty(n, order), atol=1e-12)


def test_banded_round_trip():
    """Extracting and re-expanding the bands of a banded matrix is lossless."""
    rng = np.random.default_rng(1)
    n, p = 12, 3
    banded = rng.normal(size=(p + 1, n))
    for k in range(1, p + 1):
        banded[k, n - k :] = 0.0  # entries outside the matrix
    dense = banded_to_dense(banded, p, n)
    np.testing.assert_allclose(
        banded_to_dense(banded_from_dense(dense, p), p, n), dense, atol=1e-12
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_solve_banded_matches_dense_solve(order):
    rng = np.random.default_rng(order)
    n = 40
    p, penalty = difference_penalty_bands(n, order)
    weights = rng.uniform(0.1, 2.0, size=n)
    rhs = rng.normal(size=n)
    banded = penalty.copy()
    banded[0] = banded[0] + weights
    expected = np.linalg.solve(_dense_penalty(n, order) + np.diag(weights), rhs)
    np.testing.assert_allclose(solve_banded(banded, p, rhs), expected, rtol=1e-9, atol=1e-9)


def test_ldl_factor_reconstructs_the_matrix():
    rng = np.random.default_rng(3)
    n, p = 15, 2
    _, penalty = difference_penalty_bands(n, p)
    weights = rng.uniform(0.5, 3.0, size=n)
    banded = penalty.copy()
    banded[0] = banded[0] + weights
    d, lb = ldl_factor_banded(banded, p)
    lower = np.eye(n)
    for m in range(1, p + 1):
        for i in range(m, n):
            lower[i, i - m] = lb[m, i]
    reconstructed = lower @ np.diag(d) @ lower.T
    np.testing.assert_allclose(reconstructed, banded_to_dense(banded, p, n), atol=1e-10)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("lam", [1e2, 1e5, 1e8])
def test_solve_difference_system_matches_dense(order, lam):
    rng = np.random.default_rng(4)
    n = 50
    y = rng.normal(size=n)
    weights = rng.uniform(0.2, 1.0, size=n)
    expected = np.linalg.solve(
        np.diag(weights) + lam * _dense_penalty(n, order), weights * y
    )
    np.testing.assert_allclose(
        solve_difference_system(y, weights, lam, order), expected, rtol=1e-8, atol=1e-8
    )


def test_solve_difference_system_satisfies_the_normal_equations():
    rng = np.random.default_rng(5)
    n = 60
    y = rng.normal(size=n)
    weights = rng.uniform(0.3, 1.0, size=n)
    lam, order = 1e4, 2
    z = solve_difference_system(y, weights, lam, order)
    residual = np.diag(weights) @ z + lam * _dense_penalty(n, order) @ z - weights * y
    np.testing.assert_allclose(residual, np.zeros(n), atol=1e-8)


def test_singular_penalty_is_reported():
    p, penalty = difference_penalty_bands(8, 2)  # rank deficient on its own
    with pytest.raises(np.linalg.LinAlgError):
        ldl_factor_banded(penalty, p)


def test_large_problem_stays_tractable():
    rng = np.random.default_rng(6)
    n = 20000
    y = np.sin(np.linspace(0.0, 30.0, n)) + 0.1 * rng.normal(size=n)
    weights = np.ones(n)
    z = solve_difference_system(y, weights, 1e5, 2)
    assert z.shape == y.shape
    assert np.all(np.isfinite(z))
    # the smoothed solution must be far smoother than the input
    assert np.abs(np.diff(z, 2)).max() < np.abs(np.diff(y, 2)).max()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        difference_penalty_bands(10, 10)
    with pytest.raises(ValueError):
        difference_penalty_bands(0, 0)
    with pytest.raises(ValueError):
        solve_difference_system(np.zeros(5), np.ones(4), 1e3, 2)  # length mismatch
    with pytest.raises(ValueError):
        solve_difference_system(np.zeros(5), np.ones(5), -1.0, 2)
    with pytest.raises(ValueError):
        banded_from_dense(np.ones((3, 4)), 1)
