"""Tests for :mod:`spectral_peaks.smoothing`."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks.smoothing import (
    gaussian_kernel,
    gaussian_smoothing,
    moving_average,
    savgol_coeffs,
    savgol_filter,
)


def test_savgol_coefficients_match_textbook_values():
    """Classic hand-computed Savitzky-Golay filters."""
    np.testing.assert_allclose(
        savgol_coeffs(5, 2), np.array([-3, 12, 17, 12, -3]) / 35.0, atol=1e-14
    )
    np.testing.assert_allclose(
        savgol_coeffs(5, 2, deriv=1), np.array([-2, -1, 0, 1, 2]) / 10.0, atol=1e-14
    )
    np.testing.assert_allclose(
        savgol_coeffs(7, 2, deriv=2), np.array([5, 0, -3, -4, -3, 0, 5]) / 42.0, atol=1e-13
    )
    # a smoothing filter always sums to 1; a derivative filter to 0
    assert savgol_coeffs(11, 3).sum() == pytest.approx(1.0)
    assert savgol_coeffs(11, 3, deriv=1).sum() == pytest.approx(0.0, abs=1e-12)


def test_savgol_coefficients_match_scipy():
    """Cross-check against SciPy's filter design.

    ``scipy.signal.savgol_coeffs`` defaults to ``use="conv"``, i.e. coefficients
    laid out for a linear convolution; this implementation uses the equivalent
    ``use="dot"`` convention (``out[i] = sum_j c[j] * y[i + j - half]``).  The
    two differ by a kernel reversal, which is visible as a sign flip for odd
    derivative orders.
    """
    sps = pytest.importorskip("scipy.signal")
    for window, order in [(5, 2), (7, 3), (11, 2), (9, 5), (21, 4)]:
        for deriv in range(0, order + 1):
            try:
                expected = sps.savgol_coeffs(window, order, deriv=deriv, use="dot")
            except TypeError:  # SciPy < 1.15 exposes only one convention
                expected = sps.savgol_coeffs(window, order, deriv=deriv)
                if deriv % 2 == 1:
                    expected = -expected
            np.testing.assert_allclose(
                savgol_coeffs(window, order, deriv=deriv), expected, atol=1e-10
            )


def test_savgol_filter_matches_scipy_interior():
    sps = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(0)
    y = rng.normal(size=200) + 3.0
    for window, order, deriv in [(11, 2, 0), (11, 3, 1), (21, 4, 2), (5, 4, 0)]:
        ours = savgol_filter(y, window, order, deriv=deriv)
        theirs = sps.savgol_filter(y, window, order, deriv=deriv)
        half = window // 2
        np.testing.assert_allclose(ours[half:-half], theirs[half:-half], atol=1e-9)


def test_savgol_preserves_constants_and_linear_derivatives():
    y = np.full(50, 2.5)
    np.testing.assert_allclose(savgol_filter(y, 9, 3), y, atol=1e-12)
    # Derivatives are exact in the interior; at the ends the reflect padding
    # decides what "outside" means, so only the interior is compared.
    x = np.arange(50, dtype=float)
    np.testing.assert_allclose(savgol_filter(x, 9, 2, deriv=1)[4:-4], 1.0, atol=1e-9)
    np.testing.assert_allclose(savgol_filter(3.0 * x, 11, 3, deriv=1)[5:-5], 3.0, atol=1e-9)
    # second derivative of a quadratic is its constant curvature
    np.testing.assert_allclose(
        savgol_filter(x**2, 11, 3, deriv=2)[5:-5], np.full(40, 2.0), atol=1e-7
    )


def test_savgol_delta_scales_derivatives():
    """A derivative in sample units is converted by dividing by the spacing."""
    x = np.linspace(0.0, 10.0, 101)
    y = np.sin(x / 2.0)
    delta = x[1] - x[0]
    physical = savgol_filter(y, 11, 3, deriv=1, delta=delta)
    sample = savgol_filter(y, 11, 3, deriv=1)
    np.testing.assert_allclose(physical[5:-5], (sample / delta)[5:-5], atol=1e-10)
    # and it agrees with the analytic derivative dy/dX = 0.5 * cos(X / 2)
    np.testing.assert_allclose(
        physical[5:-5], (0.5 * np.cos(x[5:-5] / 2.0)), atol=2e-4
    )


def test_savgol_output_shape_and_edge_modes():
    y = np.arange(30, dtype=float)
    for mode in ("reflect", "symmetric", "edge", "wrap", "constant"):
        assert savgol_filter(y, 7, 2, mode=mode).shape == y.shape
    # edge padding must not introduce NaNs even for very short signals
    assert np.all(np.isfinite(savgol_filter(np.arange(3.0), 3, 1)))


def test_moving_average_properties():
    y = np.ones(40) * 4.0
    np.testing.assert_allclose(moving_average(y, 5), y, atol=1e-12)
    np.testing.assert_allclose(moving_average(y, 4), y, atol=1e-12)  # even windows
    ramp = np.arange(40, dtype=float)
    assert moving_average(ramp, 5).mean() == pytest.approx(ramp.mean(), abs=1e-9)
    assert moving_average(ramp, 1).tolist() == ramp.tolist()


def test_gaussian_smoothing_properties():
    y = np.zeros(101)
    y[50] = 1.0
    kernel = gaussian_kernel(2.0)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel.size % 2 == 1
    smoothed = gaussian_smoothing(y, 2.0)
    assert smoothed.size == y.size
    assert smoothed[50] == smoothed.max()
    assert smoothed[50] < 1.0  # the delta is spread out
    np.testing.assert_allclose(smoothed, smoothed[::-1], atol=1e-12)  # symmetric
    assert smoothed.sum() == pytest.approx(1.0, rel=1e-6)  # mass preserved


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        savgol_coeffs(6, 2)  # even window
    with pytest.raises(ValueError):
        savgol_coeffs(5, 5)  # polyorder >= window
    with pytest.raises(ValueError):
        savgol_coeffs(5, 2, deriv=3)  # deriv > polyorder
    with pytest.raises(ValueError):
        savgol_filter(np.arange(10.0), 11, 2)  # signal shorter than window
    with pytest.raises(ValueError):
        moving_average(np.arange(10.0), 0)
    with pytest.raises(ValueError):
        gaussian_kernel(0.0)
    with pytest.raises(ValueError):
        savgol_filter(np.arange(20.0), 7, 2, mode="not-a-mode")
