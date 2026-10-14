"""Tests for :mod:`spectral_peaks.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks.metrics import (
    _FWHM_OVER_SIGMA,
    estimate_noise,
    fwhm_from_sigma,
    peak_area,
    peak_areas,
    sigma_from_fwhm,
    snr,
    trapezoid,
)

from helpers import gaussian, make_spectrum


def test_trapezoid_matches_analytic_integrals():
    x = np.linspace(0.0, 1.0, 1001)
    assert trapezoid(np.ones_like(x), x) == pytest.approx(1.0)
    assert trapezoid(x, x) == pytest.approx(0.5)
    assert trapezoid(np.ones(5)) == pytest.approx(4.0)  # unit spacing
    assert trapezoid(np.array([3.0])) == 0.0


def test_trapezoid_matches_numpy():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    x = np.cumsum(rng.uniform(0.5, 2.0, size=50))
    reference = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    assert trapezoid(y, x) == pytest.approx(float(reference(y, x)))


def test_trapezoid_rejects_mismatched_axes():
    with pytest.raises(ValueError):
        trapezoid(np.zeros(5), np.zeros(4))


def test_fwhm_sigma_conversions():
    assert _FWHM_OVER_SIGMA == pytest.approx(2.3548200450309493)
    assert fwhm_from_sigma(1.0) == pytest.approx(2.3548200450309493)
    for sigma in (0.5, 1.0, 7.3):
        assert sigma_from_fwhm(fwhm_from_sigma(sigma)) == pytest.approx(sigma)


@pytest.mark.parametrize("method", ["diff", "mad", "residual"])
def test_noise_estimate_on_pure_noise(method):
    rng = np.random.default_rng(1)
    sigma = 0.03
    y = rng.normal(scale=sigma, size=20000)
    estimate = estimate_noise(y, method=method)
    assert estimate == pytest.approx(sigma, rel=0.15)


@pytest.mark.parametrize("method", ["diff", "residual"])
def test_noise_estimate_is_robust_to_peaks(method):
    rng = np.random.default_rng(2)
    sigma = 0.02
    x = np.arange(4000.0)
    signal = 1.5 * np.exp(-x / 800.0)
    for center, height in [(500.0, 2.0), (900.0, 3.0), (2600.0, 4.0)]:
        signal += gaussian(x, center, height, 20.0)
    y = signal + rng.normal(scale=sigma, size=x.size)
    assert estimate_noise(y, method=method) == pytest.approx(sigma, rel=0.3)


def test_noise_estimators_differ_on_a_wide_peak():
    """A wide peak biases the naive MAD estimator but not the difference one."""
    x = np.linspace(0.0, 1000.0, 5001)
    y = gaussian(x, 500.0, 50.0, 300.0)
    diff = estimate_noise(y, method="diff")
    mad = estimate_noise(y, method="mad")
    assert diff < 0.05 * mad
    assert diff < 0.05


def test_noise_estimate_ignores_a_smooth_background():
    rng = np.random.default_rng(3)
    x = np.arange(3000.0)
    sigma = 0.05
    y = 1.5 * np.exp(-x / 600.0) + rng.normal(scale=sigma, size=x.size)
    assert estimate_noise(y, method="diff") == pytest.approx(sigma, rel=0.15)


def test_noise_estimate_of_short_or_flat_signals():
    assert estimate_noise(np.zeros(2)) == 0.0
    assert estimate_noise(np.zeros(100)) == 0.0
    with pytest.raises(ValueError):
        estimate_noise(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        estimate_noise(np.zeros(100), method="nope")


def test_snr_values_and_edge_cases():
    assert snr(3.0, 1.5) == pytest.approx(2.0)
    assert snr(1.0, 0.0) == float("inf")
    ratios = snr(np.array([1.0, 2.0, 4.0]), 2.0)
    np.testing.assert_allclose(ratios, [0.5, 1.0, 2.0])
    with pytest.raises(ValueError):
        snr(1.0, -1.0)


def test_peak_area_of_a_gaussian_matches_the_analytic_value():
    """``peak_area`` takes *sample positions*, so x units must be converted."""
    dx = 0.05
    x = np.arange(2001.0) * dx
    height, fwhm = 2.0, 20.0
    y = gaussian(x, 50.0, height, fwhm)
    sigma = sigma_from_fwhm(fwhm)  # in x units
    sigma_samples = sigma / dx
    center_index = 1000.0
    analytic = height * sigma * np.sqrt(2.0 * np.pi)
    area = peak_area(
        y, center_index - 5.0 * sigma_samples, center_index + 5.0 * sigma_samples, x=x
    )
    assert area == pytest.approx(analytic, rel=0.01)


def test_peak_area_subtracts_the_line_between_the_feet():
    x = np.arange(101.0)
    y = 1.0 + 0.5 * x  # pure sloping baseline -> zero area
    assert peak_area(y, 20.0, 60.0, x=x) == pytest.approx(0.0, abs=1e-9)
    assert peak_area(y, 20.0, 60.0, x=x, subtract_baseline=False) == pytest.approx(
        trapezoid(y[20:61], x[20:61])
    )


def test_peak_area_of_a_triangle_is_exact():
    y = np.zeros(101)
    y[50] = 2.0
    y[49] = y[51] = 1.0
    # a triangle of height 2 and base 4 samples -> area 4
    assert peak_area(y, 48.0, 52.0) == pytest.approx(4.0)


def test_peak_area_handles_degenerate_ranges():
    y = np.array([1.0, 2.0, 3.0])
    assert peak_area(y, 1.0, 1.0) == 0.0  # empty range
    # the order of the two feet does not matter, and a linear signal has no area
    assert peak_area(y, 2.0, 0.0) == peak_area(y, 0.0, 2.0) == 0.0


def test_peak_areas_matches_the_scalar_version():
    x, y, _ = make_spectrum([(520.0, 1.0, 24.0), (905.0, 1.6, 22.0)], noise=0.0)
    left = np.array([450.0, 850.0])
    right = np.array([590.0, 960.0])
    areas = peak_areas(y, left, right, x=x)
    for k in range(2):
        assert areas[k] == pytest.approx(peak_area(y, left[k], right[k], x=x))
    with pytest.raises(ValueError):
        peak_areas(y, left, right[:1], x=x)
