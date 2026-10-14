"""Tests for :mod:`spectral_peaks.baseline`."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks.baseline import (
    BASELINE_METHODS,
    airpls_baseline,
    als_baseline,
    correct_baseline,
    iterative_polynomial_baseline,
    morphological_baseline,
    rolling_ball_baseline,
    snip_baseline,
)

from helpers import exponential_background, gaussian, make_spectrum

PEAKS = [(520.0, 1.0, 24.0), (700.0, 0.6, 30.0), (905.0, 1.6, 22.0), (1300.0, 0.9, 45.0)]


def _peak_free_mask(x, half: float = 45.0) -> np.ndarray:
    mask = np.ones_like(x, dtype=bool)
    for center, _, fwhm in PEAKS:
        mask &= np.abs(x - center) > max(half, 1.5 * fwhm)
    return mask


@pytest.fixture(scope="module")
def spectrum():
    return make_spectrum(PEAKS, noise=0.0, background="exponential")


def test_snip_recovers_a_flat_background():
    x = np.arange(300.0)
    baseline = 3.0 + 0.004 * x
    y = baseline + gaussian(x, 150.0, 2.0, 12.0) + gaussian(x, 40.0, 1.2, 8.0)
    estimate = snip_baseline(y, iterations=15)
    assert np.all(estimate <= y + 1e-9)  # clipped at the signal
    mask = (np.abs(x - 150.0) > 20) & (np.abs(x - 40.0) > 15)
    assert np.abs(estimate[mask] - baseline[mask]).max() < 0.02


def test_snip_handles_negative_signals():
    x = np.arange(300.0)
    baseline = -5.0 + 0.002 * x
    y = baseline + gaussian(x, 150.0, 2.0, 12.0)
    estimate = snip_baseline(y, iterations=15)
    mask = np.abs(x - 150.0) > 20
    assert np.abs(estimate[mask] - baseline[mask]).max() < 0.05


def test_snip_identity_without_passes():
    x = np.arange(100.0)
    y = 1.0 + gaussian(x, 50.0, 5.0, 10.0)
    assert snip_baseline(y).shape == y.shape
    # no clipping passes and no transform -> the input itself
    np.testing.assert_allclose(snip_baseline(y, iterations=0, transform=False), y)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("airpls", {}),
        ("als", {}),
        ("snip", {"iterations": 40}),
        ("morphological", {"half_window": 60}),
        ("rolling_ball", {"radius": 25}),
        ("polynomial", {"order": 4}),
    ],
)
def test_every_baseline_method_recovers_peak_heights(spectrum, method, kwargs):
    x, y, truth = spectrum
    corrected = correct_baseline(y, method=method, **kwargs)
    for peak in truth:
        index = int(np.argmin(np.abs(x - peak["center"])))
        recovered = corrected[index]
        assert recovered == pytest.approx(peak["height"], rel=0.25), (
            f"{method} recovered {recovered:.3f} instead of {peak['height']:.3f}"
        )


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("airpls", {}),
        ("als", {}),
        ("snip", {"iterations": 40}),
        ("rolling_ball", {"radius": 25}),
    ],
)
def test_baseline_is_close_to_the_truth_where_there_are_no_peaks(spectrum, method, kwargs):
    x, y, _ = spectrum
    _, baseline = correct_baseline(y, method=method, return_baseline=True, **kwargs)
    truth_baseline = exponential_background(x)
    mask = _peak_free_mask(x)
    error = np.abs(baseline[mask] - truth_baseline[mask]).max()
    assert error < 0.12, f"{method} deviates by {error:.3f}"


def test_airpls_improves_on_no_correction(spectrum):
    x, y, _ = spectrum
    mask = _peak_free_mask(x)
    before = np.abs(y[mask]).mean()  # the background itself is the error before
    after = np.abs((y - airpls_baseline(y, lam=1e5))[mask]).mean()
    assert before > 0.3  # the background is not trivial
    assert after < 0.1 * before
    assert after < 0.02


def test_airpls_reports_convergence_information(spectrum):
    _, y, _ = spectrum
    baseline, info = airpls_baseline(y, lam=1e5, return_info=True)
    assert info["lam"] == 1e5 and info["order"] == 2
    assert 1 <= info["iterations"] <= 100
    assert info["weights"].shape == y.shape
    assert np.all(info["weights"] >= 0.0)
    assert baseline.shape == y.shape


def test_als_weight_controls_the_baseline_level(spectrum):
    _, y, _ = spectrum
    low = als_baseline(y, lam=1e6, p=1e-4)
    high = als_baseline(y, lam=1e6, p=0.5)
    assert low.mean() < high.mean()


def test_smooth_methods_are_much_smoother_than_the_spectrum(spectrum):
    _, y, _ = spectrum
    roughness_y = np.abs(np.diff(y, 2)).sum()
    for method in ("airpls", "als", "polynomial"):
        baseline = correct_baseline(y, method=method, return_baseline=True)[1]
        assert np.abs(np.diff(baseline, 2)).sum() < 0.02 * roughness_y


def test_clipping_methods_stay_below_the_signal(spectrum):
    _, y, _ = spectrum
    for method, kwargs in [
        ("snip", {"iterations": 40}),
        ("morphological", {"half_window": 60}),
        ("rolling_ball", {"radius": 60, "smoothing": 11}),
    ]:
        baseline = correct_baseline(y, method=method, return_baseline=True, **kwargs)[1]
        assert np.all(baseline <= y + 1e-9), method


def test_polynomial_baseline_follows_a_curved_background():
    x = np.linspace(0.0, 100.0, 501)
    background = 0.001 * (x - 50.0) ** 2 + 0.5
    y = background + gaussian(x, 20.0, 1.0, 5.0) + gaussian(x, 80.0, 0.8, 7.0)
    baseline = iterative_polynomial_baseline(y, order=2)
    mask = (np.abs(x - 20.0) > 10) & (np.abs(x - 80.0) > 14)
    assert np.abs(baseline[mask] - background[mask]).max() < 0.05


def test_morphological_window_controls_what_survives():
    x = np.arange(400.0)
    y = 1.0 + gaussian(x, 200.0, 2.0, 40.0)
    wide = morphological_baseline(y, half_window=80)
    narrow = morphological_baseline(y, half_window=5)
    assert wide[200] < narrow[200]  # a wide window swallows the peak
    assert np.all(wide <= y + 1e-9)


def test_rolling_ball_follows_a_curved_background():
    x = np.linspace(0.0, 100.0, 1001)
    background = 2.0 * np.sin(np.pi * (x - x.min()) / (x.max() - x.min())) ** 2
    y = background + gaussian(x, 30.0, 3.0, 4.0) + gaussian(x, 70.0, 3.0, 4.0)
    baseline = rolling_ball_baseline(y, radius=20)
    mask = (np.abs(x - 30.0) > 10) & (np.abs(x - 70.0) > 10)
    assert np.abs(baseline[mask] - background[mask]).max() < 0.05
    # and it does not swallow the peaks
    assert y[300] - baseline[300] > 0.75 * 3.0


def test_rolling_ball_bias_grows_with_the_radius():
    """Documented behaviour: the ball cannot enter a concave background."""
    x = np.linspace(0.0, 100.0, 1001)
    background = 2.0 * np.sin(np.pi * (x - x.min()) / (x.max() - x.min())) ** 2
    y = background + gaussian(x, 30.0, 3.0, 4.0) + gaussian(x, 70.0, 3.0, 4.0)
    mask = (np.abs(x - 30.0) > 10) & (np.abs(x - 70.0) > 10)
    small = np.abs(rolling_ball_baseline(y, radius=10)[mask] - background[mask]).max()
    large = np.abs(rolling_ball_baseline(y, radius=40)[mask] - background[mask]).max()
    assert small < large


def test_rolling_ball_is_exact_for_a_narrow_peak_on_a_flat_background():
    x = np.arange(400.0)
    y = np.exp(-0.5 * ((x - 200.0) / 5.0) ** 2)
    baseline = rolling_ball_baseline(y, radius=30)
    assert baseline[20] == pytest.approx(0.0, abs=1e-9)
    assert y[200] - baseline[200] == pytest.approx(1.0, rel=1e-3)


def test_morphological_is_accurate_on_a_flat_background():
    x = np.arange(600.0)
    y = 0.4 + gaussian(x, 200.0, 2.0, 24.0) + gaussian(x, 450.0, 1.0, 18.0)
    baseline = morphological_baseline(y, half_window=30)
    mask = (np.abs(x - 200.0) > 60) & (np.abs(x - 450.0) > 45)
    assert np.abs(baseline[mask] - 0.4).max() < 1e-6
    assert y[200] - baseline[200] == pytest.approx(2.0, rel=0.02)
    assert y[450] - baseline[450] == pytest.approx(1.0, rel=0.02)


def test_morphological_bias_equals_half_window_times_slope():
    """Documented limitation: a flat window on a sloped baseline is biased.

    The opening can always be shifted until the window clears the peaks, which
    lifts it by about ``half_window * |slope|``.
    """
    x = np.arange(2000.0)
    slope = -0.002
    truth = 1.5 + slope * x
    y = truth + gaussian(x, 400.0, 1.0, 24.0) + gaussian(x, 1200.0, 1.5, 30.0)
    mask = (np.abs(x - 400.0) > 60) & (np.abs(x - 1200.0) > 60)
    for half_window in (20, 60):
        baseline = morphological_baseline(y, half_window=half_window)
        bias = np.abs(baseline[mask] - truth[mask]).max()
        assert bias == pytest.approx(half_window * abs(slope), rel=0.25)


def test_correct_baseline_return_baseline_consistency(spectrum):
    _, y, _ = spectrum
    corrected, baseline = correct_baseline(y, method="airpls", return_baseline=True)
    np.testing.assert_allclose(corrected, y - baseline, atol=1e-12)


def test_invalid_arguments_raise(spectrum):
    _, y, _ = spectrum
    with pytest.raises(ValueError):
        correct_baseline(y, method="does-not-exist")
    with pytest.raises(ValueError):
        correct_baseline(np.zeros(0), method="snip")  # empty input
    with pytest.raises(ValueError):
        snip_baseline(np.ones((2, 2)))  # not one-dimensional
    with pytest.raises(ValueError):
        als_baseline(y, p=1.5)
    with pytest.raises(ValueError):
        airpls_baseline(y, order=len(y) + 1)
    with pytest.raises(ValueError):
        morphological_baseline(y, half_window=0)
    with pytest.raises(ValueError):
        rolling_ball_baseline(y, radius=len(y))
    with pytest.raises(ValueError):
        iterative_polynomial_baseline(y, order=-1)
    assert set(BASELINE_METHODS) == {
        "airpls",
        "als",
        "snip",
        "morphological",
        "rolling_ball",
        "polynomial",
    }
