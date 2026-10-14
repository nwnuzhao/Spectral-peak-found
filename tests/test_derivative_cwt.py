"""Tests for the derivative (Savitzky-Golay) and wavelet peak detectors."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks.peaks import cwt, find_peaks, find_peaks_cwt, find_peaks_derivative, ricker

from helpers import gaussian, make_spectrum, match_peaks

WIDTHS = np.arange(2, 26)
SPECTRUM = [
    (520.0, 1.0, 24.0),
    (700.0, 0.6, 30.0),
    (905.0, 1.6, 22.0),
    (1300.0, 0.9, 45.0),
    (1700.0, 1.2, 36.0),
    (2100.0, 0.5, 20.0),
]


# --------------------------------------------------------------------------- #
# Ricker wavelet and CWT
# --------------------------------------------------------------------------- #
def test_ricker_wavelet_properties():
    scale = 8.0
    wavelet = ricker(101, scale)
    assert wavelet.shape == (101,)
    assert np.allclose(wavelet, wavelet[::-1])  # symmetric
    assert wavelet[50] == pytest.approx(2.0 / (np.sqrt(3.0 * scale) * np.pi**0.25))
    assert wavelet.sum() == pytest.approx(0.0, abs=1e-6)  # zero mean
    # the wavelet is a second derivative: it vanishes exactly at +-scale
    np.testing.assert_allclose(
        np.nonzero(np.abs(wavelet) < 1e-12)[0] - 50, [-scale, scale]
    )


def test_ricker_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        ricker(0, 1.0)
    with pytest.raises(ValueError):
        ricker(11, 0.0)


def test_cwt_shapes_and_zeroth_moment():
    y = np.sin(np.linspace(0.0, 10.0, 600))
    assert cwt(y, [1, 4, 9]).shape == (3, 600)
    for bad_widths in ([], [0.5], [1.0, -3.0]):
        with pytest.raises(ValueError):
            cwt(y, bad_widths)
    with pytest.raises(ValueError):
        cwt(np.ones((5, 5)), [3])

    # the wavelet has zero mean *and* zero first moment (it is even-symmetric)
    offsets = np.arange(101) - 50.0
    wavelet = ricker(101, 8.0)
    assert wavelet.sum() == pytest.approx(0.0, abs=1e-6)
    assert float(np.sum(offsets * wavelet)) == pytest.approx(0.0, abs=1e-12)

    # consequently a constant signal transforms to ~0 in the interior; the
    # residual is the sampling error of the wavelet tails (~1e-4 of its amplitude)
    interior = cwt(np.ones(600), np.arange(1, 11))[:, 120:-120]
    assert np.abs(interior).max() < 1e-3


def test_cwt_scale_is_calibrated_to_the_peak_width():
    """The Ricker response of a Gaussian peaks at a scale close to its FWHM."""
    x = np.arange(1201.0)
    widths = np.arange(1, 61)
    best = []
    for fwhm in (10.0, 20.0, 40.0):
        response = np.abs(cwt(gaussian(x, 600.0, 1.0, fwhm), widths)).max(axis=1)
        best.append(float(widths[int(np.argmax(response))]))
    np.testing.assert_allclose(best, [10.0, 19.0, 38.0], rtol=0.1)
    assert best[1] / best[0] == pytest.approx(2.0, rel=0.15)


# --------------------------------------------------------------------------- #
# Derivative detector
# --------------------------------------------------------------------------- #
def test_derivative_matches_peak_indices_and_positions():
    x = np.arange(1501.0)
    y = (
        gaussian(x, 300.0, 1.0, 30.0)
        + gaussian(x, 700.0, 2.0, 60.0)
        + gaussian(x, 1200.0, 0.7, 18.0)
    )
    peaks, properties = find_peaks_derivative(y, smooth_window=11, smooth_polyorder=3)
    np.testing.assert_array_equal(peaks, [300, 700, 1200])
    # heights are read from the smoothed signal, so narrow peaks are slightly damped
    np.testing.assert_allclose(properties["peak_heights"], [1.0, 2.0, 0.7], rtol=5e-3)
    # the first derivative is zero and the second one negative at the maxima
    assert np.all(np.abs(properties["slopes"]) < 1e-9)
    assert np.all(properties["curvatures"] < 0.0)


def test_derivative_second_criterion_gives_the_same_peaks():
    x = np.arange(1501.0)
    y = gaussian(x, 400.0, 1.0, 30.0) + gaussian(x, 900.0, 1.0, 60.0)
    peaks, properties = find_peaks_derivative(
        y, criterion="second", smooth_window=21, smooth_polyorder=3
    )
    np.testing.assert_array_equal(peaks, [400, 900])
    np.testing.assert_allclose(properties["positions"], [400.0, 900.0], atol=1e-6)


def test_derivative_resolves_a_blended_pair_that_find_peaks_merges():
    x = np.arange(1001.0)
    y = gaussian(x, 490.0, 1.0, 40.0) + gaussian(x, 530.0, 0.85, 40.0)
    assert find_peaks(y, prominence=0.05)[0].size == 1  # only one local maximum
    first, props_first = find_peaks_derivative(y, smooth_window=11)
    second, _ = find_peaks_derivative(y, criterion="second", smooth_window=31)
    assert first.size == 2 and second.size == 2
    for position in list(props_first["positions"]) + list(second.astype(float)):
        assert min(abs(position - 490.0), abs(position - 530.0)) < 6.0


def test_derivative_reports_sub_sample_positions():
    x = np.arange(1001.0)
    y = gaussian(x, 500.37, 1.0, 30.0)
    peaks, properties = find_peaks_derivative(y, smooth_window=15, smooth_polyorder=3)
    assert peaks.size == 1 and peaks[0] == 501
    assert float(properties["positions"][0]) == pytest.approx(500.37, abs=1e-3)


def test_derivative_filters_are_applied():
    x = np.arange(2001.0)
    y = (
        gaussian(x, 300.0, 1.0, 30.0)
        + gaussian(x, 1000.0, 0.2, 30.0)
        + gaussian(x, 1500.0, 0.9, 30.0)
    )
    assert find_peaks_derivative(y)[0].size == 3
    np.testing.assert_array_equal(find_peaks_derivative(y, height=0.5)[0], [300, 1500])
    np.testing.assert_array_equal(find_peaks_derivative(y, prominence=0.5)[0], [300, 1500])
    assert find_peaks_derivative(y, distance=2000)[0].size == 1


def test_derivative_invalid_criterion_raises():
    with pytest.raises(ValueError):
        find_peaks_derivative(np.arange(100.0), criterion="third")
    with pytest.raises(ValueError):
        find_peaks_derivative(np.ones((4, 4)))


# --------------------------------------------------------------------------- #
# Wavelet detector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("noise", [0.0, 0.005, 0.01, 0.02])
def test_cwt_finds_every_peak_without_false_positives(noise):
    x, y, truth = make_spectrum(SPECTRUM, noise=noise, background="exponential", seed=3)
    peaks, properties = find_peaks_cwt(y, widths=WIDTHS)
    matched, false_positives, max_error = match_peaks(
        x[peaks], [t["center"] for t in truth], 4.0
    )
    assert matched == len(truth)
    assert false_positives == 0
    assert max_error <= 2.0
    assert properties["scales"].size == peaks.size
    assert np.all(properties["ridge_length"] >= 6)


def test_cwt_threshold_trades_sensitivity_for_false_positives():
    x, y, truth = make_spectrum(SPECTRUM, noise=0.05, background="exponential", seed=1)
    strict = find_peaks_cwt(y, widths=WIDTHS, snr=6.0)[0]
    loose = find_peaks_cwt(y, widths=WIDTHS, snr=1.5)[0]
    assert loose.size >= strict.size
    assert strict.size <= 8


def test_cwt_min_distance_merges_candidates():
    x, y, _ = make_spectrum(SPECTRUM, noise=0.0, background="exponential", seed=3)
    merged = find_peaks_cwt(y, widths=WIDTHS, min_distance=40)[0]
    assert merged.size <= find_peaks_cwt(y, widths=WIDTHS)[0].size


def test_cwt_invalid_arguments():
    with pytest.raises(ValueError):
        find_peaks_cwt(np.arange(4.0))
    with pytest.raises(ValueError):
        find_peaks_cwt(np.arange(100.0), widths=[0.0, 1.0])
    with pytest.raises(ValueError):
        find_peaks_cwt(np.ones((5, 5)))
