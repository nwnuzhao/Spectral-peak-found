"""Tests for the detectors in :mod:`spectral_peaks.peaks`."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks.peaks import find_peaks, peak_prominences, peak_widths

from helpers import gaussian


# --------------------------------------------------------------------------- #
# Local-maxima conventions
# --------------------------------------------------------------------------- #
def test_monotonic_signals_have_no_peaks():
    assert find_peaks(np.arange(10.0))[0].size == 0
    assert find_peaks(np.arange(10.0)[::-1])[0].size == 0
    assert find_peaks(np.full(10, 3.0))[0].size == 0  # constant
    assert find_peaks(np.array([1.0, 2.0]))[0].size == 0
    assert find_peaks(np.array([2.0, 1.0]))[0].size == 0
    assert find_peaks(np.array([5.0]))[0].size == 0
    assert find_peaks(np.array([]))[0].size == 0


def test_plateaus_report_their_midpoint():
    peaks, props = find_peaks(np.array([1.0, 3.0, 3.0, 1.0]), plateau_size=1)
    np.testing.assert_array_equal(peaks, [1])
    np.testing.assert_array_equal(props["plateau_sizes"], [2])
    np.testing.assert_array_equal(props["left_edges"], [1])
    np.testing.assert_array_equal(props["right_edges"], [2])
    # odd-length plateau -> exact centre
    np.testing.assert_array_equal(find_peaks(np.array([0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.0]))[0], [3])
    # a plateau touching an array end is not a peak
    assert find_peaks(np.array([3.0, 3.0, 1.0, 3.0, 3.0]))[0].size == 0


def test_plateau_size_filter():
    y = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 3.0, 1.0])
    peaks, props = find_peaks(y, plateau_size=(1, 1))
    np.testing.assert_array_equal(peaks, [7])  # only the single-sample peak
    np.testing.assert_array_equal(props["plateau_sizes"], [1])
    peaks, _ = find_peaks(y, plateau_size=(2, None))
    np.testing.assert_array_equal(peaks, [2])  # only the two-sample plateau


def test_flat_top_between_two_peaks_is_not_a_local_maximum():
    y = np.array([0.0, 2.0, 1.0, 2.0, 0.0])
    np.testing.assert_array_equal(find_peaks(y)[0], [1, 3])


# --------------------------------------------------------------------------- #
# Prominence and width, on a hand-computable example
# --------------------------------------------------------------------------- #
def test_prominences_and_widths_on_the_documented_example():
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0])
    peaks, props = find_peaks(y, prominence=0)
    np.testing.assert_array_equal(peaks, [2, 6])
    np.testing.assert_allclose(props["prominences"], [2.0, 3.0])
    np.testing.assert_array_equal(props["left_bases"], [0, 4])
    np.testing.assert_array_equal(props["right_bases"], [4, 8])

    widths, heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
    np.testing.assert_allclose(widths, [2.0, 1.5])
    np.testing.assert_allclose(heights, [1.0, 1.5])
    np.testing.assert_allclose(left_ips, [1.0, 5.25])
    np.testing.assert_allclose(right_ips, [3.0, 6.75])


def test_prominence_is_never_negative_and_zero_for_redundant_peaks():
    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    prominences, left, right = peak_prominences(y, [1, 3, 5])
    assert np.all(prominences >= 0.0)
    np.testing.assert_allclose(prominences, [1.0, 2.0, 3.0])
    # the *last* sample attaining the minimum of each search interval is the base
    np.testing.assert_array_equal(left, [0, 2, 4])
    np.testing.assert_array_equal(right, [2, 4, 6])


def test_wlen_limits_how_far_the_bases_are_sought():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert peak_prominences(y, [5])[0][0] == pytest.approx(5.0)
    assert peak_prominences(y, [5], wlen=5)[0][0] == pytest.approx(2.0)
    assert peak_prominences(y, [5], wlen=3)[0][0] == pytest.approx(1.0)


def test_rel_height_controls_the_width_contour():
    """``rel_height`` counts *down* from the peak, in units of the prominence."""
    x = np.arange(601.0)
    fwhm = 40.0
    sigma = fwhm / 2.3548200450309493
    y = gaussian(x, 300.0, 1.0, fwhm)
    peaks = find_peaks(y)[0]
    half = peak_widths(y, peaks, rel_height=0.5)[0][0]
    quarter = peak_widths(y, peaks, rel_height=0.25)[0][0]
    near_base = peak_widths(y, peaks, rel_height=0.95)[0][0]
    assert quarter < half < near_base
    assert half == pytest.approx(fwhm, rel=0.02)
    # analytic widths of a Gaussian at 1 - rel_height of its height
    assert quarter == pytest.approx(2.0 * sigma * np.sqrt(2.0 * np.log(1.0 / 0.75)), rel=0.01)
    assert near_base == pytest.approx(2.0 * sigma * np.sqrt(2.0 * np.log(1.0 / 0.05)), rel=0.01)


# --------------------------------------------------------------------------- #
# Cross-validation against SciPy
# --------------------------------------------------------------------------- #
FILTER_SETS = [
    {},
    {"prominence": 0.2},
    {"height": 0.5},
    {"height": (-1.0, 1.0)},
    {"distance": 5},
    {"threshold": 0.1},
    {"width": (0.0, 5.0)},
    {"plateau_size": (1, 3)},
    {"wlen": 10, "prominence": 0.1},
]


@pytest.mark.parametrize("kwargs", FILTER_SETS)
def test_find_peaks_matches_scipy(kwargs):
    """The same peaks *and* the same property arrays as scipy.signal.find_peaks."""
    sps = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(0)
    for trial in range(25):
        n = int(rng.integers(20, 250))
        y = rng.normal(size=n)
        if trial % 3 == 0:
            y = np.round(y, 1)  # introduces plateaus and height ties
        if trial % 5 == 0:
            y = np.convolve(y, np.ones(5) / 5.0, mode="same")
        peaks, properties = find_peaks(y, **kwargs)
        expected_peaks, expected_properties = sps.find_peaks(y, **kwargs)
        np.testing.assert_array_equal(peaks, expected_peaks)
        for key, value in expected_properties.items():
            np.testing.assert_allclose(properties[key], value, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rel_height", [0.5, 0.25, 0.9])
def test_peak_widths_and_prominences_match_scipy(rel_height):
    sps = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(11)
    for _ in range(25):
        y = rng.normal(size=int(rng.integers(30, 200)))
        peaks, _ = sps.find_peaks(y, prominence=0)
        if peaks.size == 0:
            continue
        for a, b in zip(peak_prominences(y, peaks), sps.peak_prominences(y, peaks)):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
        for a, b in zip(
            peak_widths(y, peaks, rel_height=rel_height),
            sps.peak_widths(y, peaks, rel_height=rel_height),
        ):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("wlen", [8, 21])
def test_prominences_match_scipy_with_a_window(wlen):
    sps = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(12)
    for _ in range(25):
        y = rng.normal(size=int(rng.integers(30, 200)))
        peaks, _ = sps.find_peaks(y, prominence=0)
        if peaks.size == 0:
            continue
        np.testing.assert_allclose(
            peak_prominences(y, peaks, wlen=wlen)[0],
            sps.peak_prominences(y, peaks, wlen=wlen)[0],
            rtol=1e-10,
            atol=1e-12,
        )


def test_fwhm_measurement_recovers_the_true_width():
    x = np.arange(2001.0)
    for width in (12.0, 30.0, 75.0):
        y = gaussian(x, 1000.0, 1.0, width)
        peaks, _ = find_peaks(y, prominence=0.01)
        measured = peak_widths(y, peaks, rel_height=0.5)[0][0]
        assert measured == pytest.approx(width, rel=0.02), width


# --------------------------------------------------------------------------- #
# Filter behaviour
# --------------------------------------------------------------------------- #
def test_height_filter_is_inclusive():
    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    np.testing.assert_array_equal(find_peaks(y, height=2.0)[0], [3, 5])
    np.testing.assert_array_equal(find_peaks(y, height=(None, 2.0))[0], [1, 3])
    np.testing.assert_array_equal(find_peaks(y, height=4.0)[0], [])


def test_array_valued_height_filter():
    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    limits = np.full(y.size, 2.0)
    np.testing.assert_array_equal(find_peaks(y, height=limits)[0], [3, 5])
    limits[3] = -5.0  # lower the bar only at the peak index 3 -> no effect
    np.testing.assert_array_equal(find_peaks(y, height=limits)[0], [3, 5])
    limits[3] = 5.0  # now index 3 is excluded
    np.testing.assert_array_equal(find_peaks(y, height=limits)[0], [5])


def test_distance_filter_keeps_the_taller_peak():
    y = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(find_peaks(y)[0], [1, 5])
    # the peaks are 4 samples apart, so distance=5 drops the smaller one
    np.testing.assert_array_equal(find_peaks(y, distance=5)[0], [5])
    np.testing.assert_array_equal(find_peaks(y, distance=4)[0], [1, 5])


def test_threshold_filter_rejects_peaks_on_a_shoulder():
    y = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 1.2, 1.15, 0.0])
    np.testing.assert_array_equal(find_peaks(y)[0], [2, 5])
    # peak 5 only stands 0.05 above its right neighbour -> rejected
    np.testing.assert_array_equal(find_peaks(y, threshold=0.1)[0], [2])
    _, props = find_peaks(y, threshold=0.1)
    np.testing.assert_allclose(props["left_thresholds"], [0.5])
    np.testing.assert_allclose(props["right_thresholds"], [0.5])


def test_prominence_filter_rejects_a_shoulder_that_is_a_local_maximum():
    x = np.arange(2001.0)
    y = gaussian(x, 600.0, 1.0, 40.0) + gaussian(x, 650.0, 0.6, 30.0)
    peaks, props = find_peaks(y, prominence=0)
    np.testing.assert_array_equal(peaks, [600, 649])
    np.testing.assert_allclose(props["prominences"][0], 1.0, atol=1e-3)
    assert props["prominences"][1] == pytest.approx(0.229, abs=0.01)
    np.testing.assert_array_equal(find_peaks(y, prominence=0.4)[0], [600])


def test_width_filter_selects_peaks_by_width():
    x = np.arange(2001.0)
    y = gaussian(x, 1000.0, 1.0, 30.0)
    np.testing.assert_array_equal(find_peaks(y, width=(0.0, 0.9 * 30.0))[0], [])
    np.testing.assert_array_equal(find_peaks(y, width=(0.9 * 30.0, None))[0], [1000])


def test_invalid_arguments_raise():
    y = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        find_peaks(y, distance=0)
    with pytest.raises(ValueError):
        find_peaks(np.ones((3, 3)))
    with pytest.raises(ValueError):
        peak_prominences(y, [7])  # out of range
    with pytest.raises(ValueError):
        peak_prominences(y, [1.5])  # non-integer position
    with pytest.raises(ValueError):
        peak_widths(y, [1], prominence_data=(np.zeros(2), np.zeros(2), np.zeros(2)))
    with pytest.raises(ValueError):
        find_peaks(y, height=np.ones(5))  # array size must match y
