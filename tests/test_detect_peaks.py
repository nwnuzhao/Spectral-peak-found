"""End-to-end tests for the :func:`spectral_peaks.detect_peaks` pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks import Peak, PeakResult, detect_peaks

from helpers import match_peaks, make_spectrum

SPECTRUM = [
    (520.0, 1.0, 24.0),
    (700.0, 0.6, 30.0),
    (905.0, 1.6, 22.0),
    (1300.0, 0.9, 45.0),
    (1700.0, 1.2, 36.0),
    (2100.0, 0.5, 20.0),
]
CWT_WIDTHS = np.arange(2, 26)
ALL_METHODS = [
    ("prominence", {}),
    ("derivative", {}),
    ("cwt", {"cwt_widths": CWT_WIDTHS}),
]


@pytest.fixture(scope="module")
def spectrum():
    return make_spectrum(SPECTRUM, noise=0.002, background="exponential", seed=1)


def _run(spectrum, **kwargs):
    x, y, truth = spectrum
    return x, y, truth, detect_peaks(y, x=x, baseline="airpls", smooth=11, **kwargs)


@pytest.mark.parametrize("method,kwargs", ALL_METHODS)
def test_every_method_finds_every_peak_by_default(spectrum, method, kwargs):
    """No filters given: the built-in 5-sigma prominence gate has to do the job."""
    _, _, truth, result = _run(spectrum, method=method, **kwargs)
    matched, false_positives, max_error = match_peaks(
        result.positions(), [t["center"] for t in truth], 2.0
    )
    assert matched == len(truth), [peak.x for peak in result]
    assert false_positives == 0
    assert max_error <= 1.0
    assert result.method == method
    assert result.noise > 0.0


@pytest.mark.parametrize("method,kwargs", ALL_METHODS)
def test_peak_properties_are_accurate(spectrum, method, kwargs):
    _, _, truth, result = _run(spectrum, method=method, **kwargs)
    assert len(result) == len(truth)
    for peak, expected in zip(result, truth):
        assert peak.x == pytest.approx(expected["center"], abs=0.3)
        assert peak.height == pytest.approx(expected["height"], rel=0.08)
        assert peak.width == pytest.approx(expected["fwhm"], rel=0.05)
        assert peak.area == pytest.approx(expected["area"], rel=0.08)
        assert peak.prominence > 0.0
        assert peak.left_base < peak.index < peak.right_base
        assert peak.snr > 20.0
        assert peak.fwhm == peak.width


def test_x_units_drive_widths_and_areas(spectrum):
    x, y, _ = spectrum
    samples = detect_peaks(y, x=x, baseline="airpls", smooth=11)
    scaled_x = x[0] + 0.5 * np.arange(y.size)
    scaled = detect_peaks(y, x=scaled_x, baseline="airpls", smooth=11)
    # positions move to the new axis, widths and areas scale with the spacing
    np.testing.assert_allclose(
        scaled.positions(), scaled_x[samples.indices()], atol=1e-9
    )
    np.testing.assert_allclose(scaled.arrays()["width"], 0.5 * samples.arrays()["width"], rtol=1e-6)
    np.testing.assert_allclose(scaled.arrays()["area"], 0.5 * samples.arrays()["area"], rtol=1e-6)


def test_baseline_correction_recovers_the_true_height(spectrum):
    x, y, truth = spectrum
    raw = detect_peaks(y, x=x, smooth=11, prominence=0.05)
    corrected = detect_peaks(y, x=x, baseline="airpls", smooth=11, prominence=0.05)
    assert raw[0].height > 1.5 * truth[0]["height"]  # the background inflates it
    assert corrected[0].height == pytest.approx(truth[0]["height"], rel=0.05)


def test_precomputed_baseline_array_is_accepted(spectrum):
    from spectral_peaks.baseline import airpls_baseline

    x, y, _ = spectrum
    from_array = detect_peaks(y, x=x, baseline=airpls_baseline(y), smooth=11, prominence=0.05)
    from_name = detect_peaks(y, x=x, baseline="airpls", smooth=11, prominence=0.05)
    np.testing.assert_allclose(from_array.positions(), from_name.positions())
    np.testing.assert_allclose(from_array.heights(), from_name.heights(), rtol=1e-9)
    assert from_array.baseline is not None
    with pytest.raises(ValueError):
        detect_peaks(y, x=x, baseline=np.zeros(3))


def test_explicit_filters_override_the_default(spectrum):
    x, y, _ = spectrum
    strict = detect_peaks(y, x=x, baseline="airpls", smooth=11, prominence=1.0)
    default = detect_peaks(y, x=x, baseline="airpls", smooth=11)
    assert len(strict) < len(default)
    tallest_only = detect_peaks(y, x=x, baseline="airpls", smooth=11, height=1.4)
    assert [round(peak.x) for peak in tallest_only] == [905]
    three = detect_peaks(y, x=x, baseline="airpls", smooth=11, max_peaks=3)
    assert len(three) == 3
    assert list(three.positions()) == sorted(three.positions())


def test_noise_override_controls_the_snr_column(spectrum):
    x, y, _ = spectrum
    result = detect_peaks(y, x=x, baseline="airpls", smooth=11, noise=0.1)
    assert result.noise == 0.1
    np.testing.assert_allclose(result.arrays()["snr"], result.heights() / 0.1, rtol=1e-6)
    automatic = detect_peaks(y, x=x, baseline="airpls", smooth=11)
    assert 0.0 < automatic.noise < 0.01


def test_peak_result_container(spectrum, tmp_path):
    _, _, _, result = _run(spectrum)
    assert isinstance(result, PeakResult)
    assert len(result) == 6
    assert all(isinstance(peak, Peak) for peak in result)
    assert result[0] is result.peaks[0]
    assert result.indices().size == 6
    assert result.positions().size == result.heights().size == 6
    assert set(result.arrays()) >= {"index", "x", "height", "prominence", "width", "area", "snr"}
    table = result.table(digits=3)
    assert table.count("\n") == len(result) + 1  # header + separator + rows
    assert "prominence" in table
    path = tmp_path / "peaks.csv"
    result.to_csv(str(path))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(result) + 1
    assert lines[0].startswith("index,x,height")


def test_empty_results_are_handled():
    x = np.arange(200.0)
    result = detect_peaks(np.full(200, 3.0), x=x)
    assert len(result) == 0
    assert result.table().count("\n") == 1  # header and separator only
    assert PeakResult(peaks=[], x=x, y=np.zeros(200)).positions().size == 0


def test_max_width_rejects_broad_baseline_residue(spectrum):
    """A low, wide residual bump passes a prominence gate but not a width limit."""
    _, _, _, result = _run(spectrum)
    wide = max(peak.width for peak in result)
    narrow = detect_peaks(
        spectrum[1], x=spectrum[0], baseline="airpls", smooth=11, max_width=0.8 * wide
    )
    assert all(peak.width <= 0.8 * wide for peak in narrow)
    with pytest.raises(ValueError):
        detect_peaks(
            spectrum[1], x=spectrum[0], baseline="airpls", smooth=11,
            width=(0.0, 100.0), max_width=50.0,
        )


def test_fit_model_is_applied_when_requested(spectrum):
    _, _, truth, result = _run(spectrum, fit_model="gaussian")
    for peak, expected in zip(result, truth):
        assert peak.fit is not None
        assert peak.fit["center"] == pytest.approx(expected["center"], abs=0.2)
        assert peak.fit["area"] == pytest.approx(expected["area"], rel=0.1)


def test_invalid_arguments_raise(spectrum):
    x, y, _ = spectrum
    with pytest.raises(ValueError):
        detect_peaks(y, x=x, method="magic")
    with pytest.raises(ValueError):
        detect_peaks(y, x=x[:-1])
    with pytest.raises(ValueError):
        detect_peaks(y, x=x[::-1])
    with pytest.raises(ValueError):
        detect_peaks(y, x=x, noise_method="nope")
    with pytest.raises(ValueError):
        detect_peaks(np.ones((4, 4)))
    with pytest.raises(ValueError):
        detect_peaks(y, x=x, baseline="airpls", smooth=11, fit_model="triangle")