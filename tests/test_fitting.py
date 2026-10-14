"""Tests for the peak profile models and the Levenberg-Marquardt fitter."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_peaks.fitting import (
    MODELS,
    fit_peak,
    fit_peaks,
    fit_region,
    gaussian,
    get_model,
    lm_fit,
    lorentzian,
    pseudo_voigt,
    voigt,
)

from helpers import gaussian as helper_gaussian
from helpers import make_spectrum


# --------------------------------------------------------------------------- #
# Profile models
# --------------------------------------------------------------------------- #
def test_gaussian_profile():
    x = np.linspace(-50.0, 50.0, 1001)
    y = gaussian(x, 2.5, 0.0, 4.0)
    assert y.max() == pytest.approx(2.5)
    assert x[int(np.argmax(y))] == pytest.approx(0.0)
    assert np.allclose(y, y[::-1])  # symmetric about the centre
    # value at one sigma is amplitude * exp(-1/2)
    sigma_step = int(round(4.0 / (x[1] - x[0])))
    assert y[500 + sigma_step] == pytest.approx(2.5 * np.exp(-0.5), rel=1e-3)
    # an off-centre profile is symmetric about its own centre
    per_unit = int(round(1.0 / (x[1] - x[0])))
    shifted = gaussian(x, 2.5, 7.0, 4.0)
    assert shifted[500 + 7 * per_unit] == pytest.approx(2.5, rel=1e-4)


def test_lorentzian_profile_has_lorentzian_fwhm():
    x = np.linspace(-50.0, 50.0, 1001)
    gamma = 3.0
    y = lorentzian(x, 2.0, 0.0, gamma)
    assert y[500] == pytest.approx(2.0)
    step = int(round(gamma / (x[1] - x[0])))
    assert y[500 + step] == pytest.approx(1.0, rel=1e-3)  # half maximum at +-gamma


def test_pseudo_voigt_blends_gaussian_and_lorentzian():
    x = np.linspace(-30.0, 30.0, 601)
    pure_gauss = pseudo_voigt(x, 1.0, 0.0, 2.0, 3.0, 0.0)
    pure_lorentz = pseudo_voigt(x, 1.0, 0.0, 2.0, 3.0, 1.0)
    mixed = pseudo_voigt(x, 1.0, 0.0, 2.0, 3.0, 0.5)
    np.testing.assert_allclose(pure_gauss, gaussian(x, 1.0, 0.0, 2.0))
    np.testing.assert_allclose(pure_lorentz, lorentzian(x, 1.0, 0.0, 3.0))
    np.testing.assert_allclose(mixed, 0.5 * (pure_gauss + pure_lorentz))


def test_voigt_profile_shape_and_limits():
    x = np.linspace(-40.0, 40.0, 801)  # odd, uniform
    y = voigt(x, 2.0, 0.0, 2.0, 2.0)
    assert y.max() == pytest.approx(2.0, rel=1e-6)
    assert np.allclose(y, y[::-1], atol=1e-12)
    # a small Gaussian width approaches the pure Lorentzian, and vice versa
    near_lorentz = voigt(x, 1.0, 0.0, 0.05, 3.0)
    np.testing.assert_allclose(near_lorentz, lorentzian(x, 1.0, 0.0, 3.0), atol=2e-2)
    near_gauss = voigt(x, 1.0, 0.0, 3.0, 0.05)
    np.testing.assert_allclose(near_gauss, gaussian(x, 1.0, 0.0, 3.0), atol=2e-2)
    # the mixed profile must be wider than the pure Gaussian of the same sigma
    step = x[1] - x[0]
    half_voigt = np.nonzero(y >= 1.0)[0]
    half_gauss = np.nonzero(gaussian(x, 2.0, 0.0, 2.0) >= 1.0)[0]
    assert (half_voigt[-1] - half_voigt[0]) > (half_gauss[-1] - half_gauss[0])
    assert step > 0


def test_voigt_rejects_non_uniform_or_even_grids():
    with pytest.raises(ValueError):
        voigt(np.array([0.0, 1.0, 3.0]), 1.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        voigt(np.arange(10.0), 1.0, 0.0, 1.0, 1.0)  # even number of samples


def test_model_registry_and_metadata():
    assert set(MODELS) == {"gaussian", "lorentzian", "pseudo_voigt", "voigt"}
    model = get_model("gaussian")
    assert model.n_params == 3
    assert get_model(model) is model
    with pytest.raises(ValueError):
        get_model("triangle")
    params = [1.5, 10.0, 2.0]
    assert model.area(params) == pytest.approx(1.5 * 2.0 * np.sqrt(2.0 * np.pi))
    assert model.fwhm(params) == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)) * 2.0)
    assert get_model("pseudo_voigt").n_params == 5
    assert get_model("voigt").fwhm([1.0, 0.0, 2.0, 2.0]) > 2.0 * 2.0


# --------------------------------------------------------------------------- #
# Levenberg-Marquardt
# --------------------------------------------------------------------------- #
def test_lm_fit_is_exact_for_a_linear_model():
    x = np.linspace(0.0, 10.0, 50)
    y = 2.5 * x - 1.5
    params, errors, chi2, n_iter, converged = lm_fit(
        lambda xx, a, b: a + b * xx, x, y, [0.0, 0.0]
    )
    np.testing.assert_allclose(params, [-1.5, 2.5], atol=1e-10)
    assert chi2 == pytest.approx(0.0, abs=1e-20)
    assert converged
    assert 1 <= n_iter <= 200
    assert np.all(np.isfinite(errors))


def test_lm_fit_recovers_gaussian_parameters_and_errors():
    rng = np.random.default_rng(0)
    x = np.arange(200.0)
    truth = (3.0, 40.3, 2.5)
    y = gaussian(x, *truth) + rng.normal(scale=0.02, size=x.size)
    params, errors, chi2, _, converged = lm_fit(gaussian, x, y, [3.0, 40.0, 2.0])
    assert converged
    # every parameter inside three standard errors of the truth
    assert np.all(np.abs(params - np.array(truth)) < 3.0 * np.maximum(errors, 1e-6))
    np.testing.assert_allclose(params, truth, rtol=0.01)
    # the reported errors have the right order of magnitude for this noise level
    assert 0.001 < float(np.max(errors)) < 0.1


def test_lm_fit_reports_a_bad_start_gracefully():
    """A poor initial guess must not crash: it either converges or reports failure."""
    x = np.arange(200.0)
    y = gaussian(x, 3.0, 100.0, 5.0)
    params, errors, chi2, n_iter, converged = lm_fit(
        gaussian, x, y, [1.0, 40.0, 40.0], max_iter=50
    )
    assert np.all(np.isfinite(params))
    assert np.all(np.isfinite(errors))
    assert isinstance(converged, bool) and isinstance(n_iter, int)


# --------------------------------------------------------------------------- #
# Region and per-peak fitting
# --------------------------------------------------------------------------- #
def test_fit_peak_recovers_a_peak_on_a_sloping_background():
    rng = np.random.default_rng(1)
    x = np.arange(301.0)
    y = 0.5 + 0.01 * x + helper_gaussian(x, 150.0, 2.0, 25.0)
    y = y + rng.normal(scale=0.01, size=x.size)
    fit = fit_peak(x, y, center=152.0, fwhm=25.0, model="gaussian", baseline_order=1)
    block = fit.peaks()[0]
    assert block["center"] == pytest.approx(150.0, abs=0.1)
    assert block["fwhm"] == pytest.approx(25.0, rel=0.02)
    assert block["amplitude"] == pytest.approx(2.0, rel=0.02)
    assert block["area"] == pytest.approx(2.0 * (25.0 / 2.3548200450309493) * np.sqrt(2 * np.pi), rel=0.03)
    assert fit.converged
    # the background is returned separately and matches the linear ramp,
    # evaluated on the fitted window (fit_peak works on a local window)
    np.testing.assert_allclose(fit.background(), 0.5 + 0.01 * fit.x, atol=0.02)
    assert 5 < fit.x.size < y.size  # a local window was used
    # the residual is at the noise level
    assert np.abs(fit.residual).max() < 0.05


def test_fit_region_separates_overlapping_peaks():
    rng = np.random.default_rng(2)
    x = np.arange(200.0)
    y = helper_gaussian(x, 100.0, 1.0, 14.0) + helper_gaussian(x, 112.0, 0.8, 14.0)
    y = y + rng.normal(scale=0.005, size=x.size)
    fit = fit_region(
        x, y, [98.0, 114.0], model="gaussian", fwhm_guesses=[14.0, 14.0], baseline_order=None
    )
    block_a, block_b = fit.peaks()
    assert block_a["center"] == pytest.approx(100.0, abs=0.3)
    assert block_b["center"] == pytest.approx(112.0, abs=0.3)
    assert block_a["amplitude"] == pytest.approx(1.0, rel=0.05)
    assert block_b["amplitude"] == pytest.approx(0.8, rel=0.05)
    assert fit.converged and fit.reduced_chi2 < 1e-3
    assert len(fit.names) == 2 * 3  # two Gaussians and no baseline
    assert fit.background().tolist() == [0.0] * x.size
    assert fit.curve().shape == y.shape
    assert np.abs(fit.residual).max() < 0.05


def test_fit_region_validates_its_input():
    x = np.arange(50.0)
    y = gaussian(x, 25.0, 1.0, 4.0)
    with pytest.raises(ValueError):
        fit_region(x, y[:40], [25.0])
    with pytest.raises(ValueError):
        fit_region(x[:2], y[:2], [1.0])
    with pytest.raises(ValueError):
        fit_region(x, y, [])
    with pytest.raises(ValueError):
        fit_region(x, y, [25.0], fwhm_guesses=[1.0, 2.0])
    with pytest.raises(ValueError):
        fit_region(x, y, [25.0], model="triangle")


def test_fit_peaks_refines_detected_peaks():
    x, y, truth = make_spectrum(
        [(520.0, 1.0, 24.0), (905.0, 1.6, 22.0), (1300.0, 0.9, 45.0)],
        noise=0.005,
        background="exponential",
        seed=2,
    )
    from spectral_peaks.peaks import detect_peaks

    result = detect_peaks(y, x=x, baseline="airpls", smooth=9, prominence=0.05)
    assert len(result) == 3
    assert all(peak.fit is None for peak in result)
    fit_peaks(result, model="gaussian")
    for peak, expected in zip(result, truth):
        assert peak.fit is not None
        assert set(peak.fit) >= {"center", "sigma", "fwhm", "area", "reduced_chi2", "window"}
        assert peak.fit["center"] == pytest.approx(expected["center"], abs=0.2)
        assert peak.fit["fwhm"] == pytest.approx(expected["fwhm"], rel=0.05)
        # with update=True (the default) the peak position becomes the fitted one
        assert peak.x == pytest.approx(peak.fit["center"])
        assert peak.width == pytest.approx(peak.fit["fwhm"])


def test_fit_peaks_handles_an_empty_result():
    x, y, _ = make_spectrum([(500.0, 1.0, 20.0)], noise=0.0)
    from spectral_peaks.peaks import detect_peaks

    result = detect_peaks(y, x=x)
    result.peaks = []
    assert fit_peaks(result, model="gaussian") is result
