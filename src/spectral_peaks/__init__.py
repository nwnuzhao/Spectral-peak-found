"""Spectral peak finding implemented with pure Python + NumPy.

The package is dependency-free apart from NumPy: Savitzky-Golay smoothing, six
baseline estimators, three peak detectors, peak metrics and a Levenberg-Marquardt
profile fitter are all implemented from scratch.

Quick start
-----------
>>> import numpy as np
>>> from spectral_peaks import detect_peaks
>>> x = np.linspace(0, 100, 2001)
>>> y = 1.5 * np.exp(-(x - 0) / 30) + 2.0 * np.exp(-((x - 40) / 1.5) ** 2)
>>> result = detect_peaks(y, x=x, baseline="airpls", smooth=11, prominence=0.2)
>>> [round(peak.x, 2) for peak in result]
[40.0]
"""

from __future__ import annotations

from .baseline import (
    BASELINE_METHODS,
    airpls_baseline,
    als_baseline,
    correct_baseline,
    iterative_polynomial_baseline,
    morphological_baseline,
    rolling_ball_baseline,
    snip_baseline,
)
from .fitting import (
    MODELS,
    FitResult,
    fit_peak,
    fit_peaks,
    fit_region,
    gaussian,
    lm_fit,
    lorentzian,
    pseudo_voigt,
    voigt,
)
from .metrics import (
    estimate_noise,
    fwhm_from_sigma,
    peak_area,
    peak_areas,
    sigma_from_fwhm,
    snr,
)
from .peaks import (
    cwt,
    detect_peaks,
    find_peaks,
    find_peaks_cwt,
    find_peaks_derivative,
    peak_prominences,
    peak_widths,
    ricker,
)
from .smoothing import (
    gaussian_kernel,
    gaussian_smoothing,
    moving_average,
    savgol_coeffs,
    savgol_filter,
)
from .types import Peak, PeakResult, peaks_to_arrays

__version__ = "0.1.0"

__all__ = [
    "BASELINE_METHODS",
    "MODELS",
    "FitResult",
    "Peak",
    "PeakResult",
    "airpls_baseline",
    "als_baseline",
    "correct_baseline",
    "cwt",
    "detect_peaks",
    "estimate_noise",
    "find_peaks",
    "find_peaks_cwt",
    "find_peaks_derivative",
    "fit_peak",
    "fit_peaks",
    "fit_region",
    "fwhm_from_sigma",
    "gaussian",
    "gaussian_kernel",
    "gaussian_smoothing",
    "iterative_polynomial_baseline",
    "lm_fit",
    "lorentzian",
    "morphological_baseline",
    "moving_average",
    "peak_area",
    "peak_areas",
    "peak_prominences",
    "peak_widths",
    "peaks_to_arrays",
    "pseudo_voigt",
    "ricker",
    "rolling_ball_baseline",
    "savgol_coeffs",
    "savgol_filter",
    "sigma_from_fwhm",
    "snip_baseline",
    "snr",
    "voigt",
    "__version__",
]
