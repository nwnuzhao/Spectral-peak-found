"""Smoothing and differentiation filters built from scratch on NumPy.

Everything here is a straightforward, dependency-free re-implementation of the
classical spectral pre-processing filters:

* :func:`savgol_coeffs` / :func:`savgol_filter` -- Savitzky-Golay polynomial
  least-squares smoothing and differentiation (Savitzky & Golay, 1964).
* :func:`moving_average` -- boxcar filter.
* :func:`gaussian_smoothing` -- Gaussian convolution filter.

All filters return an array of exactly the same length as the input; signal
edges are handled with a configurable padding mode (``reflect`` by default,
which mirrors the signal without repeating the edge sample).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "PAD_MODES",
    "gaussian_kernel",
    "gaussian_smoothing",
    "moving_average",
    "savgol_coeffs",
    "savgol_filter",
]

PAD_MODES = ("reflect", "symmetric", "edge", "constant", "wrap", "nearest")


def _pad(y: np.ndarray, left: int, right: int, mode: str) -> np.ndarray:
    """Pad ``y`` on both sides, degrading gracefully for very short signals."""
    if left == 0 and right == 0:
        return y
    if mode == "nearest":
        mode = "edge"
    if mode not in PAD_MODES:
        raise ValueError(f"unknown pad mode {mode!r}; expected one of {PAD_MODES}")
    if mode in ("reflect", "symmetric", "wrap") and (left >= y.size or right >= y.size):
        mode = "edge"  # np.pad would raise for these modes on short input
    return np.pad(y, (left, right), mode=mode)


def _convolve_centered(y: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """Correlate ``y`` with ``kernel`` keeping the kernel centre aligned.

    ``out[i] == sum_j kernel[j] * padded[i + j]``.
    """
    half_left = kernel.size // 2
    half_right = kernel.size - 1 - half_left
    padded = _pad(y, half_left, half_right, mode)
    windows = sliding_window_view(padded, kernel.size)
    return windows @ kernel


def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: float | None = None,
) -> np.ndarray:
    """Savitzky-Golay filter coefficients.

    Fit a polynomial of degree ``polyorder`` to ``window_length`` equally
    spaced samples by least squares and evaluate its ``deriv``-th derivative
    at ``pos`` (``pos = 0`` is the window centre, in sample units).  The
    returned vector ``c`` satisfies

        ``out[i] = sum_j c[j] * y[i + j - (window_length - 1) // 2]``

    with the result divided by ``delta ** deriv`` to convert from sample units
    to x units of spacing ``delta``.
    """
    window_length = int(window_length)
    polyorder = int(polyorder)
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must satisfy 0 <= polyorder < window_length")
    if deriv < 0 or deriv > polyorder:
        raise ValueError("deriv must satisfy 0 <= deriv <= polyorder")
    if delta == 0.0 and deriv > 0:
        raise ValueError("delta must be non-zero when deriv > 0")

    half = window_length // 2
    if pos is None:
        pos = 0.0
    x = np.arange(-half, half + 1, dtype=float)
    design = np.vander(x, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(design)

    factorial = math.factorial
    coeffs = np.zeros(window_length)
    for k in range(deriv, polyorder + 1):
        # d^deriv/dx^deriv x**k evaluated at pos
        scale = factorial(k) / factorial(k - deriv) * float(pos) ** (k - deriv)
        coeffs += pinv[k] * scale
    if deriv:
        coeffs /= float(delta) ** deriv
    return coeffs


def savgol_filter(
    y: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    mode: str = "reflect",
) -> np.ndarray:
    """Apply a Savitzky-Golay filter (smoothing for ``deriv == 0``)."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < window_length:
        raise ValueError("y must be at least as long as window_length")
    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    return _convolve_centered(y, coeffs, mode)


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Normalised, centred 1-D Gaussian kernel."""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smoothing(
    y: np.ndarray, sigma: float, truncate: float = 4.0, mode: str = "reflect"
) -> np.ndarray:
    """Convolve ``y`` with a Gaussian kernel of standard deviation ``sigma`` samples."""
    y = np.asarray(y, dtype=float)
    return _convolve_centered(y, gaussian_kernel(sigma, truncate), mode)


def moving_average(y: np.ndarray, window: int, mode: str = "reflect") -> np.ndarray:
    """Boxcar (moving average) filter of width ``window`` samples.

    Even window lengths are allowed and are centred as closely as possible.
    """
    y = np.asarray(y, dtype=float)
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > y.size:
        raise ValueError("window must not exceed the signal length")
    kernel = np.full(window, 1.0 / window)
    return _convolve_centered(y, kernel, mode)
