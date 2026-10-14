"""Peak-related scalar metrics: noise level, SNR, FWHM conversions and areas.

All functions work on plain 1-D NumPy arrays and return Python floats or arrays
of floats, so they can be used independently of the detectors.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "estimate_noise",
    "fwhm_from_sigma",
    "peak_area",
    "peak_areas",
    "sigma_from_fwhm",
    "snr",
    "trapezoid",
]

_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548200450309493


def trapezoid(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Trapezoidal integral of ``y`` over ``x`` (unit spacing when ``x`` is None).

    Implemented directly instead of relying on ``np.trapz``/``np.trapezoid``,
    whose name changed in NumPy 2.0.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    area = 0.5 * (y[1:] + y[:-1])
    if x is None:
        return float(area.sum())
    x = np.asarray(x, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    return float((area * np.diff(x)).sum())


def estimate_noise(y: np.ndarray, method: str = "diff", window: int | None = None) -> float:
    """Estimate the standard deviation of the noise in a spectrum.

    ``method``:

    ``"diff"`` (default)
        Robust sigma from the first differences, ``1.4826 * MAD(diff(y)) / sqrt(2)``.
        Insensitive to peaks and to smooth backgrounds, which is why it is the
        default for spectra.
    ``"mad"``
        ``1.4826 * MAD(y)`` about the median; treats the whole signal as noise
        around a constant level (bad when a wide peak dominates).
    ``"residual"``
        ``1.4826 * MAD`` of the residuals of a Savitzky-Golay smoothing, i.e. the
        noise seen after removing structure on the scale of ``window`` samples.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    method = str(method).lower()
    if method == "diff":
        if y.size < 3:
            return 0.0
        d = np.diff(y)
        return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))
    if method == "mad":
        if y.size < 2:
            return 0.0
        return float(1.4826 * np.median(np.abs(y - np.median(y))))
    if method == "residual":
        from .smoothing import savgol_filter  # local import avoids a cycle

        if y.size < 7:
            return 0.0
        win = int(window) if window else max(7, min(y.size // 20 * 2 + 1, 101))
        if win % 2 == 0:
            win += 1
        win = min(win, y.size if y.size % 2 else y.size - 1)
        resid = y - savgol_filter(y, win, 2, mode="edge")
        return float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    raise ValueError(f"unknown noise method {method!r}; expected 'diff', 'mad' or 'residual'")


def snr(signal: float | np.ndarray, noise: float | np.ndarray):
    """Signal-to-noise ratio; ``inf`` where the noise estimate is zero."""
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if np.any(noise < 0):
        raise ValueError("noise must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(noise > 0, signal / np.where(noise > 0, noise, 1.0), np.inf)
    return ratio if ratio.ndim else float(ratio)


def fwhm_from_sigma(sigma: float) -> float:
    """FWHM of a Gaussian with the given standard deviation."""
    return float(_FWHM_OVER_SIGMA * sigma)


def sigma_from_fwhm(fwhm: float) -> float:
    """Standard deviation of a Gaussian with the given FWHM."""
    return float(fwhm / _FWHM_OVER_SIGMA)


def peak_area(
    y: np.ndarray,
    left_ip: float,
    right_ip: float,
    x: np.ndarray | None = None,
    subtract_baseline: bool = True,
) -> float:
    """Integrate a peak between two (fractional) positions.

    The two positions are rounded to the sample grid, and by default a straight
    line joining the values at those two samples is subtracted, so the result is
    the area of the peak above its own local baseline.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    lo = int(np.clip(round(min(left_ip, right_ip)), 0, n - 1))
    hi = int(np.clip(round(max(left_ip, right_ip)), 0, n - 1))
    if hi <= lo:
        return 0.0
    segment = y[lo : hi + 1].copy()
    if subtract_baseline:
        segment -= np.linspace(segment[0], segment[-1], segment.size)
    if x is None:
        return trapezoid(segment)
    return trapezoid(segment, np.asarray(x, dtype=float)[lo : hi + 1])


def peak_areas(
    y: np.ndarray,
    left_ips: np.ndarray,
    right_ips: np.ndarray,
    x: np.ndarray | None = None,
    subtract_baseline: bool = True,
) -> np.ndarray:
    """Vectorised :func:`peak_area` over the peaks' left/right positions."""
    left_ips = np.asarray(left_ips, dtype=float)
    right_ips = np.asarray(right_ips, dtype=float)
    if left_ips.size != right_ips.size:
        raise ValueError("left_ips and right_ips must have the same size")
    return np.array(
        [
            peak_area(y, li, ri, x=x, subtract_baseline=subtract_baseline)
            for li, ri in zip(left_ips, right_ips)
        ],
        dtype=float,
    )
