"""Baseline (background) estimation for spectra, implemented with NumPy only.

Every routine takes a 1-D signal ``y`` and returns an array of the same length
holding the estimated baseline, so the corrected spectrum is simply ``y - baseline``.
:func:`correct_baseline` wraps them behind a single ``method`` switch.

Implemented methods
-------------------
``snip``
    Statistics-sensitive Non-linear Iterative Peak-clipping (Ryan et al., 1988),
    run in the log-log-square-root domain for better peak suppression.
``airpls``
    Adaptive iteratively reweighted Penalized Least Squares (Zhang et al., 2010),
    the standard choice for Raman/IR spectra with fluorescent backgrounds.
``als``
    Asymmetric least squares (Eilers & Boelens, 2005), the ancestor of airPLS.
``morphological`` / ``rolling_ball``
    Morphological opening (erosion followed by dilation) with a flat or a
    ball-shaped structuring element (Sternberg, 1983; Kneen & Annegarn, 1996).
``polynomial``
    Iterative polynomial fit with rejection of points lying above the fit.

References
----------
C.G. Ryan et al. *Nucl. Instrum. Methods Phys. Res. B* 34 (1988) 396-402.
Z.-M. Zhang et al. *Analyst* 135 (2010) 1138-1146.
P.H.C. Eilers, H.F.M. Boelens. *Baseline correction with asymmetric least
squares smoothing*, Leiden University (2005).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._banded import solve_difference_system
from .smoothing import _pad, moving_average, savgol_filter

__all__ = [
    "BASELINE_METHODS",
    "airpls_baseline",
    "als_baseline",
    "correct_baseline",
    "iterative_polynomial_baseline",
    "morphological_baseline",
    "rolling_ball_baseline",
    "snip_baseline",
]

BASELINE_METHODS = ("airpls", "als", "snip", "morphological", "rolling_ball", "polynomial")


def _as_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if y.size == 0:
        raise ValueError("y must not be empty")
    return y


def _min_filter(y: np.ndarray, half: int, mode: str = "edge") -> np.ndarray:
    if half <= 0:
        return y.astype(float)
    padded = _pad(y, half, half, mode)
    return sliding_window_view(padded, 2 * half + 1).min(axis=1)


def _max_filter(y: np.ndarray, half: int, mode: str = "edge") -> np.ndarray:
    if half <= 0:
        return y.astype(float)
    padded = _pad(y, half, half, mode)
    return sliding_window_view(padded, 2 * half + 1).max(axis=1)


def snip_baseline(
    y: np.ndarray,
    iterations: int | None = None,
    transform: bool = True,
    offset: float | None = None,
    clip: bool = True,
) -> np.ndarray:
    """SNIP baseline: iterative peak clipping in the log-log-square-root domain.

    Each pass ``p`` replaces every sample by ``min(z_i, (z_{i-p} + z_{i+p}) / 2)``,
    which erodes features narrower than ``2p`` samples and therefore removes
    peaks while following the background.  ``iterations`` should be chosen near
    the half-width of the broadest peak in the spectrum; the default is
    ``round(n / 20)``.

    Parameters
    ----------
    y
        Signal. Negative values are handled by shifting upwards, either by the
        supplied ``offset`` or automatically by ``-y.min()``.
    transform
        Apply the log-log-sqrt transform (recommended; better compression of
        large peak-to-background ratios).
    clip
        Clip the estimated baseline at the signal (``baseline <= y``).
    """
    y = _as_1d(y)
    n = y.size
    shift = float(offset) if offset is not None else max(0.0, -float(y.min()))
    z = y + shift
    if transform:
        z = np.log(np.log(np.sqrt(z + 1.0) + 1.0) + 1.0)
    z = np.array(z, dtype=float, copy=True)
    if iterations is None:
        iterations = max(1, int(round(n / 20.0)))
    passes = int(min(max(int(iterations), 0), (n - 1) // 2))
    for p in range(1, passes + 1):
        np.minimum(
            z[p : n - p],
            0.5 * (z[: n - 2 * p] + z[2 * p :]),
            out=z[p : n - p],
        )
    if transform:
        z = (np.exp(np.exp(z) - 1.0) - 1.0) ** 2 - 1.0
    baseline = z - shift
    if clip:
        baseline = np.minimum(baseline, y)
    return baseline


def airpls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    order: int = 2,
    max_iter: int = 100,
    tol: float = 1e-3,
    return_info: bool = False,
):
    """airPLS baseline: adaptive iteratively reweighted penalised least squares.

    Fits a smooth Whittaker spline and lets the weights punish points that lie
    *above* the fit (peaks), so the fit converges onto the background.  ``lam``
    controls smoothness (larger = smoother), ``order`` the difference penalty.

    Returns the baseline, or ``(baseline, info)`` when ``return_info`` is set,
    where ``info`` contains ``iterations``, ``lam``, ``order`` and the final
    ``weights``.
    """
    y = _as_1d(y)
    n = y.size
    if n < 3 or lam <= 0:
        return (y.copy(), {"iterations": 0, "lam": lam, "order": order, "weights": np.ones(n)}) if return_info else y.copy()
    if order < 0 or order >= n:
        raise ValueError("order must satisfy 0 <= order < len(y)")

    weights = np.ones(n)
    baseline = y.copy()
    scale = float(np.abs(y).sum())
    used = 0
    for iteration in range(1, int(max_iter) + 1):
        used = iteration
        baseline = solve_difference_system(y, weights, lam, order)
        diff = y - baseline
        negative = diff < 0.0
        if not negative.any():
            break
        neg_abs = np.abs(diff[negative])
        dssn = float(neg_abs.sum())
        if scale > 0.0 and dssn < tol * scale:
            break
        weights = np.zeros(n)
        weights[negative] = np.exp(iteration * neg_abs / dssn)
        # Anchor both ends of the spectrum into the fit, as in the reference
        # implementation (they cannot be peaks and would otherwise float away).
        end_weight = float(np.exp(iteration * neg_abs.max() / dssn))
        weights[0] = weights[-1] = max(end_weight, weights[0], weights[-1])
    info = {"iterations": used, "lam": lam, "order": order, "weights": weights}
    return (baseline, info) if return_info else baseline


def als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    order: int = 2,
    max_iter: int = 10,
) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens, 2005).

    ``p`` is the weight given to points *above* the fit (small ``p`` ~ 1e-3 to
    1e-1 pushes the fit below the peaks); ``lam`` controls smoothness.
    """
    y = _as_1d(y)
    n = y.size
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if n < 3 or lam <= 0:
        return y.copy()
    weights = np.ones(n)
    baseline = y.copy()
    for _ in range(int(max_iter)):
        baseline = solve_difference_system(y, weights, lam, order)
        weights = np.where(y > baseline, p, 1.0 - p)
    return baseline


def morphological_baseline(
    y: np.ndarray,
    half_window: int = 50,
    smooth: bool = True,
    smooth_window: int | None = None,
    mode: str = "edge",
) -> np.ndarray:
    """Morphological opening baseline with a flat structuring element.

    The opening (erosion then dilation over a window of ``2 * half_window + 1``
    samples) removes everything narrower than the window, which is exactly what
    peaks are.  ``half_window`` should be a bit larger than the widest peak, or
    the peaks survive; too large a value lets the baseline follow the peaks'
    shoulders.  The result is optionally boxcar-smoothed and always clipped at
    the signal.
    """
    y = _as_1d(y)
    half = int(half_window)
    if half < 1:
        raise ValueError("half_window must be >= 1")
    opened = _max_filter(_min_filter(y, half, mode), half, mode)
    baseline = np.minimum(opened, y)
    if smooth:
        window = int(smooth_window) if smooth_window else 2 * half + 1
        window = max(1, min(window, y.size))
        baseline = moving_average(baseline, window, mode="edge")
        baseline = np.minimum(baseline, y)
    return baseline


def rolling_ball_baseline(
    y: np.ndarray,
    radius: int = 50,
    smoothing: int | None = None,
    mode: str = "reflect",
) -> np.ndarray:
    """Rolling-ball baseline (morphological opening with a curved element).

    ``radius`` is in samples.  The structuring element is the contact profile of
    a ball of that radius,

        ``B(d) = radius - sqrt(radius**2 - d**2)``,

    which is zero at the contact point and rises to ``radius`` at both ends of
    the window.  The erosion

        ``c(x) = min_d [ y(x + d) - B(d) ]``

    gives the lowest position the contact point can take, and the dilation

        ``baseline(x) = max_d [ c(x - d) + B(d) ]``

    rebuilds the smooth background from it.

    Properties worth knowing: the result equals the signal on flat or monotone
    stretches and cuts off everything narrower than the ball, but it is biased
    *upwards* where the background curves (the ball cannot enter a concave
    region); the bias grows roughly quadratically with ``radius``.  Keep
    ``radius`` a few times the half-width of the widest peak and sanity-check
    the result on a peak-free region of your data.  ``smoothing`` applies an
    optional boxcar filter of that width on top.
    """
    y = _as_1d(y)
    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if 2 * radius + 1 > y.size:
        raise ValueError("radius is too large for the signal length")
    offsets = np.arange(-radius, radius + 1, dtype=float)
    ball = radius - np.sqrt(np.maximum(radius * radius - offsets * offsets, 0.0))

    window = 2 * radius + 1
    eroded = (sliding_window_view(_pad(y, radius, radius, mode), window) - ball).min(axis=1)
    opened = (
        sliding_window_view(_pad(eroded, radius, radius, mode), window) + ball
    ).max(axis=1)
    baseline = np.minimum(opened, y)
    if smoothing:
        baseline = moving_average(baseline, max(1, int(smoothing)), mode="edge")
        baseline = np.minimum(baseline, y)
    return baseline


def iterative_polynomial_baseline(
    y: np.ndarray,
    order: int = 5,
    x: np.ndarray | None = None,
    max_iter: int = 50,
    sigma: float = 1.0,
) -> np.ndarray:
    """Polynomial baseline with iterative rejection of points above the fit.

    Fits a polynomial of the given ``order``, then repeatedly discards samples
    whose residual exceeds ``median + sigma * 1.4826 * MAD`` and refits, so the
    fit settles onto the background.  Use only for gently curved backgrounds.
    """
    y = _as_1d(y)
    n = y.size
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    if n <= order + 1:
        return np.full(n, y.mean())
    xv = np.arange(n, dtype=float) if x is None else np.asarray(x, dtype=float)
    if xv.size != n:
        raise ValueError("x and y must have the same length")

    keep = np.ones(n, dtype=bool)
    baseline = np.zeros(n)
    for _ in range(int(max_iter)):
        if keep.sum() <= order + 1:
            break
        coef = np.polyfit(xv[keep], y[keep], order)
        baseline = np.polyval(coef, xv)
        resid = y - baseline
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) * 1.4826
        if mad <= 0.0:
            mad = float(resid[keep].std()) or 1.0
        new_keep = (resid - med) < sigma * mad
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return baseline


_BASELINE_FUNCS = {
    "airpls": airpls_baseline,
    "als": als_baseline,
    "snip": snip_baseline,
    "morphological": morphological_baseline,
    "rolling_ball": rolling_ball_baseline,
    "polynomial": iterative_polynomial_baseline,
}


def correct_baseline(
    y: np.ndarray,
    method: str = "airpls",
    return_baseline: bool = False,
    **kwargs,
):
    """Estimate and subtract a baseline.

    ``method`` is one of :data:`BASELINE_METHODS`: ``"airpls"``, ``"als"``,
    ``"snip"``, ``"morphological"``, ``"rolling_ball"``, ``"polynomial"``.
    Extra keyword arguments are forwarded to the selected function.

    Returns the corrected signal, or ``(corrected, baseline)`` when
    ``return_baseline`` is true.
    """
    y = _as_1d(y)
    key = str(method).lower()
    if key not in _BASELINE_FUNCS:
        raise ValueError(f"unknown baseline method {method!r}; expected one of {BASELINE_METHODS}")
    baseline = _BASELINE_FUNCS[key](y, **kwargs)
    corrected = y - baseline
    return (corrected, baseline) if return_baseline else corrected


def savgol_baseline(y: np.ndarray, window_length: int = 101, polyorder: int = 3) -> np.ndarray:
    """Very simple baseline: a heavily smoothed copy of the signal.

    Only suitable for flat backgrounds; provided for completeness.
    """
    return np.minimum(savgol_filter(_as_1d(y), window_length, polyorder, mode="edge"), y)


