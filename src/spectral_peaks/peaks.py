"""Peak detection algorithms, implemented on NumPy only.

Three complementary families are provided:

``find_peaks``
    Classical "local maximum + optional property filters" detection with a
    SciPy-compatible signature (``height``, ``threshold``, ``distance``,
    ``prominence``, ``width``, ``plateau_size``, ``wlen``).  Exact for well
    separated peaks and cheap to run.
``find_peaks_derivative``
    Differentiation method: smooth the spectrum, then locate the zero crossings
    of the first derivative with a negative second derivative (or the minima of
    the second derivative itself, which resolves shoulders of overlapping
    peaks).
``find_peaks_cwt``
    Continuous wavelet transform (Ricker wavelet) ridge detection -- robust
    against noise and slowly varying baselines, at the cost of two tuning
    parameters (the explored scale range).

:func:`detect_peaks` is a convenience front-end that chains baseline
correction, smoothing, one of the detectors above, and the derived metrics
(prominence, FWHM, area, SNR) into :class:`~spectral_peaks.types.Peak` objects.
"""

from __future__ import annotations

import numpy as np

from ._peaks_utils import (
    filter_by_distance as _filter_by_distance,
    select_by_property as _select_by_property,
    unpack_interval as _unpack_interval,
    unpack_thresholds as _unpack_thresholds,
    validate_peaks as _validate_peaks,
)
from .metrics import estimate_noise, fwhm_from_sigma, peak_areas
from .smoothing import savgol_filter
from .types import Peak

__all__ = [
    "cwt",
    "detect_peaks",
    "find_peaks",
    "find_peaks_cwt",
    "find_peaks_derivative",
    "peak_prominences",
    "peak_widths",
    "ricker",
]


def _local_maxima_1d(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of local maxima plus the edges of the plateau each belongs to.

    Follows the ``scipy.signal`` convention: runs of equal samples touching an
    array end are not peaks, a plateau counts as a single peak placed at its
    midpoint, and a strictly monotonic signal has none at all.

    Returns ``(peaks, left_edges, right_edges)``, each of shape ``(k,)``.
    """
    n = y.size
    empty = np.empty(0, dtype=np.intp)
    if n < 3:
        return empty, empty.copy(), empty.copy()

    change = np.nonzero(y[1:] != y[:-1])[0] + 1  # right edge of each constant run
    starts = np.r_[0, change]
    ends = np.r_[change, n]  # exclusive
    peaks: list[int] = []
    left: list[int] = []
    right: list[int] = []
    for a, b in zip(starts, ends):
        if a == 0 or b == n:
            continue  # a plateau touching an array end is not a peak
        value = y[a]
        if y[a - 1] < value and y[b] < value:
            peaks.append((a + b - 1) // 2)
            left.append(int(a))
            right.append(int(b - 1))
    return (
        np.asarray(peaks, dtype=np.intp),
        np.asarray(left, dtype=np.intp),
        np.asarray(right, dtype=np.intp),
    )


def peak_prominences(
    y: np.ndarray, peaks: np.ndarray, wlen: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topographic prominence of every peak together with its two bases.

    The prominence is ``y[peak] - max(left_min, right_min)``, where each minimum
    is taken between the peak and the nearest *higher* sample in that direction
    (or the edge of the search window).  ``wlen`` limits the search to a window
    of that many samples centred on the peak, which keeps neighbouring peaks
    from borrowing each other's bases.

    Returns ``(prominences, left_bases, right_bases)``.
    """
    y = np.asarray(y, dtype=float)
    peaks = _validate_peaks(peaks, y.size)
    n = y.size
    prominences = np.empty(peaks.size, dtype=float)
    left_bases = np.empty(peaks.size, dtype=np.intp)
    right_bases = np.empty(peaks.size, dtype=np.intp)
    half = 0 if wlen is None else max(int(wlen), 1) // 2

    for k, peak in enumerate(peaks):
        peak = int(peak)
        value = y[peak]

        lo = max(0, peak - half) if wlen is not None else 0
        i = peak
        while i > lo:
            i -= 1
            if y[i] > value:
                break
        left_seg = y[i:peak]
        if left_seg.size:
            # scipy keeps the *last* sample attaining the minimum on the left side
            offset = left_seg.size - 1 - int(np.argmin(left_seg[::-1]))
            left_bases[k] = i + offset
        else:
            left_bases[k] = peak

        hi = min(n - 1, peak + half) if wlen is not None else n - 1
        j = peak
        while j < hi:
            j += 1
            if y[j] > value:
                break
        right_seg = y[peak + 1 : j + 1]
        right_bases[k] = peak + 1 + int(np.argmin(right_seg)) if right_seg.size else peak

        prominences[k] = value - max(y[left_bases[k]], y[right_bases[k]])
    return prominences, left_bases, right_bases


def _interpolate_crossing(
    y: np.ndarray, start: int, stop: int, height: float, direction: int
) -> float:
    """Fractional index where ``y`` crosses ``height`` when walking from ``start``.

    ``direction`` is -1 to walk towards lower indices, +1 towards higher ones.
    ``stop`` is the furthest index that may be inspected (inclusive) and is
    returned when the contour is never crossed.
    """
    if direction < 0:
        segment = y[stop : start + 1]
        below = segment <= height
        if not below.any():
            return float(stop)
        j = stop + int(np.nonzero(below)[0][-1])
        if j >= start:
            return float(j)
        denom = y[j + 1] - y[j]
        return float(j) if denom == 0.0 else j + (height - y[j]) / denom

    segment = y[start : stop + 1]
    below = segment <= height
    if not below.any():
        return float(stop)
    k = start + int(np.nonzero(below)[0][0])
    if k <= start:
        return float(k)
    denom = y[k - 1] - y[k]
    return float(k) if denom == 0.0 else k - (height - y[k]) / denom


def peak_widths(
    y: np.ndarray,
    peaks: np.ndarray,
    rel_height: float = 0.5,
    prominence_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    wlen: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Width of each peak measured ``rel_height`` of the way down from its top.

    With ``rel_height=0.5`` this is the full width at half prominence, which
    equals the FWHM whenever the prominence is the peak height above its local
    baseline.  Widths come back in samples, with the contour crossings
    linearly interpolated between the two bracketing samples.

    Returns ``(widths, width_heights, left_ips, right_ips)``.
    """
    y = np.asarray(y, dtype=float)
    peaks = _validate_peaks(peaks, y.size)
    if prominence_data is None:
        prominence_data = peak_prominences(y, peaks, wlen)
    prominences = np.asarray(prominence_data[0], dtype=float)
    left_bases = np.asarray(prominence_data[1], dtype=float)
    right_bases = np.asarray(prominence_data[2], dtype=float)
    if prominences.size != peaks.size:
        raise ValueError("prominence_data does not match the number of peaks")

    n = y.size
    widths = np.empty(peaks.size, dtype=float)
    width_heights = np.empty(peaks.size, dtype=float)
    left_ips = np.empty(peaks.size, dtype=float)
    right_ips = np.empty(peaks.size, dtype=float)
    for k, peak in enumerate(peaks):
        peak = int(peak)
        height = y[peak] - prominences[k] * rel_height
        left_ips[k] = (
            float(peak)
            if left_bases[k] >= peak
            else _interpolate_crossing(y, peak, max(0, int(left_bases[k])), height, -1)
        )
        right_ips[k] = (
            float(peak)
            if right_bases[k] <= peak
            else _interpolate_crossing(y, peak, min(n - 1, int(right_bases[k])), height, +1)
        )
        width_heights[k] = height
        widths[k] = right_ips[k] - left_ips[k]
    return widths, width_heights, left_ips, right_ips


def find_peaks(
    y: np.ndarray,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size=None,
) -> tuple[np.ndarray, dict]:
    """Find peaks as local maxima, optionally filtered by their properties.

    Mirrors ``scipy.signal.find_peaks``: every filter accepts either a number
    (lower bound), a ``(min, max)`` pair, or a per-sample array, and the filters
    are applied in the order ``plateau_size``, ``height``, ``threshold``,
    ``distance``, ``prominence``, ``width``.  Requested properties are returned
    in the ``properties`` dict, aligned with ``peaks``.

    Parameters
    ----------
    y
        One-dimensional signal.
    height, threshold, prominence, width
        Inclusive value range of the corresponding property.
    distance
        Minimum horizontal distance in samples between accepted peaks; the
        smaller of two too-close peaks is dropped first.
    wlen
        Window in samples used for the prominence/width search around a peak.
    rel_height
        Relative height at which ``width`` is measured (0.5 gives FWHM).
    plateau_size
        Allowed range of plateau lengths in samples.

    Returns
    -------
    peaks, properties
        ``peaks`` holds the sample indices, ``properties`` the arrays of the
        properties that were asked for.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if distance is not None and distance < 1:
        raise ValueError("distance must be greater or equal to 1")

    peaks, left_edges, right_edges = _local_maxima_1d(y)
    properties: dict[str, np.ndarray] = {}

    def _reduce(keep: np.ndarray) -> None:
        nonlocal peaks
        peaks = peaks[keep]
        for key, array in properties.items():
            properties[key] = array[keep]

    if plateau_size is not None:
        sizes = right_edges - left_edges + 1
        properties["plateau_sizes"] = sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        lo, hi = _unpack_interval(plateau_size, y, peaks)
        _reduce(_select_by_property(sizes, lo, hi))

    if height is not None:
        heights = y[peaks]
        lo, hi = _unpack_interval(height, y, peaks)
        properties["peak_heights"] = heights
        _reduce(_select_by_property(heights, lo, hi))

    if threshold is not None:
        lo, hi = _unpack_interval(threshold, y, peaks)
        keep, left_thresholds, right_thresholds = _unpack_thresholds(y, peaks, lo, hi)
        properties["left_thresholds"] = left_thresholds
        properties["right_thresholds"] = right_thresholds
        _reduce(keep)

    if distance is not None:
        _reduce(_filter_by_distance(peaks, y[peaks], float(distance)))

    if prominence is not None or width is not None:
        prom, left_bases, right_bases = peak_prominences(y, peaks, wlen=wlen)
        properties["prominences"] = prom
        properties["left_bases"] = left_bases
        properties["right_bases"] = right_bases

    if prominence is not None:
        lo, hi = _unpack_interval(prominence, y, peaks)
        _reduce(_select_by_property(properties["prominences"], lo, hi))

    if width is not None:
        widths, width_heights, left_ips, right_ips = peak_widths(
            y,
            peaks,
            rel_height=rel_height,
            prominence_data=(
                properties["prominences"],
                properties["left_bases"],
                properties["right_bases"],
            ),
        )
        properties["widths"] = widths
        properties["width_heights"] = width_heights
        properties["left_ips"] = left_ips
        properties["right_ips"] = right_ips
        lo, hi = _unpack_interval(width, y, peaks)
        _reduce(_select_by_property(widths, lo, hi))

    return peaks, properties
