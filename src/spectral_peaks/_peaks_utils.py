"""Small helpers shared by the peak-finding routines.

Kept in a separate module so that :mod:`spectral_peaks.peaks` reads as the
algorithmic code it is meant to be.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["unpack_interval", "unpack_thresholds", "select_by_property", "filter_by_distance"]


def unpack_interval(
    interval: Any, y: np.ndarray, peaks: np.ndarray
) -> tuple[Any, Any]:
    """Split a ``(min, max)``-style filter argument into its two bounds.

    ``interval`` may be a scalar (interpreted as the lower bound alone), a
    two-element sequence, or an array of per-sample bounds.  Array bounds are
    reduced to the peaks exactly like ``scipy.signal.find_peaks`` does.
    """
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = interval, None
    if isinstance(imin, np.ndarray):
        if imin.size != y.size:
            raise ValueError("array size of lower interval border must match y")
        imin = imin[peaks]
    if isinstance(imax, np.ndarray):
        if imax.size != y.size:
            raise ValueError("array size of upper interval border must match y")
        imax = imax[peaks]
    return imin, imax


def select_by_property(properties: np.ndarray, pmin: Any, pmax: Any) -> np.ndarray:
    """Boolean mask of peaks whose property lies inside ``[pmin, pmax]``."""
    properties = np.asarray(properties, dtype=float)
    keep = np.ones(properties.size, dtype=bool)
    if pmin is not None:
        keep &= pmin <= properties
    if pmax is not None:
        keep &= properties <= pmax
    return keep


def unpack_thresholds(
    y: np.ndarray, peaks: np.ndarray, tmin: Any, tmax: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the minimum vertical distance of each peak to its neighbours.

    Returns ``(keep, left_thresholds, right_thresholds)``.
    """
    stacked = np.vstack([y[peaks] - y[peaks - 1], y[peaks] - y[peaks + 1]])
    keep = np.ones(len(peaks), dtype=bool)
    if tmin is not None:
        keep &= tmin <= stacked.min(axis=0)
    if tmax is not None:
        keep &= stacked.max(axis=0) <= tmax
    return keep, stacked[0], stacked[1]


def filter_by_distance(peaks: np.ndarray, heights: np.ndarray, distance: float) -> np.ndarray:
    """Greedily drop peaks that are closer than ``distance`` samples to a taller one.

    Peaks are visited from the tallest to the shortest; a peak survives only if
    no already-accepted peak lies within ``distance`` samples of it (distance is
    rounded up to whole samples).  This matches
    ``scipy.signal.find_peaks(..., distance=...)``.

    Peaks of exactly equal height form a tie: they are visited in the order
    returned by :func:`numpy.argsort`, which is what SciPy does as well, so the
    selection is reproducible for a given input and NumPy version.  Sort the
    input yourself first if you need a different tie-break rule.
    """
    if peaks.size == 0:
        return np.ones(0, dtype=bool)
    min_distance = int(np.ceil(float(distance)))
    keep = np.ones(peaks.size, dtype=bool)

    def _drop_neighbours(j: int) -> None:
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < min_distance:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < peaks.size and peaks[k] - peaks[j] < min_distance:
            keep[k] = False
            k += 1

    for j in np.argsort(heights)[::-1]:  # highest priority first
        if not keep[j]:
            continue
        _drop_neighbours(int(j))
    return keep


def validate_peaks(peaks: np.ndarray, n: int) -> np.ndarray:
    """Cast ``peaks`` to an integer index array and check it is in range."""
    peaks = np.asarray(peaks)
    if peaks.dtype.kind == "f":
        if not np.all(np.equal(np.mod(peaks, 1), 0)):
            raise ValueError("peak positions must be integers")
    peaks = peaks.astype(np.intp)
    if peaks.ndim != 1:
        raise ValueError("peaks must be one-dimensional")
    if peaks.size and (peaks.min() < 0 or peaks.max() >= n):
        raise ValueError("peak positions must lie within the signal")
    return peaks
