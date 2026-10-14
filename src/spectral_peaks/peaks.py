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
from .baseline import correct_baseline
from .metrics import estimate_noise, peak_areas
from .smoothing import savgol_filter
from .types import Peak, PeakResult

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


def _filter_peaks(
    y: np.ndarray,
    peaks: np.ndarray,
    properties: dict[str, np.ndarray],
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen: int | None = None,
    rel_height: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Apply the property filters of :func:`find_peaks` to candidate ``peaks``.

    The filters run in the SciPy order (``height``, ``threshold``, ``distance``,
    ``prominence``, ``width``) and every computed property is trimmed to match
    the surviving peaks.  Shared with :func:`find_peaks_derivative`.
    """
    properties = dict(properties)

    def _reduce(keep: np.ndarray) -> None:
        nonlocal peaks
        peaks = peaks[keep]
        for key, array in properties.items():
            properties[key] = array[keep]

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

    if plateau_size is not None:
        sizes = right_edges - left_edges + 1
        lo, hi = _unpack_interval(plateau_size, y, peaks)
        keep = _select_by_property(sizes, lo, hi)
        peaks = peaks[keep]
        properties["plateau_sizes"] = sizes[keep]
        properties["left_edges"] = left_edges[keep]
        properties["right_edges"] = right_edges[keep]

    return _filter_peaks(
        y,
        peaks,
        properties,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
    )


def find_peaks_derivative(
    y: np.ndarray,
    criterion: str = "first",
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    delta: float = 1.0,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen: int | None = None,
    rel_height: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Peak detection through the derivatives of a Savitzky-Golay smoothed signal.

    Because the Savitzky-Golay filter returns a differentiable polynomial fit,
    differentiating it is far more stable than differentiating raw data.

    ``criterion="first"``
        A peak is where the smoothed first derivative crosses zero downwards
        while the second derivative is negative.  The crossing index is refined
        by linear interpolation, which yields sub-sample positions -- useful
        when the sampling is coarse compared with the peak width.
    ``criterion="second"``
        Peaks are the local minima of the second derivative (maxima of ``-d2``).
        This is the classical way to locate the components of overlapping or
        heavily broadened peaks, where the first derivative shows no separate
        zero crossing.

    The property filters (``height``, ``threshold``, ``distance``,
    ``prominence``, ``width``) are evaluated on the *smoothed* signal -- read
    ``y[peak]`` yourself if you need raw amplitudes.

    Returns ``(peaks, properties)`` where ``properties`` additionally contains
    ``positions`` (fractional sample positions), ``slopes`` (first derivative)
    and ``curvatures`` (second derivative) at each peak.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = y.size
    window = int(smooth_window)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 else n - 1)
    polyorder = int(smooth_polyorder)
    if window <= polyorder:
        polyorder = max(1, window - 1)

    smoothed = savgol_filter(y, window, polyorder, mode="reflect")
    d1 = savgol_filter(y, window, polyorder, deriv=1, delta=delta, mode="reflect")
    d2 = savgol_filter(y, window, polyorder, deriv=2, delta=delta, mode="reflect")

    if criterion in ("first", "first_derivative"):
        crossings = np.nonzero((d1[:-1] > 0.0) & (d1[1:] <= 0.0))[0] + 1
        crossings = crossings[d2[crossings] < 0.0]
        prev = np.maximum(crossings - 1, 0)
        denom = d1[prev] - d1[crossings]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(denom > 0.0, d1[prev] / np.where(denom > 0.0, denom, 1.0), 0.0)
        peaks = crossings
        positions = prev + np.clip(frac, 0.0, 1.0)
    elif criterion in ("second", "second_derivative"):
        candidates, _, _ = _local_maxima_1d(-d2)
        candidates = candidates[d2[candidates] < 0.0]
        peaks = candidates
        positions = candidates.astype(float)
    else:
        raise ValueError("criterion must be 'first' or 'second'")

    properties: dict[str, np.ndarray] = {
        "peak_heights": smoothed[peaks],
        "positions": positions,
        "slopes": d1[peaks],
        "curvatures": d2[peaks],
    }
    return _filter_peaks(
        smoothed,
        peaks,
        properties,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
    )


def ricker(points: int, a: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet of ``points`` samples and scale ``a``.

    This is the negative, unit-normalised second derivative of a Gaussian; it is
    the default analysing wavelet of :func:`find_peaks_cwt`.
    """
    points = int(points)
    a = float(a)
    if points < 1:
        raise ValueError("points must be at least 1")
    if a <= 0:
        raise ValueError("a must be positive")
    amplitude = 2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25)
    vec = np.arange(points, dtype=float) - (points - 1.0) / 2.0
    xsq = vec * vec
    wsq = a * a
    return amplitude * (1.0 - xsq / wsq) * np.exp(-0.5 * xsq / wsq)


def cwt(y: np.ndarray, widths, wavelet=ricker) -> np.ndarray:
    """Continuous wavelet transform of a 1-D signal.

    Returns a ``(len(widths), len(y))`` matrix; row ``i`` is the transform at
    scale ``widths[i]``, obtained by convolving the signal with the wavelet
    (``mode="same"``, wavelet length ``min(10 * width, len(y))`` as in
    ``scipy.signal.cwt``).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    widths = np.atleast_1d(np.asarray(widths, dtype=float))
    if widths.ndim != 1:
        raise ValueError("widths must be one-dimensional")
    if widths.size == 0:
        raise ValueError("widths must not be empty")
    out = np.empty((widths.size, y.size))
    for i, width in enumerate(widths):
        scale = max(2, int(round(float(width))))
        wavelet_data = wavelet(int(min(10 * scale, y.size)), scale)
        out[i] = np.convolve(y, wavelet_data, mode="same")
    return out


def find_peaks_cwt(
    y: np.ndarray,
    widths=None,
    snr: float = 2.0,
    noise_perc: float = 10.0,
    min_scales: int | None = None,
    min_distance: int = 1,
    refine: bool = True,
    wavelet=ricker,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen: int | None = None,
    rel_height: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Peak detection by ridge tracking in a continuous wavelet transform.

    The signal is transformed with a Ricker wavelet at every scale in
    ``widths``.  At each scale the ridges (local maxima of ``|CWT|``) that exceed
    ``snr`` times the ``noise_perc``-th percentile of that scale are collected,
    and a sample becomes a candidate only if it is a ridge at ``min_scales``
    different scales.  The surviving candidates are finally snapped to the local
    maximum of ``y`` within half of the largest scale.

    Because the wavelet responds to shape rather than to absolute height, this
    detector copes well with noise and with baselines that were not perfectly
    removed; it is the most robust and the slowest of the three detectors.

    Parameters
    ----------
    widths
        Wavelet scales in samples.  Defaults to ``1..10``; the useful range is
        roughly a quarter to a whole of the expected peak FWHM.
    snr
        Threshold in units of the per-scale noise level.
    noise_perc
        Percentile of ``|CWT|`` taken as the noise level of a scale.
    min_scales
        How many scales must agree on a candidate (default: 2 when more than
        two scales are explored, otherwise 1).
    min_distance
        Candidates closer than this many samples are merged, keeping the
        strongest.
    refine
        Snap candidates to the local maximum of the signal.

    Returns ``(peaks, properties)`` with ``properties['scales']`` (the largest
    scale at which each peak is a ridge, a relative width indicator),
    ``properties['cwt_response']`` and, when any property filter is requested,
    the usual ``height``/``prominence``/``width`` arrays.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = y.size
    if n < 5:
        raise ValueError("y is too short for wavelet detection")
    if widths is None:
        widths = np.arange(1, min(11, max(2, n // 3)))
    widths = np.atleast_1d(np.asarray(widths, dtype=float))
    if np.any(widths < 1):
        raise ValueError("widths must be >= 1")

    transform = np.abs(cwt(y, widths, wavelet=wavelet))
    counts = np.zeros(n, dtype=int)
    response = np.zeros(n)
    scale_of = np.zeros(n)
    for i, scale in enumerate(widths):
        row = transform[i]
        level = float(snr) * float(np.percentile(row, noise_perc))
        ridges, _, _ = _local_maxima_1d(row)
        ridges = ridges[row[ridges] >= level]
        if ridges.size == 0:
            continue
        counts[ridges] += 1
        np.maximum.at(response, ridges, row[ridges])
        np.maximum.at(scale_of, ridges, float(scale))

    if min_scales is None:
        min_scales = 2 if widths.size > 2 else 1
    candidates = np.nonzero(counts >= max(1, int(min_scales)))[0]
    if candidates.size and min_distance > 1:
        candidates = candidates[_filter_by_distance(candidates, response[candidates], min_distance)]

    if refine and candidates.size:
        half = max(1, int(np.ceil(widths.max() / 2.0)))
        refined = candidates.copy()
        for k, position in enumerate(candidates):
            lo = max(0, position - half)
            hi = min(n, position + half + 1)
            refined[k] = lo + int(np.argmax(y[lo:hi]))
        candidates = np.unique(refined)

    properties: dict[str, np.ndarray] = {
        "scales": scale_of[candidates],
        "cwt_response": response[candidates],
    }
    return _filter_peaks(
        y,
        candidates,
        properties,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
    )


def detect_peaks(
    y: np.ndarray,
    x=None,
    method: str = "prominence",
    baseline=None,
    baseline_kwargs: dict | None = None,
    smooth: int | None = None,
    smooth_polyorder: int = 3,
    noise: float | None = None,
    noise_method: str = "diff",
    rel_height: float = 0.5,
    wlen: int | None = None,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    plateau_size=None,
    derivative_criterion: str = "first",
    cwt_widths=None,
    cwt_snr: float = 2.0,
    cwt_min_scales: int | None = None,
    max_peaks: int | None = None,
    fit_model: str | None = None,
    fit_kwargs: dict | None = None,
) -> PeakResult:
    """One-call peak detection: baseline, smoothing, detection, metrics.

    The pipeline is

    1. optionally subtract a baseline: ``baseline="airpls"`` (or any name from
       :data:`spectral_peaks.baseline.BASELINE_METHODS`) with extra keyword
       arguments in ``baseline_kwargs``, or a precomputed baseline array;
    2. optionally smooth with a Savitzky-Golay filter of ``smooth`` samples;
    3. detect with ``method``: ``"prominence"``/``"local"``
       (:func:`find_peaks`), ``"derivative"`` (:func:`find_peaks_derivative`) or
       ``"cwt"`` (:func:`find_peaks_cwt`);
    4. measure prominence, the width at ``rel_height`` of the prominence, the
       area above the local baseline and the SNR of every peak.

    Parameters
    ----------
    y
        Signal. ``x`` is the matching coordinate axis; when it is omitted the
        sample index is used and widths/areas come out in samples.
    x
        Strictly increasing x axis of ``y``.
    method
        ``"prominence"`` (default), ``"derivative"`` or ``"cwt"``.
    noise
        Override the automatic noise estimate used for the SNR column.
    height, threshold, distance, prominence, width, plateau_size
        Filters forwarded to the detector (see :func:`find_peaks`).
    max_peaks
        Keep only the ``max_peaks`` most prominent peaks.
    fit_model
        When given (``"gaussian"``, ``"lorentzian"``, ``"pseudo_voigt"`` or
        ``"voigt"``) every peak is refined by a local least-squares fit, whose
        parameters are stored in ``Peak.fit``.

    Returns
    -------
    PeakResult
        Iterable over :class:`~spectral_peaks.types.Peak` objects, also carrying
        the processed signal, the baseline and the noise estimate.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = y.size
    xs = np.arange(n, dtype=float) if x is None else np.asarray(x, dtype=float)
    if xs.size != n:
        raise ValueError("x and y must have the same length")
    if x is not None and n > 1 and np.any(np.diff(xs) <= 0):
        raise ValueError("x must be strictly increasing")
    x_units = x is not None

    processed = y
    baseline_curve = None
    if baseline is not None:
        if isinstance(baseline, str):
            processed, baseline_curve = correct_baseline(
                y, method=baseline, return_baseline=True, **(baseline_kwargs or {})
            )
        else:
            baseline_curve = np.asarray(baseline, dtype=float)
            if baseline_curve.size != n:
                raise ValueError("baseline must have the same length as y")
            processed = y - baseline_curve

    if smooth:
        window = int(smooth)
        if window % 2 == 0:
            window += 1
        window = min(window, n if n % 2 else n - 1)
        if window > 2:
            processed = savgol_filter(
                processed, window, min(int(smooth_polyorder), window - 1), mode="reflect"
            )

    common = dict(
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
    )
    key = str(method).lower()
    positions = None
    if key in ("prominence", "local", "local_maxima", "find_peaks"):
        peaks, _ = find_peaks(processed, plateau_size=plateau_size, **common)
    elif key in ("derivative", "deriv", "differentiation"):
        peaks, props = find_peaks_derivative(
            processed,
            criterion=derivative_criterion,
            smooth_window=int(smooth) if smooth else 11,
            smooth_polyorder=smooth_polyorder,
            **common,
        )
        positions = props["positions"]
    elif key in ("cwt", "wavelet"):
        peaks, _ = find_peaks_cwt(
            processed,
            widths=cwt_widths,
            snr=cwt_snr,
            min_scales=cwt_min_scales,
            **common,
        )
    else:
        raise ValueError(
            f"method must be 'prominence', 'derivative' or 'cwt' (got {method!r})"
        )
    if positions is None:
        positions = peaks.astype(float)

    sigma = (
        float(estimate_noise(processed, method=noise_method)) if noise is None else float(noise)
    )
    prom, left_bases, right_bases = peak_prominences(processed, peaks, wlen=wlen)
    widths, _, left_ips, right_ips = peak_widths(
        processed,
        peaks,
        rel_height=rel_height,
        prominence_data=(prom, left_bases, right_bases),
    )
    if x_units:
        grid = np.arange(n, dtype=float)
        left_x = np.interp(left_ips, grid, xs)
        right_x = np.interp(right_ips, grid, xs)
        widths_x = right_x - left_x
        positions_x = np.interp(positions, grid, xs)
        areas = peak_areas(processed, left_ips, right_ips, x=xs)
    else:
        left_x = left_ips
        right_x = right_ips
        widths_x = widths
        positions_x = positions
        areas = peak_areas(processed, left_ips, right_ips)

    detected = [
        Peak(
            index=int(index),
            x=float(position),
            height=float(processed[int(index)]),
            prominence=float(p),
            left_base=int(lb),
            right_base=int(rb),
            width=float(w),
            left_ip=float(li),
            right_ip=float(ri),
            area=float(a),
            snr=float("inf") if sigma <= 0.0 else float(processed[int(index)] / sigma),
        )
        for index, position, p, lb, rb, w, li, ri, a in zip(
            peaks, positions_x, prom, left_bases, right_bases, widths_x, left_x, right_x, areas
        )
    ]

    if max_peaks is not None and len(detected) > int(max_peaks):
        def _rank(peak: Peak) -> float:
            return peak.prominence if peak.prominence == peak.prominence else peak.height

        strongest = sorted(detected, key=_rank, reverse=True)[: int(max_peaks)]
        detected = sorted(strongest, key=lambda peak: peak.index)

    result = PeakResult(
        peaks=detected,
        x=xs,
        y=processed,
        method=key,
        baseline=baseline_curve,
        noise=sigma,
    )

    if fit_model is not None:
        from .fitting import fit_peaks  # imported here to keep module deps flat

        fit_peaks(result, model=fit_model, **(fit_kwargs or {}))
    return result
