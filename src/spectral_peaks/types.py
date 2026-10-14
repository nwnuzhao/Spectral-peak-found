"""Data containers returned by the detectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

__all__ = ["Peak", "peaks_to_arrays"]

_NAN = float("nan")


@dataclass
class Peak:
    """A detected spectral peak and the properties measured for it.

    Attributes
    ----------
    index
        Sample index of the peak maximum in the (possibly corrected) signal.
    x
        Position of the peak on the x axis (same as ``index`` when no x was
        supplied).
    height
        Signal value at the peak maximum.
    prominence
        Topographic prominence, i.e. the height above the higher of the two
        bases (see :func:`spectral_peaks.peaks.peak_prominences`).
    left_base, right_base
        Sample indices of the two bases that define the prominence.
    width
        Full width at the requested relative height (``0.5`` -> FWHM), in x
        units when an x axis is available, otherwise in samples.
    left_ip, right_ip
        Fractional sample positions where the contour at ``width`` is crossed.
    area
        Integrated area above the local baseline, in x*y units.
    snr
        ``height / noise`` using the noise estimate of the detector call.
    fit
        Optional dict of fitted model parameters (filled in by
        :func:`spectral_peaks.fitting.fit_peaks`).
    """

    index: int
    x: float
    height: float
    prominence: float = _NAN
    left_base: int = -1
    right_base: int = -1
    width: float = _NAN
    left_ip: float = _NAN
    right_ip: float = _NAN
    area: float = _NAN
    snr: float = _NAN
    fit: dict | None = None

    @property
    def fwhm(self) -> float:
        """Alias for :attr:`width` when the width was measured at half height."""
        return self.width

    def to_dict(self) -> dict:
        """Plain-dict view of the peak (handy for CSV/JSON export)."""
        return asdict(self)

    def __repr__(self) -> str:  # compact, readable in notebooks
        return (
            f"Peak(index={self.index}, x={self.x:g}, height={self.height:.6g}, "
            f"prominence={self.prominence:.6g}, width={self.width:.4g}, "
            f"area={self.area:.4g}, snr={self.snr:.3g})"
        )


def peaks_to_arrays(peaks: list[Peak]) -> dict[str, np.ndarray]:
    """Convert a list of :class:`Peak` into a dict of NumPy arrays."""
    names = [f.name for f in fields(Peak) if f.name != "fit"]
    return {name: np.array([getattr(p, name) for p in peaks]) for name in names}
