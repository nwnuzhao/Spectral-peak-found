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
        Integrated area of the processed signal over a ``+-area_span * width``
        window around the maximum, above the straight line joining the window
        ends, in x*y units.  Accurate to a few percent for resolved peaks; for
        blended peaks use the model area from
        :func:`spectral_peaks.fitting.fit_peaks` instead.
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


@dataclass
class PeakResult:
    """A detection run: the peaks plus the signal they were found in.

    Iterating a ``PeakResult`` yields its :class:`Peak` objects, so
    ``for peak in result`` and ``len(result)`` behave as expected.
    """

    peaks: list[Peak]
    x: np.ndarray
    y: np.ndarray
    method: str = ""
    baseline: np.ndarray | None = None
    noise: float = _NAN

    def __len__(self) -> int:
        return len(self.peaks)

    def __iter__(self):
        return iter(self.peaks)

    def __getitem__(self, item):
        return self.peaks[item]

    def arrays(self) -> dict[str, np.ndarray]:
        """Peak properties as a dict of arrays."""
        return peaks_to_arrays(self.peaks)

    def indices(self) -> np.ndarray:
        return np.array([p.index for p in self.peaks], dtype=int)

    def positions(self) -> np.ndarray:
        return np.array([p.x for p in self.peaks], dtype=float)

    def heights(self) -> np.ndarray:
        return np.array([p.height for p in self.peaks], dtype=float)

    def table(self, digits: int = 4) -> str:
        """Fixed-width table of the detected peaks, ready to print."""
        header = ("index", "x", "height", "prominence", "width", "area", "snr")
        rows = [header] + [
            (
                f"{p.index:d}",
                f"{p.x:.{digits}g}",
                f"{p.height:.{digits}g}",
                f"{p.prominence:.{digits}g}",
                f"{p.width:.{digits}g}",
                f"{p.area:.{digits}g}",
                f"{p.snr:.{digits}g}",
            )
            for p in self.peaks
        ]
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        lines = [
            "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) for row in rows
        ]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write the peaks to a CSV file (header included)."""
        import csv

        columns = list(peaks_to_arrays(self.peaks)) if self.peaks else [
            f.name for f in fields(Peak) if f.name != "fit"
        ]
        arrays = peaks_to_arrays(self.peaks)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in zip(*(arrays[c] for c in columns)):
                writer.writerow(row)
