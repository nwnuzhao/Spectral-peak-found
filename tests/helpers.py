"""Helpers shared by the test suite (synthetic spectra and peak matching).

Kept inside ``tests/`` rather than in the package so the public API stays
minimal; if a ``spectral_peaks.synthetic`` module is added later these helpers
can be promoted to it without touching the assertions.
"""

from __future__ import annotations

import numpy as np

FWHM_OVER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))


def gaussian(x, center: float, height: float, fwhm: float) -> np.ndarray:
    """Gaussian peak of the requested height and FWHM."""
    x = np.asarray(x, dtype=float)
    sigma = fwhm / FWHM_OVER_SIGMA
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def exponential_background(x, amplitude: float = 1.5, decay: float = 700.0) -> np.ndarray:
    """Smoothly decaying fluorescence-like background."""
    x = np.asarray(x, dtype=float)
    return amplitude * np.exp(-(x - x.min()) / decay) + 0.08


def make_spectrum(
    peak_specs,
    n: int = 2001,
    x_range: tuple[float, float] = (400.0, 2400.0),
    background="exponential",
    noise: float = 0.0,
    seed: int = 0,
):
    """Build a synthetic spectrum.

    ``peak_specs`` is a sequence of ``(center, height, fwhm)`` triples.
    Returns ``(x, y, truth)`` where ``truth`` is a list of dicts holding the
    true centre, height, FWHM and analytic area (``height * sigma * sqrt(2pi)``)
    of every peak.
    """
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.zeros_like(x)
    truth = []
    for center, height, fwhm in peak_specs:
        y += gaussian(x, center, height, fwhm)
        truth.append(
            {
                "center": float(center),
                "height": float(height),
                "fwhm": float(fwhm),
                "area": float(height * (fwhm / FWHM_OVER_SIGMA) * np.sqrt(2.0 * np.pi)),
            }
        )
    baseline = np.zeros_like(x)
    if callable(background):
        baseline = np.asarray(background(x), dtype=float)
    elif background == "exponential":
        baseline = exponential_background(x)
    y = y + baseline
    if noise:
        y = y + np.random.default_rng(seed).normal(scale=noise, size=x.size)
    return x, y, truth


def match_peaks(found, truth, tol: float):
    """Nearest-neighbour matching of detected positions against the truth.

    Returns ``(n_matched, n_false_positives, max_position_error)``.
    """
    found = np.sort(np.asarray(found, dtype=float))
    truth = np.asarray(truth, dtype=float)
    matched = 0
    errors: list[float] = []
    used: set[int] = set()
    for target in truth:
        if found.size == 0:
            break
        distances = np.abs(found - target)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol and nearest not in used:
            used.add(nearest)
            matched += 1
            errors.append(float(distances[nearest]))
    false_positives = int(found.size - len(used))
    return matched, false_positives, (max(errors) if errors else float("nan"))
