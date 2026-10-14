"""Peak profile models and a self-contained Levenberg-Marquardt fitter.

No SciPy: the optimiser is a plain implementation of the damped Gauss-Newton
(Levenberg-Marquardt) algorithm with a central-difference Jacobian, and the
profiles are closed-form expressions except for the Voigt profile, which is
evaluated as the numerical convolution of a Gaussian with a Lorentzian by FFT.

Every profile is parameterised so that ``amplitude`` is the height of the peak
above its own baseline:

===============  ===============================================================
``gaussian``     ``amplitude * exp(-0.5 * ((x - center) / sigma)**2)``
``lorentzian``   ``amplitude / (1 + ((x - center) / gamma)**2)``
``pseudo_voigt`` ``eta * lorentzian + (1 - eta) * gaussian``, ``eta`` in [0, 1]
``voigt``        convolution of a Gaussian (``sigma``) with a Lorentzian (``gamma``)
===============  ===============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "MODELS",
    "FitResult",
    "PeakModel",
    "fit_peak",
    "fit_peaks",
    "fit_region",
    "gaussian",
    "get_model",
    "lm_fit",
    "lorentzian",
    "pseudo_voigt",
    "voigt",
]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_FWHM_OVER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile; ``sigma`` is the standard deviation."""
    x = np.asarray(x, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Lorentzian profile; ``gamma`` is the half width at half maximum."""
    x = np.asarray(x, dtype=float)
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    eta: float,
) -> np.ndarray:
    """Linear mixture of a Gaussian and a Lorentzian of equal height.

    ``eta = 0`` is purely Gaussian, ``eta = 1`` purely Lorentzian.
    """
    return eta * lorentzian(x, amplitude, center, gamma) + (1.0 - eta) * gaussian(
        x, amplitude, center, sigma
    )


def _voigt_profile(offsets: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """Unit-height Voigt profile on an odd, uniformly spaced ``offsets`` grid."""
    n = offsets.size
    gaussian_kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()
    lorentz_kernel = gamma * gamma / (offsets * offsets + gamma * gamma)
    lorentz_kernel /= lorentz_kernel.sum()
    size = 2 * n  # zero padding -> the FFT computes a linear convolution
    spectrum = np.fft.rfft(gaussian_kernel, size) * np.fft.rfft(lorentz_kernel, size)
    full = np.fft.irfft(spectrum, size)
    start = n - 1 - (n // 2)
    profile = full[start : start + n]
    return profile / profile.max()


def voigt(
    x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float
) -> np.ndarray:
    """Voigt profile as the numerical convolution of a Gaussian and a Lorentzian.

    ``x`` must be uniformly spaced with an odd number of samples; the profile is
    normalised so that its maximum equals ``amplitude``.
    """
    x = np.asarray(x, dtype=float)
    if x.size % 2 == 0:
        raise ValueError("voigt requires an odd number of samples on a uniform grid")
    steps = np.diff(x)
    if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        raise ValueError("voigt requires a uniformly spaced grid")
    return amplitude * _voigt_profile(
        x - center, max(float(sigma), 1e-12), max(float(gamma), 1e-12)
    )


def _voigt_metrics(params: Sequence[float]) -> tuple[float, float]:
    """Numerical FWHM and area of a Voigt profile (fine local grid)."""
    amplitude, _, sigma, gamma = (float(v) for v in params[:4])
    sigma, gamma = max(abs(sigma), 1e-12), max(abs(gamma), 1e-12)
    span = 20.0 * max(sigma, gamma)
    grid = np.linspace(-span, span, 4001)
    profile = amplitude * _voigt_profile(grid, sigma, gamma)
    step = grid[1] - grid[0]
    area = float(np.sum(0.5 * (profile[1:] + profile[:-1])) * step)
    half = 0.5 * profile.max()
    above = np.nonzero(profile >= half)[0]
    if above.size < 2:
        return 0.0, area
    left, right = int(above[0]), int(above[-1])
    lo = grid[left - 1] + (half - profile[left - 1]) / (profile[left] - profile[left - 1]) * step
    hi = grid[right] + (half - profile[right]) / (profile[right + 1] - profile[right]) * step
    return float(hi - lo), area


@dataclass(frozen=True)
class PeakModel:
    """A profile function together with the metadata needed to fit it."""

    name: str
    func: Callable[..., np.ndarray]
    param_names: tuple[str, ...]
    init_width: Callable[[float], tuple[float, ...]]
    fwhm: Callable[[Sequence[float]], float]
    area: Callable[[Sequence[float]], float]

    @property
    def n_params(self) -> int:
        return len(self.param_names)


def _gaussian_fwhm(params: Sequence[float]) -> float:
    return float(_FWHM_OVER_SIGMA * params[2])


def _gaussian_area(params: Sequence[float]) -> float:
    return float(params[0] * params[2] * _SQRT_2PI)


def _lorentzian_fwhm(params: Sequence[float]) -> float:
    return float(2.0 * params[2])


def _lorentzian_area(params: Sequence[float]) -> float:
    return float(params[0] * np.pi * params[2])


def _pseudo_voigt_fwhm(params: Sequence[float]) -> float:
    eta = float(np.clip(params[4], 0.0, 1.0))
    return float(eta * 2.0 * params[3] + (1.0 - eta) * _FWHM_OVER_SIGMA * params[2])


def _pseudo_voigt_area(params: Sequence[float]) -> float:
    eta = float(np.clip(params[4], 0.0, 1.0))
    return float(eta * _lorentzian_area(params) + (1.0 - eta) * _gaussian_area(params))


MODELS: dict[str, PeakModel] = {
    "gaussian": PeakModel(
        name="gaussian",
        func=gaussian,
        param_names=("amplitude", "center", "sigma"),
        init_width=lambda fwhm: (fwhm / _FWHM_OVER_SIGMA,),
        fwhm=_gaussian_fwhm,
        area=_gaussian_area,
    ),
    "lorentzian": PeakModel(
        name="lorentzian",
        func=lorentzian,
        param_names=("amplitude", "center", "gamma"),
        init_width=lambda fwhm: (fwhm / 2.0,),
        fwhm=_lorentzian_fwhm,
        area=_lorentzian_area,
    ),
    "pseudo_voigt": PeakModel(
        name="pseudo_voigt",
        func=pseudo_voigt,
        param_names=("amplitude", "center", "sigma", "gamma", "eta"),
        init_width=lambda fwhm: (fwhm / _FWHM_OVER_SIGMA, fwhm / 2.0, 0.5),
        fwhm=_pseudo_voigt_fwhm,
        area=_pseudo_voigt_area,
    ),
    "voigt": PeakModel(
        name="voigt",
        func=voigt,
        param_names=("amplitude", "center", "sigma", "gamma"),
        init_width=lambda fwhm: (fwhm / 2.5, fwhm / 2.5),
        fwhm=lambda params: _voigt_metrics(params)[0],
        area=lambda params: _voigt_metrics(params)[1],
    ),
}


def get_model(model) -> PeakModel:
    """Resolve a model name (or pass a :class:`PeakModel` through unchanged)."""
    if isinstance(model, PeakModel):
        return model
    try:
        return MODELS[str(model).lower()]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; available: {sorted(MODELS)}") from None


def lm_fit(
    func: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    p0: Sequence[float],
    weights: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
    lam0: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    """Levenberg-Marquardt least-squares fit of a generic model function.

    ``func`` is called as ``func(x, *params)``.  Derivatives are approximated by
    central differences, so any callable works without analytic Jacobians.

    Returns ``(params, errors, chi2, n_iter, converged)`` where ``errors`` are
    the 1-sigma standard errors from the covariance matrix ``(J^T J)^{-1} s^2``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.array(p0, dtype=float)
    if params.ndim != 1:
        raise ValueError("p0 must be one-dimensional")
    n_params = params.size
    scale = np.ones_like(y) if weights is None else np.sqrt(np.asarray(weights, dtype=float))

    def residual(values: np.ndarray) -> np.ndarray:
        return (y - func(x, *values)) * scale

    resid = residual(params)
    chi2 = float(resid @ resid)
    lam = float(lam0)
    converged = False
    iterations = 0

    for iterations in range(1, int(max_iter) + 1):
        jacobian = np.empty((x.size, n_params))
        for j in range(n_params):
            step = 1e-6 * max(abs(params[j]), 1.0)
            up, down = params.copy(), params.copy()
            up[j] += step
            down[j] -= step
            jacobian[:, j] = (func(x, *down) - func(x, *up)) / (2.0 * step) * scale

        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ resid
        diagonal = np.diag(normal).copy()
        diagonal[diagonal <= 0.0] = 1.0

        improved = False
        for _ in range(30):
            matrix = normal + lam * np.diag(diagonal)
            try:
                step_vec = -np.linalg.solve(matrix, gradient)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            trial_params = params + step_vec
            trial_resid = residual(trial_params)
            trial_chi2 = float(trial_resid @ trial_resid)
            if trial_chi2 < chi2:
                relative = (chi2 - trial_chi2) / max(chi2, 1e-300)
                params, resid, chi2 = trial_params, trial_resid, trial_chi2
                lam = max(lam / 3.0, 1e-14)
                improved = True
                converged = relative < tol
                break
            lam *= 10.0
            if lam > 1e14:
                break
        if not improved or converged:
            break

    jacobian = np.empty((x.size, n_params))
    for j in range(n_params):
        step = 1e-6 * max(abs(params[j]), 1.0)
        up, down = params.copy(), params.copy()
        up[j] += step
        down[j] -= step
        jacobian[:, j] = (func(x, *down) - func(x, *up)) / (2.0 * step) * scale
    dof = max(1, x.size - n_params)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * (chi2 / dof)
    errors = np.sqrt(np.abs(np.diag(covariance)))
    return params, errors, chi2, iterations, converged


@dataclass
class FitResult:
    """Outcome of a multi-peak least-squares fit of a spectrum region."""

    model: PeakModel
    n_peaks: int
    baseline_order: int | None
    params: np.ndarray
    errors: np.ndarray
    names: list[str]
    x: np.ndarray
    y: np.ndarray
    residual: np.ndarray
    chi2: float
    reduced_chi2: float
    n_iter: int
    converged: bool
    x_shift: float = 0.0
    x_scale: float = 1.0

    def curve(self) -> np.ndarray:
        """Model evaluated on the fitted grid (peaks plus baseline)."""
        return self.y - self.residual

    def background(self) -> np.ndarray:
        """Baseline component of the model.

        The polynomial is evaluated in the normalised coordinate
        ``(x - x_shift) / x_scale``, which keeps the coefficients well scaled
        for x axes spanning several orders of magnitude.
        """
        if self.baseline_order is None:
            return np.zeros_like(self.x)
        offset = self.n_peaks * self.model.n_params
        return np.polyval(self.params[offset:], (self.x - self.x_shift) / self.x_scale)

    def peaks(self) -> list[dict]:
        """Per-peak parameters, standard errors, FWHM and analytic area."""
        blocks = []
        per = self.model.n_params
        for i in range(self.n_peaks):
            values = self.params[i * per : (i + 1) * per]
            errors = self.errors[i * per : (i + 1) * per]
            block = {name: float(v) for name, v in zip(self.model.param_names, values)}
            block.update(
                {f"{name}_err": float(e) for name, e in zip(self.model.param_names, errors)}
            )
            block["fwhm"] = float(self.model.fwhm(values))
            block["area"] = float(self.model.area(values))
            blocks.append(block)
        return blocks

    def as_dict(self) -> dict:
        return {
            "model": self.model.name,
            "n_peaks": self.n_peaks,
            "params": self.params.tolist(),
            "errors": self.errors.tolist(),
            "names": self.names,
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "peaks": self.peaks(),
        }


def fit_region(
    x: np.ndarray,
    y: np.ndarray,
    centers,
    model="gaussian",
    fwhm_guesses=None,
    amplitudes=None,
    baseline_order: int | None = 1,
    max_iter: int = 200,
    tol: float = 1e-10,
    weights: np.ndarray | None = None,
) -> FitResult:
    """Fit several peaks plus a polynomial background simultaneously.

    This is the routine to use when peaks overlap: all parameters are refined
    together against the whole region, so the components of a blended band are
    separated properly instead of being fitted one by one.

    Parameters
    ----------
    x, y
        Region of the spectrum to fit.
    centers
        Initial peak positions (in x units); one peak per entry.
    model
        Profile name (``"gaussian"``, ``"lorentzian"``, ``"pseudo_voigt"``,
        ``"voigt"``) or a :class:`PeakModel`.
    fwhm_guesses, amplitudes
        Optional initial widths (in x units) and heights, one per centre.
    baseline_order
        Degree of the added polynomial background (``None`` disables it).
    weights
        Optional per-sample weights (e.g. ``1 / sigma**2``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 3:
        raise ValueError("at least three samples are needed")
    spec = get_model(model)
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    n_peaks = centers.size
    if n_peaks == 0:
        raise ValueError("at least one peak centre is required")
    per = spec.n_params

    x_shift = float(np.mean(x))
    x_scale = float(np.std(x)) or 1.0
    normalized = (x - x_shift) / x_scale

    if baseline_order is None:
        background_coef = np.zeros(0)
        background = np.zeros_like(x)
    else:
        background_coef = np.polyfit(normalized, y, int(baseline_order))
        background = np.polyval(background_coef, normalized)

    if fwhm_guesses is None:
        span = float(x.max() - x.min()) / max(n_peaks, 1) / 4.0
        fwhm_guesses = np.full(n_peaks, span if span > 0 else 1.0)
    fwhm_guesses = np.atleast_1d(np.asarray(fwhm_guesses, dtype=float))
    if fwhm_guesses.size != n_peaks:
        raise ValueError("fwhm_guesses must have one entry per centre")

    if amplitudes is None:
        floor = 1e-6 * max(float(np.ptp(y)), 1.0)
        amplitudes = np.array(
            [
                max(float(y[int(np.argmin(np.abs(x - c)))] - background[int(np.argmin(np.abs(x - c)))]), floor)
                for c in centers
            ]
        )
    amplitudes = np.atleast_1d(np.asarray(amplitudes, dtype=float))
    if amplitudes.size != n_peaks:
        raise ValueError("amplitudes must have one entry per centre")

    initial: list[float] = []
    for amplitude, center, fwhm in zip(amplitudes, centers, fwhm_guesses):
        initial.extend([float(amplitude), float(center), *spec.init_width(float(fwhm))])
    initial.extend(float(c) for c in background_coef)

    def curve(xx, *params):
        values = np.zeros_like(xx, dtype=float)
        for i in range(n_peaks):
            values += spec.func(xx, *params[i * per : (i + 1) * per])
        if baseline_order is not None:
            values += np.polyval(np.asarray(params[n_peaks * per :], dtype=float),
                                 (xx - x_shift) / x_scale)
        return values

    params, errors, chi2, n_iter, converged = lm_fit(
        curve, x, y, initial, weights=weights, max_iter=max_iter, tol=tol
    )
    residual = y - curve(x, *params)
    dof = max(1, x.size - params.size)
    names = [f"p{i}_{name}" for i in range(n_peaks) for name in spec.param_names]
    if baseline_order is not None:
        names += [f"baseline_{j}" for j in range(int(baseline_order) + 1)]
    return FitResult(
        model=spec,
        n_peaks=n_peaks,
        baseline_order=baseline_order,
        params=params,
        errors=errors,
        names=names,
        x=x,
        y=y,
        residual=residual,
        chi2=chi2,
        reduced_chi2=chi2 / dof,
        n_iter=n_iter,
        converged=converged,
        x_shift=x_shift,
        x_scale=x_scale,
    )


def fit_peak(
    x: np.ndarray,
    y: np.ndarray,
    center: float | None = None,
    model="gaussian",
    fwhm: float | None = None,
    amplitude: float | None = None,
    half_window: float | None = None,
    baseline_order: int | None = 0,
    **kwargs,
) -> FitResult:
    """Fit one peak inside a window around ``center``.

    The window defaults to ``+-2 * fwhm`` (or a quarter of the full x span when
    no width guess is available).  Remaining keyword arguments go to
    :func:`fit_region`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if center is None:
        center = float(x[int(np.argmax(y))])
    if half_window is None:
        half_window = 2.0 * fwhm if fwhm else 0.25 * float(x.max() - x.min())
    mask = np.abs(x - center) <= abs(float(half_window))
    if mask.sum() < 5:
        mask = np.ones_like(x, dtype=bool)
    if get_model(model).name == "voigt" and mask.sum() % 2 == 0:
        mask[np.argmax(mask)] = False  # voigt needs an odd, uniform grid
    return fit_region(
        x[mask],
        y[mask],
        [center],
        model=model,
        fwhm_guesses=None if fwhm is None else [fwhm],
        amplitudes=None if amplitude is None else [amplitude],
        baseline_order=baseline_order,
        **kwargs,
    )


def fit_peaks(
    result,
    model="gaussian",
    baseline_order: int | None = 1,
    joint: bool = True,
    update: bool = True,
    window: float | None = None,
    max_iter: int = 100,
    weights: np.ndarray | None = None,
):
    """Refine every peak of a :class:`~spectral_peaks.types.PeakResult` by fitting.

    Each peak is fitted inside a local window of the (already baseline-corrected
    and smoothed) signal; with ``joint=True`` peaks whose windows overlap are
    refined *together* as a sum of profiles with a shared polynomial background,
    which is what makes the fit able to separate blended peaks.

    The fitted parameters, their standard errors, the FWHM and the analytic peak
    area are stored in ``Peak.fit``.  With ``update=True`` the peak position,
    width and area are overwritten with the fitted values.

    ``window`` sets the half-width of the fitting region in x units; the default
    is ``2 * measured width`` per peak (at least five samples).
    """
    peaks = list(getattr(result, "peaks", []))
    if not peaks:
        return result
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    spec = get_model(model)
    full_half = 0.25 * float(x.max() - x.min())

    spans: list[tuple[int, int]] = []
    for peak in peaks:
        if window is not None:
            half = abs(float(window))
        elif peak.width == peak.width and peak.width > 0:
            half = 2.0 * float(peak.width)
        else:
            half = full_half
        lo = int(np.searchsorted(x, peak.x - half, side="left"))
        hi = int(np.searchsorted(x, peak.x + half, side="right")) - 1
        lo = max(0, min(lo, x.size - 1))
        hi = max(lo, min(hi, x.size - 1))
        spans.append((lo, hi))

    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(peaks)):
        if joint and spans[i][0] <= spans[current[-1]][1]:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    for group in groups:
        lo = min(spans[i][0] for i in group)
        hi = max(spans[i][1] for i in group)
        xw = x[lo : hi + 1]
        yw = y[lo : hi + 1]
        if spec.name == "voigt" and xw.size % 2 == 0:
            xw, yw = xw[:-1], yw[:-1]
        if xw.size < spec.n_params + 1 + (0 if baseline_order is None else int(baseline_order) + 1):
            continue

        widths = [
            peaks[i].width
            if peaks[i].width == peaks[i].width and peaks[i].width > 0
            else max(float(xw.max() - xw.min()) / 4.0, 1e-12)
            for i in group
        ]
        fit = fit_region(
            xw,
            yw,
            [peaks[i].x for i in group],
            model=spec,
            fwhm_guesses=widths,
            amplitudes=[float(peaks[i].height) for i in group],
            baseline_order=baseline_order,
            max_iter=max_iter,
            weights=None if weights is None else np.asarray(weights, dtype=float)[lo : lo + xw.size],
        )
        blocks = fit.peaks()
        for position, index in enumerate(group):
            block = dict(blocks[position])
            block.update(
                {
                    "model": fit.model.name,
                    "reduced_chi2": fit.reduced_chi2,
                    "n_iter": fit.n_iter,
                    "converged": fit.converged,
                    "window": (float(xw[0]), float(xw[-1])),
                }
            )
            peaks[index].fit = block
            if update:
                peaks[index].x = float(block["center"])
                peaks[index].width = float(block["fwhm"])
                peaks[index].area = float(block["area"])
    return result