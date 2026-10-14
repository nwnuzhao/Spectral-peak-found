# Spectral-peaks

**Spectral peak finding implemented from scratch with pure Python + NumPy — no SciPy
required.**

Baseline correction, Savitzky–Golay smoothing, three independent peak detectors, peak
metrics and a Levenberg–Marquardt profile fitter in one small package. SciPy is used
only by the test suite to *cross-validate* the results, and those tests skip
automatically when SciPy is not installed.

```python
import numpy as np
from spectral_peaks import detect_peaks

# synthetic spectrum: exponential background + four Gaussians + noise
x = np.linspace(400.0, 2400.0, 2001)
y = 1.5 * np.exp(-(x - x[0]) / 700.0)
for centre, height, fwhm in [(520, 1.0, 24), (905, 1.6, 22), (1300, 0.9, 45), (1700, 1.2, 36)]:
    y += height * np.exp(-0.5 * ((x - centre) / (fwhm / 2.3548)) ** 2)
y += np.random.default_rng(3).normal(scale=0.005, size=x.size)

result = detect_peaks(y, x=x, baseline="airpls", smooth=11, max_width=120)

print(result.table(digits=4))
# index     x  height  prominence  width   area    snr
# -----  ----  ------  ----------  -----  -----  -----
#   120   520  0.9953       0.995  24.12  25.33  164.4
#   505   905   1.596       1.598  22.05  37.22  263.6
#   900  1300  0.8638      0.8723   43.8  40.02  142.7
#  1300  1700   1.179       1.187  35.88  44.55  194.7

result.positions()          # array([ 520.,  905., 1300., 1700.])
result.to_csv("peaks.csv")  # the full table
```

The truth was 520/905/1300/1700 with FWHM 24/22/45/36, so the positions are exact, the
widths are within 3 % and the areas within 4 %.

## Features

| Area | What is provided |
| --- | --- |
| **Smoothing** | Savitzky–Golay filter and coefficients (any derivative order), moving average, Gaussian smoothing, five edge-padding modes |
| **Baseline** | airPLS, asymmetric least squares, SNIP, morphological opening, rolling ball, iterative polynomial, plus a unified `correct_baseline()` switch |
| **Detection** | `find_peaks()` (SciPy-compatible local maxima + property filters), `find_peaks_derivative()` (first/second derivative), `find_peaks_cwt()` (Ricker-wavelet ridge tracking) |
| **Metrics** | robust noise estimation, SNR, topological prominence, widths/FWHM at any relative height, peak areas |
| **Fitting** | Gaussian / Lorentzian / pseudo-Voigt / Voigt profiles, a self-contained Levenberg–Marquardt solver with parameter errors and box constraints, simultaneous multi-peak fitting with a shared background |
| **Pipeline** | `detect_peaks()` chains baseline → smoothing → detection → metrics → optional fitting and returns `Peak` objects |

Everything is written against NumPy arrays only. Even the linear algebra for the
baseline smoothers — a banded LDLᵀ factorisation — is implemented here, which keeps the
package dependency-free while staying fast for spectra with tens of thousands of points
(a 20 000-point airPLS solve takes ~0.1 s).

## Installation

The package lives in `src/`; either install it or put it on the path:

```bash
pip install -e .            # runtime requirement: numpy
pip install -e ".[test]"    # + pytest and scipy for the test suite
```

```python
import spectral_peaks
print(spectral_peaks.__version__)   # '0.1.0'
```

## Quick start

### 1. Detect peaks

```python
from spectral_peaks import detect_peaks

result = detect_peaks(
    y, x=x,
    method="prominence",        # "prominence" | "derivative" | "cwt"
    baseline="airpls",          # any name from BASELINE_METHODS, or a precomputed array
    baseline_kwargs={"lam": 1e5},
    smooth=11,                  # Savitzky-Golay window, in samples
    prominence=0.05,            # filters, as in scipy.signal.find_peaks
    distance=5,
    fit_model="gaussian",       # optional: refine every peak by fitting
)

for peak in result:             # PeakResult is iterable, indexable and sized
    print(peak.x, peak.height, peak.fwhm, peak.area, peak.snr)
```

Each `Peak` carries `index`, `x`, `height`, `prominence`, `left_base`, `right_base`,
`width` (= FWHM at `rel_height`), `left_ip`, `right_ip`, `area`, `snr` and an optional
`fit` dict. Widths and areas come out in *x units* whenever an `x` axis is supplied, and
in samples otherwise.

When no explicit `height`, `prominence` or `threshold` is given, `detect_peaks` applies
a default five-sigma prominence gate based on the robust MAD scale of the processed
signal, which keeps noise ripples and baseline-correction artefacts out of the result.

### 2. Correct a baseline

```python
from spectral_peaks import correct_baseline

corrected, baseline = correct_baseline(y, method="airpls", return_baseline=True)
snip = correct_baseline(y, method="snip", iterations=40)
ball = correct_baseline(y, method="rolling_ball", radius=40)
```

Available methods: `airpls`, `als`, `snip`, `morphological`, `rolling_ball`,
`polynomial`.

### 3. Fit peaks, including blends

```python
from spectral_peaks import fit_peak, fit_peaks, fit_region

# one peak inside a local window (with a linear background)
fit = fit_peak(x, y, center=905.0, fwhm=22.0, model="pseudo_voigt", baseline_order=1)
fit.peaks()[0]
# {'amplitude': 1.6006, 'center': 905.005, 'sigma': 9.338, 'gamma': 12.05,
#  'eta': 0.003, 'fwhm': 21.996, 'area': 37.5,
#  'amplitude_err': 0.004, 'center_err': 0.012, ...}
# (the data was a Gaussian of height 1.6 and FWHM 22 with a 0.005 noise level)
fit.reduced_chi2, fit.converged     # (2.87e-05, True)

# two overlapping peaks fitted *together* -> deconvolution
fit = fit_region(x, y, centers=[100.0, 112.0], model="gaussian", baseline_order=1)
fit.curve()          # peaks + background
fit.background()     # just the polynomial background

# refine a whole detection result in place
fit_peaks(result, model="gaussian")
```

The fitter reports parameter standard errors, `reduced_chi2` and a `converged` flag,
needs no analytic Jacobian (central differences), and applies box constraints
(positive amplitudes and widths, centre inside the fitted region, `eta` in `[0, 1]`)
so that the long Lorentzian/Voigt tails cannot run away into a degenerate solution.

## How the algorithms work

**Savitzky–Golay** coefficients are obtained by projecting `y` onto a polynomial basis
with a pseudo-inverse of the Vandermonde matrix, so smoothing *and* any derivative order
are supported. Coefficients and filter output match SciPy to `1e-10` in the interior;
the edges are handled by the chosen padding mode (`reflect` by default), which is the
only place where edge behaviour can differ from other implementations.

**Baselines.** airPLS and ALS are iteratively reweighted Whittaker smoothers; each
iteration solves

    (diag(w) + lambda * DᵀD) z = w ⊙ y

with `D` the difference operator of the chosen order. That system is symmetric,
positive definite and banded, so it is solved here with a banded LDLᵀ factorisation in
`O(n)` per iteration instead of the `O(n³)` a dense solve would need — which is what
makes airPLS practical on long spectra. SNIP clips peaks iteratively in the
log-log-square-root domain; the morphological and rolling-ball methods are
non-flat structuring-element openings; the polynomial method refits with points above
the current fit rejected.

**Prominence and widths** follow the SciPy definitions exactly: the prominence is the
height above the higher of the two bases found by walking towards the nearest higher
sample (when two samples tie, the *later* one is the base, as in SciPy), and widths are
the interpolated contour crossings at `y_peak - rel_height * prominence`. On random and
synthetic signals the peak indices and every property array agree with
`scipy.signal` bit-for-bit (see the test suite).

**Derivative detection** differentiates the Savitzky–Golay fit (numerically stable, in
contrast to differentiating raw data) and looks for downward zero crossings of `d1`
with `d2 < 0`, or for minima of `d2` (which resolves shoulders that show no separate
first-derivative crossing). Positions are refined by linear interpolation of the
crossing, giving sub-sample accuracy (a peak at sample 500.37 is located to ~1e-3
samples in the tests), and the reported index is snapped to the true apex sample.

**Wavelet detection** computes a Ricker (Mexican-hat) CWT and tracks *ridge lines*
across scales. Ridge points must exceed `snr` times the per-scale noise level, estimated
robustly as `1.4826 · MAD(|CWT|)`; a line is accepted when it spans at least
`min_length` scales and keeps its position while the scale changes — the side lobes of a
broad peak drift outwards with the scale and are rejected that way. For a Gaussian the
strongest response occurs at a scale close to the FWHM in samples, which calibrates the
`widths` range (roughly half to twice the widths of interest).

**Fitting** is a plain Levenberg–Marquardt (damped Gauss-Newton) iteration with a
central-difference Jacobian and optional box projection. Models are summed peaks plus a
polynomial background evaluated in the normalised coordinate
`(x - x̄) / std(x)`, which keeps the normal equations well conditioned for x axes
spanning orders of magnitude. Covariance is `(JᵀJ)⁻¹ s²`, so the reported errors are the
usual 1-sigma uncertainties.

## Accuracy (as measured by the test suite)

| Check | Result |
| --- | --- |
| `find_peaks`, `peak_prominences`, `peak_widths` vs `scipy.signal` | identical peak indices and property arrays (9 filter combinations × 25 signals, plus widths at `rel_height` 0.25/0.5/0.9 and two `wlen` values, ties included) |
| Savitzky–Golay coefficients/filter vs `scipy.signal` | agree to `1e-10` (interior) |
| Banded solver vs dense  `numpy.linalg.solve` | `1e-13` relative error; 0.1 s for a 20 000-point airPLS step |
| Peak positions, 6-peak synthetic spectrum (noise σ = 0.002, SNR 180–580, exponential background + airPLS) | exact to ≤ 0.3 samples with all three detectors, no false positives |
| Peak widths (FWHM) on the same spectrum | within 1.7 % |
| Peak areas (visible/x-units) | within 6 % |
| Deriving a peak at sample 500.37 | interpolated position 500.370 ± 0.001 |
| Separating two Gaussians 12 samples apart (σ ≈ 6) | centres recovered to 0.12 samples, amplitudes to 2 % |
| A Gaussian fitted with the Gaussian profile | all parameters within one standard error |

Run the suite with:

```bash
pytest              # 173 tests, scipy-dependent ones skipped automatically
pytest -q -k scipy  # only the cross-validation tests
```

## Conventions and caveats

* **x units vs samples.** `Peak.x`, `Peak.width` and `Peak.area` are expressed in x units
  and `Peak.index`, `left_ip`, `right_ip`, `left_base`, `right_base` in samples. Without
  an x axis the two coincide. `x` must be strictly increasing.
* **`rel_height`** counts *down* from the maximum in units of the prominence: 0.5 gives
  the FWHM, 0.25 a width at 75 % height (narrower), 0.95 a width near the base (wider).
* **`Peak.area`** integrates the processed signal over `±area_span/2 × width`
  (default `±1.5 FWHM`) above the straight line joining the window ends. It is accurate
  to a few percent for resolved Gaussian-like peaks and truncates the long tails of
  Lorentzian/Voigt profiles; for blended peaks take the area from
  `fit_peaks()` instead, which uses the analytical model area.
* **Height ties.** `distance` filtering visits equal-height peaks in the order produced
  by `numpy.argsort` (the same tie-break as SciPy). Pass your own ordering rules if you
  need something else.
* **Default gate.** Without explicit filters, `detect_peaks` keeps peaks with a
  prominence above `5 × 1.4826 · MAD(processed)`. Raise it if your baseline correction
  leaves artefacts, lower it for very weak peaks. An imperfect baseline can also leave
  behind *broad, low* residual bumps that no prominence threshold removes (they are
  genuinely wide); use `max_width` to reject them, as in the example above.
* **Rolling ball.** The ball is isotropic in data units, so the bias of the result grows
  roughly quadratically with `radius` where the background curves; keep the radius a few
  times the half-width of the widest peak and check the outcome on a peak-free region.
* **Morphological opening** with a flat window is biased by about
  `half_window × |background slope|` on sloped backgrounds (this is a property of the
  method, and the test suite pins it down).
* **Voigt** profiles require a uniformly spaced grid with an odd number of samples
  (the profile is evaluated as an FFT convolution).
* **Fitting needs a sensible starting point.** Initial guesses come from the detection
  result, which is normally accurate enough; `fit_region` accepts explicit centres,
  widths and amplitudes when it is not.

## Repository layout

```
src/spectral_peaks/
    peaks.py        detectors, prominence/widths, detect_peaks pipeline
    baseline.py     SNIP / airPLS / ALS / morphological / rolling ball / polynomial
    smoothing.py    Savitzky-Golay, moving average, Gaussian smoothing
    metrics.py      noise, SNR, areas, FWHM conversions
    fitting.py      profile models + Levenberg-Marquardt + multi-peak fitting
    types.py        Peak / PeakResult containers
    _banded.py      banded LDL^T linear algebra for the Whittaker smoothers
tests/              173 tests, including SciPy cross-validation
```

Not included (yet): a CLI, file readers/writers for instrument formats, a synthetic
spectrum generator, and plotting helpers. `PeakResult.to_csv()` covers simple export in
the meantime.

## References

* A. Savitzky, M.J.E. Golay, *Smoothing and Differentiation of Data by Simplified Least
  Squares Procedures*, Anal. Chem. 36 (1964) 1627.
* C.G. Ryan et al., *SNIP, a statistics-sensitive background treatment for the
  quantitative analysis of PIXE spectra*, Nucl. Instrum. Methods B 34 (1988) 396.
* P.H.C. Eilers, H.F.M. Boelens, *Baseline correction with asymmetric least squares
  smoothing*, Leiden University (2005).
* Z.-M. Zhang, S. Chen, Y.-Z. Liang, *Baseline correction using adaptive iteratively
  reweighted penalized least squares*, Analyst 135 (2010) 1138.
* S. Sternberg, *Biomedical image processing*, IEEE Computer 16 (1983) 22 (morphological
  opening; the rolling-ball background estimator).
* M.A. Kneen, H.J. Annegarn, *Algorithm for fitting XRF, SEM and PIXE X-ray spectra
  backgrounds*, Nucl. Instrum. Methods B 109 (1996) 209.
* P. Du, W.A. Kibbe, S.M. Lin, *Improved peak detection in mass spectrum by incorporating
  continuous wavelet transform-based pattern matching*, Bioinformatics 22 (2006) 2059.
* SciPy's `find_peaks`/`peak_prominences`/`peak_widths` (BSD-3) served as the reference
  for the local-maxima and prominence conventions; the implementation here is
  independent NumPy code.

## License

MIT.

