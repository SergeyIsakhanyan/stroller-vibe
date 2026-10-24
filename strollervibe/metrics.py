"""Vibration metrics from ISO 2631-1:1997."""
from __future__ import annotations

import numpy as np

# ISO 2631-1 Annex C: likely reactions of (adult) passengers to the vibration
# total value a_v. The bands overlap in the standard; a value inside an overlap
# gets both labels.
COMFORT_BANDS = [
    (0.0, 0.315, "not uncomfortable"),
    (0.315, 0.63, "a little uncomfortable"),
    (0.5, 1.0, "fairly uncomfortable"),
    (0.8, 1.6, "uncomfortable"),
    (1.25, 2.5, "very uncomfortable"),
    (2.0, np.inf, "extremely uncomfortable"),
]


def rms(a, mask=None) -> float:
    a = np.asarray(a, float)
    if mask is not None:
        a = a[mask]
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")


def vdv(a, fs: float) -> float:
    """Vibration dose value, m/s^1.75. Grows with exposure time; sensitive to shocks."""
    a = np.asarray(a, float)
    return float((np.sum(a**4) / fs) ** 0.25)


def running_rms(a, fs: float, tau: float = 1.0) -> np.ndarray:
    """Linear-integration running RMS with a window of `tau` seconds (centred, same length)."""
    a = np.asarray(a, float)
    n = max(1, int(round(tau * fs)))
    if len(a) < n:
        return np.full_like(a, rms(a))
    c = np.cumsum(np.concatenate([[0.0], a**2]))
    win = (c[n:] - c[:-n]) / n  # len(a) - n + 1 values
    out = np.empty_like(a)
    left = n // 2
    out[left:left + len(win)] = np.sqrt(win)
    out[:left] = out[left]
    out[left + len(win):] = out[left + len(win) - 1]
    return out


def mtvv(a, fs: float, tau: float = 1.0) -> float:
    """Maximum transient vibration value: the worst 1-second running RMS."""
    return float(np.max(running_rms(a, fs, tau)))


def crest_factor(a, mask=None) -> float:
    a = np.asarray(a, float)
    if mask is not None:
        a = a[mask]
    r = rms(a)
    return float(np.max(np.abs(a)) / r) if a.size and r > 0 else float("nan")


def comfort_label(av: float) -> str:
    if not np.isfinite(av):
        return ""
    return " / ".join(label for lo, hi, label in COMFORT_BANDS if lo <= av < hi) or COMFORT_BANDS[-1][2]
