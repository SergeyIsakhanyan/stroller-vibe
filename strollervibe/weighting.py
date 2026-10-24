"""ISO 2631-1:1997 frequency weightings (Annex A), applied in the frequency domain.

Each weighting is the product of analogue transfer functions:
  band limiting (2nd-order Butterworth high-pass f1, low-pass f2),
  acceleration-velocity transition (f3, f4, Q4),
  upward step (f5, Q5, f6, Q6).
Parameters are from ISO 2631-1:1997 Table A.1.

The weighting is applied by multiplying the FFT of the signal by the exact
analogue response H(j*2*pi*f). This keeps the response exact up to the Nyquist
frequency, which matters for phones sampling at ~100 Hz (the band-limiting
low-pass sits at 100 Hz, above Nyquist, where a bilinear-transform IIR filter
would distort the response).
"""
from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)

# None means the section is absent for that weighting.
PARAMS = {
    # Vertical (z) for seated / standing / recumbent comfort and health.
    "Wk": dict(f1=0.4, f2=100.0, f3=12.5, f4=12.5, Q4=0.63, f5=2.37, Q5=0.91, f6=3.35, Q6=0.91, K=1.0),
    # Horizontal (x, y).
    "Wd": dict(f1=0.4, f2=100.0, f3=2.0, f4=2.0, Q4=0.63, f5=None, Q5=None, f6=None, Q6=None, K=1.0),
    # Band limiting only (0.4-100 Hz): the "unweighted" reference.
    "band": dict(f1=0.4, f2=100.0, f3=None, f4=None, Q4=None, f5=None, Q5=None, f6=None, Q6=None, K=1.0),
}


def response(f, name: str) -> np.ndarray:
    """Complex frequency response of weighting `name` at frequencies `f` (Hz)."""
    p = PARAMS[name]
    s = 2j * np.pi * np.asarray(f, dtype=float)
    w1 = 2 * np.pi * p["f1"]
    w2 = 2 * np.pi * p["f2"]
    # Band limiting: 2nd-order Butterworth high-pass and low-pass (Q = 1/sqrt(2)).
    h = (s**2 / (s**2 + SQRT2 * w1 * s + w1**2)) * (w2**2 / (s**2 + SQRT2 * w2 * s + w2**2))
    if p["f3"] is not None:
        w3 = 2 * np.pi * p["f3"]
        w4 = 2 * np.pi * p["f4"]
        h = h * (1 + s / w3) / (1 + s / (p["Q4"] * w4) + (s / w4) ** 2)
    if p["f5"] is not None:
        w5 = 2 * np.pi * p["f5"]
        w6 = 2 * np.pi * p["f6"]
        # Low-frequency gain (w5/w6)^2, high-frequency gain 1.
        h = h * (s**2 + w5 / p["Q5"] * s + w5**2) / (s**2 + w6 / p["Q6"] * s + w6**2)
    return p["K"] * h


def magnitude(f, name: str) -> np.ndarray:
    return np.abs(response(f, name))


def apply_weighting(x: np.ndarray, fs: float, name: str, pad_s: float = 10.0) -> np.ndarray:
    """Return the weighted time history of acceleration `x` (1-D, uniform rate `fs`).

    The mean is removed first (it carries gravity), and the signal is zero-padded
    so the filter tail does not wrap around. Drop ~2 s at each edge before using
    the result for metrics.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n + int(pad_s * fs))))
    spec = np.fft.rfft(x - x.mean(), nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return np.fft.irfft(spec * response(freqs, name), nfft)[:n]
