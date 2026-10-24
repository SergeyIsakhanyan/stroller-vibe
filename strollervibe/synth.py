"""Synthetic phyphox-style recordings, for tests and the demo.

These are NOT measurements. Levels are invented to exercise the pipeline;
every walk written here is flagged synthetic=yes in walks.csv and the report
shows a banner.
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

from .loader import G
from .weighting import apply_weighting

FINE_FS = 1000.0

# Invented target levels (Wk-weighted vertical RMS at 4 km/h) and texture.
SURFACES = {
    "indoor_floor":  dict(level=0.12, joint_m=None),
    "smooth_asphalt": dict(level=0.30, joint_m=None),
    "paving_slabs":  dict(level=0.55, joint_m=0.40),
    "cobblestones":  dict(level=1.05, joint_m=0.14),
    "gravel":        dict(level=0.85, joint_m=None),
}


def _resonator(x, fs, f0, q):
    b, a = signal.iirpeak(f0 / (fs / 2), q)
    return signal.lfilter(b, a, x)


def synth_walk(surface, speed_kmh=4.0, duration_s=100.0, stroller_factor=1.0, fs=100.0,
               stop=None, tilt_deg=4.0, seed=0, include_gravity=True):
    """Return (t, acc[n,3]) as a phone would record it, including handling at both ends."""
    rng = np.random.default_rng(seed)
    spec = SURFACES[surface]
    n = int(duration_s * FINE_FS)
    tf = np.arange(n) / FINE_FS
    v = speed_kmh / 3.6

    # Road texture through the frame and mattress: two resonances, rolled off above ~25 Hz.
    rough = rng.standard_normal(n)
    z = _resonator(rough, FINE_FS, 4.5, 2.5) + 0.6 * _resonator(rough, FINE_FS, 11.0, 4.0)
    if spec["joint_m"]:
        hits = np.zeros(n)
        pos = np.cumsum(np.full(n, v / FINE_FS))
        idx = np.flatnonzero(np.diff(np.floor(pos / spec["joint_m"] * (1 + 0.05 * rng.standard_normal(n)))) > 0)
        hits[idx] = rng.uniform(0.5, 1.5, idx.size)
        z = z + 3.0 * _resonator(hits, FINE_FS, 9.0, 3.0)
    sos = signal.butter(2, 25 / (FINE_FS / 2), output="sos")
    z = signal.sosfilt(sos, z)
    fx = 0.5 * signal.sosfilt(sos, _resonator(rng.standard_normal(n), FINE_FS, 3.0, 2.0))
    fy = 0.35 * signal.sosfilt(sos, _resonator(rng.standard_normal(n), FINE_FS, 2.5, 2.0))

    # Scale so the Wk-weighted vertical RMS hits the invented target for this run.
    target = spec["level"] * stroller_factor * (speed_kmh / 4.0) ** 0.8 * rng.uniform(0.9, 1.1)
    k = target / np.sqrt(np.mean(apply_weighting(z, FINE_FS, "Wk") ** 2))
    dyn = np.column_stack([fx * k, fy * k, z * k])

    # Parent's gait (~1.8 Hz) reaches the stroller through the handle.
    gait = 2 * np.pi * 1.8 * tf
    dyn[:, 2] += 0.08 * np.sin(gait) + 0.03 * np.sin(2 * gait)
    dyn[:, 1] += 0.15 * np.sin(gait + 0.4)

    rolling = np.ones(n)
    if stop:
        a, b = int(stop[0] * FINE_FS), int((stop[0] + stop[1]) * FINE_FS)
        ramp = int(1.0 * FINE_FS)
        rolling[a:b] = 0
        rolling[a:a + ramp] = np.linspace(1, 0, ramp)
        rolling[b - ramp:b] = np.linspace(0, 1, ramp)
        dyn[a:b, 1] += 0.15 * np.sin(2 * np.pi * 0.7 * tf[a:b])  # gentle rocking while stopped
    dyn *= rolling[:, None]

    # World frame (z up) -> phone frame: the phone lies slightly tilted, so both
    # gravity and the (world-vertical) vibration appear rotated in its axes.
    acc = dyn + (np.array([0.0, 0.0, G]) if include_gravity else 0.0)
    th = np.radians(tilt_deg)
    if th > 0:
        axis = np.array([0.8, -0.6, 0.0])  # unit vector in the horizontal plane
        rot = Rotation.from_rotvec(axis * th).as_matrix()
        acc = acc @ rot.T

    # Handling: phone in hand for the first 4 s and last 3 s.
    for a, b in ((0, int(4 * FINE_FS)), (n - int(3 * FINE_FS), n)):
        hand = G * np.array([0.0, 0.7, 0.7]) if include_gravity else np.zeros(3)
        acc[a:b] = hand + 0.8 * rng.standard_normal((b - a, 3))

    # Phone timestamps: ~fs with jitter; sensor noise.
    dt = (1.0 / fs) * (1 + 0.03 * rng.standard_normal(int(duration_s * fs * 1.1)))
    t = np.cumsum(np.clip(dt, 0.3 / fs, None))
    t = t[t < tf[-1]]
    out = np.column_stack([np.interp(t, tf, acc[:, i]) for i in range(3)])
    out += 0.012 * rng.standard_normal(out.shape)
    return t - t[0], out


def write_phyphox_zip(path, t, acc, include_gravity=True, device_model="Synthetic phone"):
    """Write a zip laid out like a phyphox 'CSV (Comma, decimal point)' export."""
    label = "Acceleration" if include_gravity else "Linear Acceleration"
    header = ["Time (s)", f"{label} x (m/s^2)", f"{label} y (m/s^2)", f"{label} z (m/s^2)", "Absolute acceleration (m/s^2)"]
    cols = np.column_stack([t, acc, np.linalg.norm(acc, axis=1)])
    buf = io.StringIO()
    buf.write(",".join(f'"{h}"' for h in header) + "\n")
    np.savetxt(buf, cols, fmt="%.9E", delimiter=",")
    data = buf.getvalue()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Raw Data.csv", data)
        zf.writestr("meta/device.csv", f'"property","value"\n"deviceModel","{device_model}"\n')
        zf.writestr("meta/time.csv", '"event","experiment time","system time"\n"START","0.0","0"\n')


def make_demo_dataset(out_dir, seed=7):
    """A small invented dataset: 2 strollers x 4 surfaces x 2 speeds (rice bag) + 3 walks with a stop."""
    out_dir = Path(out_dir)
    data = out_dir / "data"
    rows = []
    rng = np.random.default_rng(seed)
    strollers = {"Stroller A (air tyres)": 0.8, "Stroller B (foam tyres)": 1.2}
    i = 0
    for stroller, factor in strollers.items():
        for surface in ("indoor_floor", "smooth_asphalt", "paving_slabs", "cobblestones"):
            for speed in (3.5, 5.0):
                i += 1
                name = f"walk{i:02d}.zip"
                t, acc = synth_walk(surface, speed, duration_s=float(rng.uniform(80, 120)),
                                    stroller_factor=factor, seed=int(rng.integers(1 << 30)))
                write_phyphox_zip(data / name, t, acc)
                rows.append(dict(walk_id=f"W{i:02d}", file=name, date="2026-10-21", city="Demo City",
                                 surface=surface, stroller=stroller, mode="bassinet", load="rice_5kg",
                                 placement="under_load", speed_kmh=speed, synthetic="yes"))
    for surface in ("smooth_asphalt", "paving_slabs", "gravel"):
        i += 1
        name = f"walk{i:02d}.zip"
        dur = float(rng.uniform(150, 200))
        t, acc = synth_walk(surface, 4.2, duration_s=dur, stroller_factor=0.8,
                            stop=(dur * 0.45, 35.0), seed=int(rng.integers(1 << 30)))
        write_phyphox_zip(data / name, t, acc)
        rows.append(dict(walk_id=f"W{i:02d}", file=name, date="2026-10-21", city="Demo City",
                         surface=surface, stroller="Stroller A (air tyres)", mode="bassinet", load="baby",
                         placement="mattress_foot", distance_m=round(4.2 / 3.6 * (dur - 50)),
                         notes="35 s stop at a crossing", synthetic="yes"))
    from .analysis import META_COLUMNS
    with open(out_dir / "walks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=META_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return data, out_dir / "walks.csv"
