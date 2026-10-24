"""Turn one recording into ISO 2631-1 metrics, and a folder of recordings into tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from . import metrics as M
from .loader import G, SUPPORTED, load_recording
from .weighting import apply_weighting

META_COLUMNS = [
    "walk_id", "file", "date", "city", "surface", "stroller", "mode", "load",
    "placement", "speed_kmh", "distance_m", "trim_start_s", "trim_end_s", "notes", "synthetic",
]


@dataclass
class Settings:
    trim_s: float = 3.0            # removed inside the stable part, after the phone was placed / before it was picked up
    trim_no_gravity_s: float = 5.0  # used when the export has no gravity ("without g"), so placement can't be detected
    settle_s: float = 2.0          # filter settling at each edge
    detect_stops: bool = True
    stop_threshold: float = 0.05   # m/s^2: 1-s RMS of the >2 Hz vector below this = stroller not moving
    min_stop_s: float = 3.0
    max_tilt_deg: float = 20.0


@dataclass
class WalkResult:
    walk_id: str
    meta: dict
    metrics: dict
    warnings: list = field(default_factory=list)
    series: dict = field(default_factory=dict)


# ---------------------------------------------------------------- preparation

def resample_uniform(t, acc):
    """Phones deliver jittery timestamps; put the samples on a uniform grid."""
    dt = np.diff(t)
    med = float(np.median(dt))
    fs_native = 1.0 / med
    fs = float(round(fs_native)) if fs_native >= 20 else fs_native
    n = int(np.floor(t[-1] * fs)) + 1
    tu = np.arange(n) / fs
    accu = np.column_stack([np.interp(tu, t, acc[:, i]) for i in range(3)])
    gap_s = float(dt[dt > max(5 * med, 0.05)].sum())
    return tu, accu, fs, gap_s


def stable_region(acc, fs, max_angle_deg=10.0, min_len_s=3.0):
    """First and last sample where the phone lies still in its walk orientation,
    or None if no such stretch exists.

    Placing the phone and picking it up change the direction of gravity in the
    phone's frame; everything outside the stable part is handling, not walking.
    """
    win = max(1, int(round(fs)))
    g_lp = ndimage.uniform_filter1d(acc, size=win, axis=0, mode="nearest")
    mag = np.linalg.norm(g_lp, axis=1)
    dirs = g_lp / np.maximum(mag, 1e-9)[:, None]
    ref = np.median(dirs, axis=0)
    ref /= np.linalg.norm(ref)
    ang = np.degrees(np.arccos(np.clip(dirs @ ref, -1, 1)))
    ok = (ang < max_angle_deg) & (np.abs(mag - G) < 0.3 * G)
    runs = _runs(ok)
    long_runs = [(a, b) for a, b in runs if b - a >= min_len_s * fs]
    if not long_runs:
        return None
    return long_runs[0][0], long_runs[-1][1]


def rotation_to_z(g):
    """Rotation matrix that turns the gravity direction `g` into +z (smallest rotation)."""
    gh = np.asarray(g, float) / np.linalg.norm(g)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(gh, z)
    s = np.linalg.norm(v)
    c = float(gh @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def moving_mask(acc, fs, threshold, min_stop_s):
    """True where the stroller is rolling: >2 Hz vibration above the threshold."""
    nyq = fs / 2
    sos = signal.butter(4, 2.0 / nyq, btype="highpass", output="sos")
    hp = signal.sosfiltfilt(sos, acc - acc.mean(axis=0), axis=0)
    vec = np.sqrt(sum(M.running_rms(hp[:, i], fs, 1.0) ** 2 for i in range(3)))
    moving = vec > threshold
    for a, b in _runs(~moving):
        if b - a < min_stop_s * fs:
            moving[a:b] = True
    return moving


def _runs(mask):
    """(start, end) index pairs of True runs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _num(value):
    try:
        v = float(str(value).strip())
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- one walk

def analyze_walk(path, meta: dict | None = None, settings: Settings | None = None) -> WalkResult:
    meta = dict(meta or {})
    st = settings or Settings()
    path = Path(path)
    rec = load_recording(path)
    walk_id = str(meta.get("walk_id") or "").strip() or path.stem
    warnings: list[str] = []

    t, acc, fs, gap_s = resample_uniform(rec.t, rec.acc)
    duration_total = float(t[-1])

    tilt = float("nan")
    g_measured = float("nan")
    if rec.includes_gravity:
        region = stable_region(acc, fs)
        if region is None:
            i0, i1 = 0, len(acc)
            start = st.trim_no_gravity_s
            end = duration_total - st.trim_no_gravity_s
            warnings.append(f"couldn't find where the phone lay still; cut a fixed {st.trim_no_gravity_s:.0f} s at each end, "
                            "so handling may be in the metrics. Was the phone moving around?")
        else:
            i0, i1 = region
            start = i0 / fs + st.trim_s
            end = i1 / fs - st.trim_s
        g_mean = acc[i0:i1].mean(axis=0)
        g_measured = float(np.linalg.norm(g_mean))
        tilt = float(np.degrees(np.arccos(abs(g_mean[2]) / g_measured)))
        acc = acc @ rotation_to_z(g_mean).T
        if tilt > st.max_tilt_deg:
            warnings.append(f"phone tilted {tilt:.0f}° from flat; corrected, but lay it flat next time")
        if abs(g_measured - G) > 0.25:
            warnings.append(f"phone reads gravity as {g_measured:.2f} m/s² (expected 9.81); its accelerometer may be off by {100 * (g_measured / G - 1):+.1f}%")
    else:
        start = st.trim_no_gravity_s
        end = duration_total - st.trim_no_gravity_s
        warnings.append("export has no gravity ('Acceleration without g'); assumed the phone lay flat. Use 'Acceleration with g'")

    user_start = _num(meta.get("trim_start_s"))
    user_end = _num(meta.get("trim_end_s"))
    if user_start is not None:
        start = max(start, user_start)
    if user_end is not None:
        end = min(end, duration_total - user_end)

    a = int(max(0, np.floor(start * fs)))
    b = int(min(len(t), np.ceil(end * fs)))
    if b - a < 20 * fs:
        raise ValueError(f"{path.name}: less than 20 s left after trimming (recording is {duration_total:.0f} s)")
    seg = acc[a:b]

    weights = {"x": "Wd", "y": "Wd", "z": "Wk"}
    settle = int(st.settle_s * fs)
    keep = slice(settle, len(seg) - settle)  # not [settle:-settle], which is empty when settle is 0
    w = {ax: apply_weighting(seg[:, i], fs, weights[ax])[keep] for i, ax in enumerate("xyz")}
    u_z = apply_weighting(seg[:, 2], fs, "band")[keep]
    seg = seg[keep]
    t_seg = t[a + settle:a + settle + len(seg)]
    seg_s = len(seg) / fs

    if st.detect_stops:
        moving = moving_mask(seg, fs, st.stop_threshold, st.min_stop_s)
    else:
        moving = np.ones(len(seg), bool)
    moving_s = float(moving.sum() / fs)
    if moving_s < 30:
        warnings.append(f"only {moving_s:.0f} s of rolling detected; check the stop threshold or record longer")
        if moving.sum() < fs:
            moving = np.ones(len(seg), bool)
            warnings.append("stop detection found no rolling at all, so metrics use the whole recording")

    aw = {ax: M.rms(w[ax], moving) for ax in "xyz"}
    av = float(np.sqrt(sum(v**2 for v in aw.values())))
    mtvv_z = M.mtvv(w["z"], fs)
    vdv_z = M.vdv(w["z"], fs)
    crest_z = M.crest_factor(w["z"], moving)
    aw_z_all = M.rms(w["z"])
    mtvv_ratio = mtvv_z / aw["z"] if aw["z"] > 0 else float("nan")
    # VDV covers the whole analysed part, stops included, so compare it with the RMS over that same span.
    vdv_ratio = vdv_z / (aw_z_all * seg_s**0.25) if aw_z_all > 0 else float("nan")
    # ISO 2631-1 6.3.1: past these limits basic RMS may understate peaks; report MTVV and VDV too.
    rms_may_understate = bool(crest_z > 9 or mtvv_ratio > 1.5 or vdv_ratio > 1.75)

    nper = int(min(len(seg), 4 * fs))
    f_psd, psd_u = signal.welch(u_z, fs, nperseg=nper)
    _, psd_w = signal.welch(w["z"], fs, nperseg=nper)
    band = f_psd >= 0.5
    peak_hz = float(f_psd[band][np.argmax(psd_w[band])]) if band.any() else float("nan")

    if fs < 60:
        warnings.append(f"sample rate {fs:.0f} Hz captures vibration only up to {fs / 2:.0f} Hz")
    if gap_s > 0.01 * duration_total:
        warnings.append(f"{gap_s:.1f} s of missing samples (app paused or backgrounded?)")

    speed = _num(meta.get("speed_kmh"))
    distance = _num(meta.get("distance_m"))
    if speed is None and distance is not None and moving_s > 0:
        speed = distance / moving_s * 3.6

    metrics = {
        "aw_z": aw["z"], "aw_x": aw["x"], "aw_y": aw["y"], "av": av,
        "comfort_ref": M.comfort_label(av),
        "mtvv_z": mtvv_z, "vdv_z": vdv_z, "crest_z": crest_z,
        "mtvv_ratio": mtvv_ratio, "vdv_ratio": vdv_ratio,
        "rms_may_understate": rms_may_understate,
        "arms_z_unweighted": M.rms(u_z, moving),
        "aw_z_incl_stops": aw_z_all,
        "peak_hz": peak_hz,
        "speed_kmh": speed,
        "analysed_s": seg_s, "moving_s": moving_s, "stopped_s": seg_s - moving_s,
        "recording_s": duration_total,
        "fs_hz": fs, "gap_s": gap_s, "tilt_deg": tilt, "g_measured": g_measured,
        "device": rec.device.get("deviceModel") or rec.device.get("Device model") or "",
    }
    series = {
        "t": t_seg, "az_w": w["z"], "run_rms_z": M.running_rms(w["z"], fs, 1.0),
        "moving": moving, "f": f_psd, "psd_u": psd_u, "psd_w": psd_w, "fs": fs,
    }
    return WalkResult(walk_id=walk_id, meta=meta, metrics=metrics, warnings=warnings, series=series)


# ---------------------------------------------------------------- a folder of walks

def read_meta(path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=META_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in META_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in ("surface", "load", "mode", "placement"):
        df[c] = df[c].str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    return df


def analyze_folder(data_dir, meta_path=None, settings: Settings | None = None):
    data_dir = Path(data_dir)
    meta = read_meta(meta_path)
    results, problems = [], []
    by_file = {}
    for _, r in meta.iterrows():
        name = Path(str(r["file"]).strip()).name
        if not name:
            continue
        if name in by_file:
            problems.append(f"{name}: listed more than once in walks.csv; used the first row")
            continue
        by_file[name] = r.to_dict()
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    seen = set()
    walk_ids = set()
    for p in files:
        if p.name in seen:
            problems.append(f"{p.relative_to(data_dir)}: another file is also called {p.name}; skipped (rename one)")
            continue
        seen.add(p.name)
        row = by_file.get(p.name)
        if row is None:
            row = {"surface": "unknown"}
            problems.append(f"{p.name}: not listed in walks.csv, analysed as surface 'unknown'")
        try:
            res = analyze_walk(p, row, settings)
        except Exception as e:  # keep going; report at the end
            problems.append(f"{p.name}: {e}")
            continue
        if res.walk_id in walk_ids:
            new_id = next(f"{res.walk_id}-{k}" for k in range(2, len(files) + 2) if f"{res.walk_id}-{k}" not in walk_ids)
            problems.append(f"{p.name}: walk_id '{res.walk_id}' is used twice; renamed to '{new_id}'")
            res.walk_id = new_id
        walk_ids.add(res.walk_id)
        results.append(res)
    for name in by_file:
        if name not in seen:
            problems.append(f"{name}: listed in walks.csv but no such file in {data_dir}")
    return results, problems


def walks_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"walk_id": r.walk_id}
        for c in ("file", "date", "city", "surface", "stroller", "mode", "load", "placement", "synthetic"):
            row[c] = str(r.meta.get(c, "") or "")
        row.update(r.metrics)
        row["warnings"] = " | ".join(r.warnings)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["speed_kmh"] = pd.to_numeric(df["speed_kmh"])  # all None (no speeds given) would be an object column
        df["surface"] = df["surface"].replace("", "unknown")
        df["load"] = df["load"].replace("", "unspecified")
        df["stroller"] = df["stroller"].replace("", "unspecified")
    return df


def group_table(walks: pd.DataFrame, keys=("load", "surface")) -> pd.DataFrame:
    if walks.empty:
        return pd.DataFrame()
    g = walks.groupby(list(keys), sort=False)
    out = g.agg(
        walks=("walk_id", "count"),
        aw_z_median=("aw_z", "median"),
        aw_z_min=("aw_z", "min"),
        aw_z_max=("aw_z", "max"),
        av_median=("av", "median"),
        mtvv_z_median=("mtvv_z", "median"),
        peak_hz_median=("peak_hz", "median"),
        speed_kmh_median=("speed_kmh", "median"),
        rolling_min=("moving_s", lambda s: s.sum() / 60),
    ).reset_index()
    return out.sort_values(list(keys[:-1]) + ["aw_z_median"]).reset_index(drop=True)


def findings(walks: pd.DataFrame) -> list[str]:
    """A few plain sentences that hold for this data set (with n stated)."""
    out = []
    if walks.empty:
        return out
    for load, sub in walks.groupby("load"):
        med = sub.groupby("surface")["aw_z"].agg(["median", "count"]).sort_values("median")
        if len(med) >= 2:
            lo, hi = med.index[0], med.index[-1]
            ratio = med.loc[hi, "median"] / med.loc[lo, "median"]
            out.append(
                f"Load '{load}': {hi.replace('_', ' ')} gave the highest median vertical vibration "
                f"({med.loc[hi, 'median']:.2f} m/s², n={int(med.loc[hi, 'count'])}), "
                f"{ratio:.1f}× {lo.replace('_', ' ')} ({med.loc[lo, 'median']:.2f} m/s², n={int(med.loc[lo, 'count'])})."
            )
        strollers = sub["stroller"].unique()
        if len(strollers) >= 2:
            # Compare strollers only on surfaces every stroller was tested on.
            common = set.intersection(*[set(sub[sub.stroller == s].surface) for s in strollers])
            if common:
                cmp = sub[sub.surface.isin(common)].groupby("stroller")["aw_z"].median().sort_values()
                out.append(
                    f"Load '{load}', on the {len(common)} surface(s) all strollers share: "
                    + ", ".join(f"{s} {v:.2f}" for s, v in cmp.items()) + " m/s² median vertical."
                )
    above = int((walks["av"] >= 0.315).sum())
    out.append(f"{above} of {len(walks)} walks reached a total value of 0.315 m/s² or more, where ISO 2631-1 Annex C stops calling vibration 'not uncomfortable' for adults.")
    peaky = int(walks["rms_may_understate"].sum())
    if peaky:
        out.append(f"{peaky} walk(s) cross an ISO 2631-1 limit (crest factor > 9, MTVV/a_w > 1.5 or VDV ratio > 1.75) "
                   "beyond which RMS alone may understate peaks; quote MTVV and VDV alongside a_wz for those.")
    return out
