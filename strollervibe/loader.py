"""Load accelerometer recordings (phyphox exports or any time + x/y/z CSV)."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665
SUPPORTED = (".zip", ".csv", ".tsv", ".txt", ".xlsx")


@dataclass
class Recording:
    t: np.ndarray          # seconds, increasing, starts at 0
    acc: np.ndarray        # (n, 3) m/s^2 in the sensor frame (x, y, z)
    includes_gravity: bool
    source: str
    device: dict = field(default_factory=dict)


def load_recording(path) -> Recording:
    path = Path(path)
    suffix = path.suffix.lower()
    device: dict = {}
    if suffix == ".zip":
        table, device = _from_zip(path)
    elif suffix in (".csv", ".tsv", ".txt"):
        table = _read_table(path.read_bytes())
    elif suffix == ".xlsx":
        table = pd.read_excel(path)
    else:
        raise ValueError(f"{path.name}: unsupported file type (use a phyphox CSV export: {', '.join(SUPPORTED)})")
    return _to_recording(table, source=path.name, device=device)


def _from_zip(path: Path):
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        data_files = [n for n in names if n.lower().endswith((".csv", ".tsv", ".txt")) and "meta/" not in n.lower()]
        # phyphox names the main sheet "Raw Data.csv"; try it first.
        data_files.sort(key=lambda n: (0 if "raw data" in n.lower() else 1, n))
        table = None
        errors = []
        for name in data_files:
            try:
                candidate = _read_table(zf.read(name))
                _axis_columns(candidate)  # raises if it isn't an acceleration table
                table = candidate
                break
            except ValueError as e:
                errors.append(f"{name}: {e}")
        if table is None:
            raise ValueError(f"{path.name}: no accelerometer table found in the zip. " + "; ".join(errors))
        device = {}
        for name in names:
            if name.lower().endswith("meta/device.csv"):
                try:
                    meta = _read_table(zf.read(name))
                    if meta.shape[1] >= 2:
                        device = {str(k): str(v) for k, v in zip(meta.iloc[:, 0], meta.iloc[:, 1])}
                except Exception:
                    pass
    return table, device


def _read_table(raw: bytes) -> pd.DataFrame:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")
    header = lines[0]
    if "\t" in header:
        sep = "\t"
    elif ";" in header:
        sep = ";"
    else:
        sep = ","
    decimal = "."
    # phyphox offers "decimal comma" exports with tab or semicolon separators.
    if sep != "," and len(lines) > 1 and re.search(r"\d,\d", lines[1]):
        decimal = ","
    return pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal)


_AXIS_PATTERNS = [
    r"acceleration\s*{a}\b",          # phyphox: "Acceleration x (m/s^2)", "Linear Acceleration x (m/s^2)"
    r"\bacc(el)?[\s_\-]*{a}\b",       # acc_x, accel x
    r"\bgf{a}\b",                     # Physics Toolbox: gFx (in g)
    r"^a{a}\b",                       # ax
    r"^{a}\b",                        # x
]


def _axis_columns(df: pd.DataFrame):
    cols = [str(c) for c in df.columns]
    found = []
    for axis in "xyz":
        match = None
        for pat in _AXIS_PATTERNS:
            rx = re.compile(pat.format(a=axis))
            match = next((c for c in cols if rx.search(c.strip().lower())), None)
            if match:
                break
        if match is None:
            raise ValueError(f"no {axis}-axis acceleration column (columns: {cols})")
        found.append(match)
    return found


def _time_column(df: pd.DataFrame, axis_cols) -> str:
    for c in df.columns:
        name = str(c).strip().lower()
        if c not in axis_cols and ("time" in name or name in ("t", "t (s)", "timestamp")):
            return c
    return df.columns[0]


def _to_seconds(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        t = numeric.to_numpy(dtype=float)
        # Epoch milliseconds / microseconds / nanoseconds -> seconds.
        span = np.nanmax(t) - np.nanmin(t)
        med = np.nanmedian(np.abs(t))
        if med > 1e11 and span > 1e3:
            t = t / (1e9 if med > 1e17 else 1e6 if med > 1e14 else 1e3)
        return t
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.95:
        return (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy()
    raise ValueError("could not read the time column")


def _to_recording(df: pd.DataFrame, source: str, device: dict) -> Recording:
    axis_cols = _axis_columns(df)
    time_col = _time_column(df, axis_cols)
    t = _to_seconds(df[time_col])
    acc = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in axis_cols])
    ok = np.isfinite(t) & np.all(np.isfinite(acc), axis=1)
    t, acc = t[ok], acc[ok]
    order = np.argsort(t, kind="stable")
    t, acc = t[order], acc[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, acc = t[keep], acc[keep]
    if len(t) < 10:
        raise ValueError(f"{source}: fewer than 10 valid samples")

    names = " ".join(axis_cols).lower()
    in_g = "gf" in names or "(g)" in names
    if not in_g and "m/s" not in names:
        # No units in the header. Gravity has a steady direction, so its mean
        # vector keeps a length of ~1 g; vibration alone averages out near zero.
        in_g = 0.7 < np.linalg.norm(np.median(acc, axis=0)) < 1.3
    if in_g:
        acc = acc * G
    includes_gravity = np.median(np.linalg.norm(acc, axis=1)) > 0.5 * G
    return Recording(t=t - t[0], acc=acc, includes_gravity=bool(includes_gravity), source=source, device=device)
