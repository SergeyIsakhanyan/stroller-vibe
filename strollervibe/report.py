"""Write CSV tables, PNG charts and a single self-contained HTML report."""
from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

import pandas as pd

from . import plots
from .analysis import findings, group_table, walks_table

WALK_COLS = {
    "walk_id": "Walk", "surface": "Surface", "stroller": "Stroller", "load": "Load", "speed_kmh": "Speed km/h",
    "aw_z": "a_wz", "av": "a_v", "mtvv_z": "MTVV_z", "vdv_z": "VDV_z", "crest_z": "Crest",
    "peak_hz": "Peak Hz", "moving_s": "Rolling s", "fs_hz": "Rate Hz", "comfort_ref": "Annex C (adult)",
}
GROUP_COLS = {
    "load": "Load", "surface": "Surface", "stroller": "Stroller", "walks": "Walks", "aw_z_median": "a_wz median",
    "aw_z_min": "min", "aw_z_max": "max", "av_median": "a_v median", "mtvv_z_median": "MTVV_z median",
    "peak_hz_median": "Peak Hz", "speed_kmh_median": "Speed km/h", "rolling_min": "Rolling min",
}

CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #f9f9f7; color: #0b0b0b; font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 40px 0 10px; } h3 { font-size: 16px; margin: 0 0 8px; }
.sub { color: #52514e; margin: 0 0 20px; }
.banner { background: #fdecea; border: 1px solid #d03b3b; color: #7a1f1f; padding: 10px 14px; border-radius: 8px; margin: 16px 0; font-weight: 600; }
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.stat { background: #fcfcfb; border: 1px solid rgba(11,11,11,.10); border-radius: 10px; padding: 10px 14px; min-width: 120px; }
.stat b { display: block; font-size: 22px; } .stat span { color: #52514e; font-size: 13px; }
.card { background: #fcfcfb; border: 1px solid rgba(11,11,11,.10); border-radius: 10px; padding: 14px; margin: 12px 0; overflow-x: auto; }
img { max-width: 100%; height: auto; display: block; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
th, td { padding: 5px 10px; border-bottom: 1px solid #e1e0d9; text-align: right; }
th { color: #52514e; font-weight: 600; background: #f9f9f7; position: sticky; top: 0; }
.t1 :is(td,th):nth-child(-n+1), .t2 :is(td,th):nth-child(-n+2), .t3 :is(td,th):nth-child(-n+3), .t4 :is(td,th):nth-child(-n+4) { text-align: left; }
ul { padding-left: 20px; } li { margin: 4px 0; }
details { background: #fcfcfb; border: 1px solid rgba(11,11,11,.10); border-radius: 10px; margin: 8px 0; padding: 8px 14px; }
summary { cursor: pointer; font-weight: 600; }
.warn { color: #8a4b00; } .muted { color: #898781; font-size: 13px; }
code { background: #f0efec; padding: 1px 4px; border-radius: 4px; }
"""


def _img(png: bytes, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def _table(df: pd.DataFrame, cols: dict) -> str:
    d = df[[c for c in cols if c in df.columns]].rename(columns=cols).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            fmt = "{:.0f}" if c in ("Rolling s", "Rate Hz", "Walks") else "{:.2f}"
            d[c] = d[c].map(lambda v, fmt=fmt: "" if pd.isna(v) else fmt.format(v))
        else:
            d[c] = d[c].astype(str).str.replace("_", " ")
    n_text = sum(1 for c in cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]) and c != "comfort_ref")
    return f"<div class='t{min(n_text, 4)}'>" + d.to_html(index=False, border=0, escape=True) + "</div>"


def write_outputs(results, problems, out_dir, title="Stroller vibration report"):
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    walks = walks_table(results)
    if walks.empty:
        raise ValueError("No walks could be analysed:\n  " + "\n  ".join(problems))
    by_surface = group_table(walks, ("load", "surface"))
    walks.to_csv(out / "walks_summary.csv", index=False)
    by_surface.to_csv(out / "surface_summary.csv", index=False)
    multi_stroller = walks["stroller"].nunique() > 1
    by_stroller = group_table(walks, ("load", "surface", "stroller")) if multi_stroller else None
    if by_stroller is not None:
        by_stroller.to_csv(out / "stroller_summary.csv", index=False)
    else:
        (out / "stroller_summary.csv").unlink(missing_ok=True)  # left over from an earlier run

    cmp_png = plots.comparison_figure(walks)
    (out / "plots" / "comparison_by_surface.png").write_bytes(cmp_png)
    speed_png = plots.speed_figure(walks)
    if speed_png:
        (out / "plots" / "speed.png").write_bytes(speed_png)
    else:
        (out / "plots" / "speed.png").unlink(missing_ok=True)
    walk_pngs = []
    for r in results:
        png = plots.walk_figure(r)
        walk_pngs.append(png)
        (out / "plots" / f"{_safe(r.walk_id)}.png").write_bytes(png)

    synthetic = walks["synthetic"].str.lower().isin(["yes", "true", "1"]).any()
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"]
    parts.append(f"<h1>{html.escape(title)}</h1><p class='sub'>Generated {dt.date.today().isoformat()} by stroller-vibe · ISO 2631-1:1997 weighting</p>")
    if synthetic:
        parts.append("<div class='banner'>SYNTHETIC DEMO DATA — these walks were generated by a script, not measured. "
                     "Do not cite or share these numbers.</div>")
    rolling_min = walks["moving_s"].sum() / 60
    stats = [(len(walks), "walks"), (f"{rolling_min:.0f}", "minutes rolling"),
             (walks["surface"].nunique(), "surfaces"), (walks["stroller"].nunique(), "strollers"),
             (f"{walks['aw_z'].median():.2f}", "median a_wz, m/s²")]
    parts.append("<div class='stats'>" + "".join(f"<div class='stat'><b>{v}</b><span>{k}</span></div>" for v, k in stats) + "</div>")
    if problems:
        parts.append("<h2>Files that need attention</h2><ul class='warn'>" + "".join(f"<li>{html.escape(p)}</li>" for p in problems) + "</ul>")

    parts.append("<h2>What this data says</h2><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in findings(walks)) + "</ul>")
    parts.append("<p class='muted'>Small samples: treat differences as leads to test with more repeats, not conclusions.</p>")
    parts.append("<h2>By surface</h2><div class='card'>" + _img(cmp_png, "Vertical vibration by surface") + "</div>")
    parts.append("<div class='card'>" + _table(by_surface, GROUP_COLS) + "</div>")
    if by_stroller is not None:
        parts.append("<h2>By surface and stroller</h2><div class='card'>" + _table(by_stroller, GROUP_COLS) + "</div>")
    if speed_png:
        parts.append("<h2>Speed</h2><div class='card'>" + _img(speed_png, "Vibration vs speed") + "</div>")
    parts.append("<h2>All walks</h2><div class='card'>" + _table(walks, WALK_COLS) + "</div>")
    parts.append("<p class='muted'>All accelerations in m/s² (VDV in m/s<sup>1.75</sup>). Metrics exclude detected stops; "
                 "the full numbers per walk are in <code>walks_summary.csv</code>.</p>")

    parts.append("<h2>Each walk</h2>")
    for r, png in zip(results, walk_pngs):
        warn = "".join(f"<li class='warn'>{html.escape(w)}</li>" for w in r.warnings)
        meta_bits = ", ".join(f"{k}: {html.escape(str(v))}" for k, v in r.meta.items()
                              if k in ("date", "city", "mode", "placement", "notes") and str(v).strip())
        parts.append(f"<details><summary>{html.escape(r.walk_id)} — {html.escape(str(r.meta.get('surface', '')).replace('_', ' '))} "
                     f"· a_wz {r.metrics['aw_z']:.2f}</summary>"
                     f"<p class='muted'>{meta_bits}</p>{'<ul>' + warn + '</ul>' if warn else ''}"
                     f"{_img(png, r.walk_id)}</details>")

    parts.append(METHOD_HTML)
    parts.append("</main></body></html>")
    (out / "report.html").write_text("".join(parts), encoding="utf-8")
    return out / "report.html", walks


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


METHOD_HTML = """
<h2>Method</h2>
<ul>
<li>Phone accelerometer recordings (phyphox, 'Acceleration with g') are resampled to a uniform rate. The part where the phone lay still in its walk orientation is found from the direction of gravity; handling at both ends is cut, plus a few seconds of margin.</li>
<li>Axes are rotated so z is perpendicular to the surface the phone lies on. Vertical (z) is weighted with W<sub>k</sub>, horizontal (x, y) with W<sub>d</sub> (ISO 2631-1:1997, Annex A, exact analogue responses applied in the frequency domain).</li>
<li>Stops (no vibration above 2 Hz for 3 s or more) are excluded from RMS values, so a wait at a crossing doesn't dilute the result.</li>
<li><b>a_wz</b>: frequency-weighted RMS vertical acceleration. <b>a_v</b>: vibration total value, √(a_wx² + a_wy² + a_wz²). <b>MTVV</b>: worst 1-second running RMS. <b>VDV</b>: vibration dose value (grows with duration, sensitive to shocks). <b>Crest</b>: peak ÷ RMS; past 9, RMS alone may understate peaks.</li>
</ul>
<h2>Caveats</h2>
<ul>
<li>There is no vibration standard for infants. ISO 2631-1 and its Annex C comfort bands describe adults; they are used here only as a common reference scale, as in published stroller studies.</li>
<li>Annex C bands formally apply to the total value a_v; the comparison chart shows the vertical a_wz against them because vertical vibration dominates in strollers.</li>
<li>Phones typically sample at 100–500 Hz. At 100 Hz, content above 50 Hz is not captured (the ISO band extends to 80 Hz).</li>
<li>A phone on the mattress measures the mattress near the baby, not the baby's body. Runs with the phone under a 5 kg load measure the body-mattress interface, closer to ISO seat-pad practice.</li>
<li>Phone accelerometers are not calibrated instruments. The report checks each phone's reading of gravity (should be 9.81 m/s²).</li>
</ul>
"""
